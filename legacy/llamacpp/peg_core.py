from log_utils import format_log_entry, append_log
import pandas as pd
import concurrent.futures
from collections import defaultdict
import re
import threading
import time
import torch
import os

# ---------- Helpers ----------
_LOG_LOCK = threading.Lock()

def _append_log_threadsafe(log_path, log):
    with _LOG_LOCK:
        append_log(log_path, log)

def _safe_agent_call(agent_fn, prompt, max_new_tokens=256, **gen_kwargs):
    """
    尽量返回 list[{"generated_text": <str>}]
    - 支持 text-generation 风格
    - 支持 messages 风格
    - 正确展开 OpenAI-like 的多条 choices
    """
    try:
        # 常见 HF pipeline: 文本串
        resp = agent_fn(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)
    except TypeError:
        # 有些是 messages 风格
        resp = agent_fn([{"role": "user", "content": prompt}],
                        max_new_tokens=max_new_tokens, **gen_kwargs)
    except Exception as e:
        return [{"generated_text": f"[ERROR] {type(e).__name__}: {e}"}]

    # --- 统一归一化 ---
    out = []

    # OpenAI-like: resp 为 dict 且含 choices（可能多条）
    if isinstance(resp, dict) and "choices" in resp:
        choices = resp.get("choices", [])
        for ch in choices:
            txt = ch.get("text") or ch.get("message", {}).get("content", "")
            out.append({"generated_text": txt})
        return out or [{"generated_text": ""}]

    # HF pipeline 常见：list[dict] 或 dict
    if isinstance(resp, dict):
        resp = [resp]

    for r in resp if isinstance(resp, (list, tuple)) else [resp]:
        if isinstance(r, dict):
            if "generated_text" in r:
                out.append({"generated_text": r["generated_text"]})
            elif "text" in r:
                out.append({"generated_text": r["text"]})
            elif "choices" in r:
                # 小心：有些返回 list 里嵌 dict 带 choices（再展开）
                for ch in r.get("choices", []):
                    txt = ch.get("text") or ch.get("message", {}).get("content", "")
                    out.append({"generated_text": txt})
            else:
                out.append({"generated_text": str(r)})
        else:
            out.append({"generated_text": str(r)})

    return out or [{"generated_text": ""}]


def _extract_text(resp) -> str:
    """Accepts list/dict/str (from _safe_agent_call) and returns a single string."""
    if isinstance(resp, list):
        return "\n".join(
            [x.get("generated_text", "") if isinstance(x, dict) else str(x) for x in resp]
        )
    if isinstance(resp, dict):
        return resp.get("generated_text") or resp.get("text") or ""
    return str(resp) if resp is not None else ""

def _extract_yes_no(text: str) -> str:
    """Find the first standalone yes/no; default to 'no' if not found."""
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "no"

# ---------- Pipeline stages ----------
def initial_planning_parallel(prompt, agents, file, qid, round_num, log_path):
    plans = {}

    def generate_plan(name):
        plan_prompt = (
            "You are part of a multi-agent reasoning team. Your task is to independently develop a reasoning plan "
            "to answer the following question. Please outline your thought process step-by-step using clear logic.\n\n"
            f"Question:\n{prompt}\n\nYour Reasoning Plan:"
        )
        raw = _safe_agent_call(agents[name], plan_prompt, max_new_tokens=100)
        reply_full = _extract_text(raw).strip()

        # If upstream appended something like "Reasoning Plan:", try to keep only the plan body
        reply = reply_full.split("Reasoning Plan:")[-1].strip() if "Reasoning Plan:" in reply_full else reply_full

        log = format_log_entry(file, qid, round_num, name, "initial plan", reply)
        _append_log_threadsafe(log_path, log)
        return name, reply

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for name, reply in executor.map(generate_plan, agents.keys()):
            plans[name] = reply

    return plans

def peer_critiquing_parallel(plans, agents, file, qid, round_num, log_path, max_new_tokens=120):
    """
    Full peer review: every agent reviews all others' initial plans (no self-review).
    Returns: { target_agent_name: ["From reviewerA: ...", "From reviewerB: ...", ...], ... }
    """
    critiques = defaultdict(list)
    agent_names = list(agents.keys())

    if len(agent_names) <= 1:
        for n in agent_names:
            critiques[n] = []
        return critiques

    def generate_critique(reviewer, target):
        critique_prompt = (
            "Your peer has proposed the following reasoning plan:\n"
            "---\n" + plans.get(target, "") + "\n---\n"
            "Please provide constructive and specific feedback. Identify potential flaws, missing considerations, "
            "assumptions, data inconsistencies, or unclear steps. Be concise and helpful."
        )
        raw = _safe_agent_call(agents[reviewer], critique_prompt, max_new_tokens=max_new_tokens)
        out = _extract_text(raw).strip()

        log = format_log_entry(file, qid, round_num, reviewer, f"critique on {target}", out)
        _append_log_threadsafe(log_path, log)
        return target, f"From {reviewer}: {out}"

    pairs = [(r, t) for r in agent_names for t in agent_names if r != t]

    with concurrent.futures.ThreadPoolExecutor() as exe:
        for target, critique in exe.map(lambda p: generate_critique(*p), pairs):
            critiques[target].append(critique)

    for c in critiques:
        print(c)
        
    return critiques

def refine_plans_parallel(plans, critiques, agents, file, qid, round_num, log_path, max_new_tokens=300):
    refined = {}

    def refine(name):
        crit_list = critiques.get(name, [])
        critique_text = "\n".join(crit_list) if crit_list else "(no peer feedback received)"
        update_prompt = (
            "You initially proposed the following reasoning plan:\n"
            "<PLAN>\n" + plans.get(name, "").strip() + "\n</PLAN>\n\n"
            "Your peers provided the following feedback:\n"
            "<FEEDBACK>\n" + critique_text + "\n</FEEDBACK>\n\n"
            "Carefully analyze the feedback, decide which parts are insightful, and revise your plan to improve "
            "clarity, coverage, and correctness.\n\n"
            "Return ONLY the revised plan inside <REVISED_PLAN>...</REVISED_PLAN> with no preface."
        )
        raw = _safe_agent_call(agents[name], update_prompt, max_new_tokens=max_new_tokens)
        out = _extract_text(raw).strip()

        # Extract tag body if present
        if "<REVISED_PLAN>" in out and "</REVISED_PLAN>" in out:
            s = out.find("<REVISED_PLAN>") + len("<REVISED_PLAN>")
            e = out.find("</REVISED_PLAN>")
            new_plan = out[s:e].strip()
        else:
            new_plan = out

        log = format_log_entry(file, qid, round_num, name, "refined plan", new_plan)
        _append_log_threadsafe(log_path, log)
        return name, new_plan

    with concurrent.futures.ThreadPoolExecutor() as exe:
        for name, new_plan in exe.map(refine, agents.keys()):
            refined[name] = new_plan
    return refined

def answer_with_updated_plan_parallel(prompt, agents, plans, file, qid, round_num, log_path):
    answers, explanations = {}, {}

    def generate_answer(name):
        full_prompt = (
            f"You have revised your reasoning plan as follows:\n{plans.get(name, '')}\n\n"
            f"Now, use this updated reasoning to answer the following question:\n{prompt}\n\n"
            "Start your response with either 'yes' or 'no', then provide a concise justification."
        )
        raw = _safe_agent_call(agents[name], full_prompt, max_new_tokens=256)
        out = _extract_text(raw).strip()

        # Robust answer extraction
        answer = _extract_yes_no(out)

        log = format_log_entry(file, qid, round_num, name, "answer", out)
        _append_log_threadsafe(log_path, log)
        return name, answer, out

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for name, answer, out in executor.map(generate_answer, agents.keys()):
            answers[name] = answer
            explanations[name] = out

    return answers, explanations

def peer_review(answers):
    """
    answers: dict{name -> 'yes'/'no'}
    Returns (df, scores) where df has one row of raw answers + majority_vote,
    and scores is per-agent agreement (1.0 if matches majority, else 0.0).
    """
    ser = pd.Series(answers)
    majority_vote = ser.value_counts().idxmax() if not ser.empty else None
    scores = {k: float(v == majority_vote) for k, v in answers.items()}
    df = pd.DataFrame([answers])
    df["majority_vote"] = majority_vote
    return df, scores

# === NEW 1: 并行生成“每个模型的 initial_plan”（保留你现有日志格式）===
def initial_plans_parallel_all(prompt, agents, file, qid, round_num, log_path, max_new_tokens=220):
    """
    为每个模型各自起草 initial_plan（并行），返回 dict{name -> plan_text}
    同时写 [initial plan] 日志
    """
    plans = {}

    def _one(name):
        plan_prompt = (
            "You are part of a relay reasoning team. Draft a clear, step-by-step reasoning plan "
            "to answer the question below.\n\n"
            f"Question:\n{prompt}\n\nReturn ONLY the plan body (no headers):"
        )
        raw = _safe_agent_call(agents[name], plan_prompt, max_new_tokens=max_new_tokens)
        full = _extract_text(raw).strip()
        plan = full.split("Reasoning Plan:")[-1].strip() if "Reasoning Plan:" in full else full

        # log
        log = format_log_entry(file, qid, round_num, name, "initial plan", plan)
        _append_log_threadsafe(log_path, log)
        return name, plan

    with concurrent.futures.ThreadPoolExecutor() as exe:
        for name, plan in exe.map(_one, agents.keys()):
            plans[name] = plan
    return plans


# === NEW 2: “环形接力一轮”（同步轮）：上一棒的 plan -> 我 refine+answer -> 产出下一轮我传出的 plan ===
def relay_ring_round(
    prompt,
    agents,
    plans_state,
    order,
    file,
    qid,
    round_num,
    log_path,
    plan_tokens=220,
    answer_tokens=256,
    n_samples=10,              # 新增：一次性生成的样本数（Top-K）
    do_sample=True,            # 采样开关（默认 True）
    temperature=0.7,
    top_p=0.95
):
    """
    params:
      - plans_state: dict{model_name -> plan_text}
      - order: 模型固定顺序列表
    返回:
      - next_plans_state: dict{model -> refined_plan_this_round}
      - answers: dict{model -> List[str]}  # 每个模型本轮的 top-n yes/no 列表
      - explanations: dict{model -> List[str]}  # 每个模型本轮的 top-n 原始回答
    """
    names = list(order)

    # 拍快照：本轮每个模型要接收的“上一棒”计划
    incoming_map = {}
    for idx, name in enumerate(names):
        prev_name = names[(idx - 1) % len(names)]
        incoming_map[name] = plans_state.get(prev_name, "")

    next_plans_state = {}
    answers, explanations = {}, {}

    def _work(name):
        incoming_plan = incoming_map[name]

        # === refine 上一棒的 plan ===
        refine_prompt = (
            "You are the next agent in a relay. Refine the previous agent's plan to improve clarity, coverage, "
            "and correctness. Return ONLY the revised plan inside <REVISED_PLAN>...</REVISED_PLAN>.\n\n"
            "<PREV_PLAN>\n" + incoming_plan + "\n</PREV_PLAN>"
        )
        raw_ref = _safe_agent_call(
            agents[name], refine_prompt,
            max_new_tokens=plan_tokens
        )
        out_ref = _extract_text(raw_ref).strip()
        if "<REVISED_PLAN>" in out_ref and "</REVISED_PLAN>" in out_ref:
            s = out_ref.find("<REVISED_PLAN>") + len("<REVISED_PLAN>")
            e = out_ref.find("</REVISED_PLAN>")
            refined_plan = out_ref[s:e].strip()
        else:
            refined_plan = out_ref

        # 记录 refined plan
        _append_log_threadsafe(log_path, format_log_entry(file, qid, round_num, name, "refined plan", refined_plan))

        # === 基于 refined_plan 做一次性 Top-n 采样回答 ===
                # === 基于 refined_plan 做一次性 Top-n 采样回答 ===
                # === 基于 refined_plan：逐次独立生成 10 次（完全分开调用） ===
        ans_prompt = (
            f"You will now answer based on this plan:\n{refined_plan}\n\n"
            f"Question:\n{prompt}\n\n"
            "Start your response with either 'yes' or 'no', then give a concise justification."
        )

        pipe_device = getattr(agents[name], "device", None)
        device_str = "cuda" if (pipe_device is not None and getattr(pipe_device, "type", "") == "cuda") else "cpu"
        gens = _make_generators(n_samples, device_str=device_str)

        texts = []
        for i, g in enumerate(gens, 1):
            # 单独调用一次（非批量、非 num_return_sequences）
            raw_i = _safe_agent_call(
                agents[name],
                ans_prompt,
                max_new_tokens=answer_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generator=g,          # 独立随机源
                # 可选：略微提升多样性
                # repetition_penalty=1.05,
            )
            txt_i = _extract_text(raw_i).strip()
            texts.append(txt_i)

            # 日志：answer_01..answer_10（若 log_utils 是 no-op，则不会落盘）
            _append_log_threadsafe(
                log_path,
                format_log_entry(file, qid, round_num, name, f"answer_{i:02d}", txt_i)
            )

        yn_list = [_extract_yes_no(t) for t in texts]

        return name, refined_plan, yn_list, texts


    with concurrent.futures.ThreadPoolExecutor() as exe:
        for name, refined_plan, yn_list, texts in exe.map(_work, names):
            next_plans_state[name] = refined_plan
            answers[name] = yn_list          # List[str] 长度 = n_samples
            explanations[name] = texts       # List[str] 长度 = n_samples

    return next_plans_state, answers, explanations

def _make_generators(n, device_str="cpu", base_seed=None):
    """
    构造 n 个独立的 torch.Generator，确保每个样本的随机流独立。
    device_str: "cpu" 或 "cuda"（必须与 pipeline.device 匹配）
    """
    if base_seed is None:
        # 用时间与进程/线程信息混合做个基础种子，避免不同问题/回合之间相同
        base_seed = (int(time.time() * 1e6) ^ (os.getpid() << 16)) & 0x7FFFFFFF
    gens = []
    for i in range(n):
        g = torch.Generator(device=device_str)
        g.manual_seed(base_seed + i * 9973)  # 质数步长，降低碰撞
        gens.append(g)
    return gens


def _safe_agent_call_batched(agent_fn, prompts, max_new_tokens=256, **gen_kwargs):
    """
    批量生成版本：prompts 是一个 List[str]（或 List[messages]）。
    尽量触发 HF pipeline 的批处理，通常比 num_return_sequences 更快。
    返回 List[{"generated_text": <str>}]
    """
    try:
        # 绝大多数 HF pipelines 支持 list[str] 作为输入
        resp = agent_fn(prompts, max_new_tokens=max_new_tokens, **gen_kwargs)
    except TypeError:
        # 有的包装器只认 messages 格式；把每个 prompt 包成 messages
        msgs_batch = [[{"role": "user", "content": p}] for p in prompts]
        resp = agent_fn(msgs_batch, max_new_tokens=max_new_tokens, **gen_kwargs)
    except Exception as e:
        # 失败时，保证返回长度一致
        return [{"generated_text": f"[ERROR] {type(e).__name__}: {e}"} for _ in range(len(prompts))]

    # 规范化输出为 list[{"generated_text": ...}]
    out = []
    if resp is None:
        return [{"generated_text": ""} for _ in range(len(prompts))]
    if isinstance(resp, dict):
        resp = [resp]
    if isinstance(resp, (list, tuple)):
        for r in resp:
            if isinstance(r, dict):
                if "generated_text" in r:
                    out.append({"generated_text": r["generated_text"]})
                elif "text" in r:
                    out.append({"generated_text": r["text"]})
                elif "choices" in r:
                    ch0 = r["choices"][0]
                    txt = ch0.get("text") or ch0.get("message", {}).get("content", "")
                    out.append({"generated_text": txt})
                else:
                    out.append({"generated_text": str(r)})
            else:
                out.append({"generated_text": str(r)})
    else:
        out.append({"generated_text": _extract_text(resp)})

    # 如果模型返回数量与期望不等，做截断/补齐，避免后续索引错位
    if len(out) < len(prompts):
        out += [{"generated_text": ""}] * (len(prompts) - len(out))
    elif len(out) > len(prompts):
        out = out[:len(prompts)]
    return out
