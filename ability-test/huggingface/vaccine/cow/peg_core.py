from log_utils import format_log_entry, append_log
import pandas as pd
import concurrent.futures
from collections import defaultdict
import re
import threading
import time
import torch
import os
import json

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



def _coerce_bundle(prev_name: str, incoming_text: str):
    """
    将上一棒传来的 payload 规范为“累积包”结构：
    {
      "summaries": { "<agent>": "<latest concise summary>" , ... },
      "opinions":  { "<agent>": "<that agent's own view (raw)>" , ... }
    }
    - 若 incoming_text 是 JSON 且含上述键，直接返回
    - 否则把 incoming_text 当作 prev_name 的原始观点，初始化包
    """
    try:
        obj = json.loads(incoming_text)
        if isinstance(obj, dict) and "summaries" in obj and "opinions" in obj:
            # 基本校验
            obj["summaries"] = obj.get("summaries", {}) or {}
            obj["opinions"]  = obj.get("opinions",  {}) or {}
            return obj
    except Exception:
        pass

    # 退化为仅含上一棒观点的初始包
    return {
        "summaries": {},
        "opinions": { prev_name: (incoming_text or "").strip() }
    }

def _merge_summaries(bundle, new_summaries: dict):
    """用当前模型新产出的 summaries 覆盖/更新累积包中的 summaries。"""
    if not isinstance(new_summaries, dict):
        return bundle
    summ = bundle.get("summaries", {}) or {}
    for k, v in new_summaries.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            summ[k] = v.strip()
    bundle["summaries"] = summ
    return bundle

# === NEW 2: “环形接力一轮”（同步轮）：上一棒的 plan -> 我 refine+answer -> 产出下一轮我传出的 plan ===
def relay_ring_round(
    prompt,
    agents,
    plans_state,          # 现在被当作“上一棒传来的 payload（字符串或JSON包）”
    order,
    file,
    qid,
    round_num,
    log_path,
    plan_tokens=220,      # 保留（用于 summary 步）
    answer_tokens=256,    # 自己观点（多样性采样）的最大长度
    n_samples=10,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=None,  # 可选
):
    """
    新语义：
      - 每个模型接过“累积包”（summaries/opinions）
      - 先对包里所有“前序参与者”的过程做简明总结（覆盖其 summaries）
      - 再给出自己观点（Top-N 独立采样）
      - 把更新后的包（含最新 summaries + 所有 opinions）传给下一棒
    返回：
      - next_plans_state: { name -> json.dumps(bundle) }
      - answers: { name -> [ 'yes'/'no', ... ] }  # N 个
      - explanations: { name -> [ full_text, ... ] }  # N 个
    """
    names = list(order)

    # 本轮“入场包”由上一棒提供：对每个 name，来自其前一个 prev_name
    incoming_map = {}
    for idx, name in enumerate(names):
        prev_name = names[(idx - 1) % len(names)]
        incoming_payload = plans_state.get(prev_name, "")  # 兼容旧：可能仅是文本
        # 标准化为“累积包”
        incoming_map[name] = (prev_name, _coerce_bundle(prev_name, incoming_payload))

    next_plans_state = {}
    answers, explanations = {}, {}

    def _detect_device_str(agent):
        dev = getattr(agent, "device", None)
        try:
            if dev is None:
                return "cuda" if torch.cuda.is_available() else "cpu"
            if isinstance(dev, int):
                return "cuda" if (dev >= 0 and torch.cuda.is_available()) else "cpu"
            t = getattr(dev, "type", None)
            if t == "cuda" and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _make_generators(n, device_str="cpu", base_seed=None):
        if base_seed is None:
            base_seed = (int(time.time() * 1e6) ^ (os.getpid() << 16)) & 0x7FFFFFFF
        gens = []
        for i in range(n):
            g = torch.Generator(device=device_str)
            g.manual_seed(base_seed + i * 9973)
            gens.append(g)
        return gens

    def _work(name):
        prev_name, bundle = incoming_map[name]
        # 需要总结的对象集合：所有已出现过的参与者（不含自己）
        prior_agents = [k for k in (set(bundle.get("summaries", {}).keys()) |
                                    set(bundle.get("opinions", {}).keys()))
                        if k != name]

        # === (1) 生成“对所有前序参与者的简明总结” ===
        # 组装材料：对每个 prior_agent 提供“目前可见的材料”（优先用 opinions，否则用 summaries）
        blocks = []
        for who in prior_agents:
            # 优先总结其“原始观点”，若没有再总结“已存在的总结”
            source = bundle["opinions"].get(who) or bundle["summaries"].get(who) or ""
            if source.strip():
                blocks.append(f"[{who}]\n{source.strip()}")
        material = "\n\n".join(blocks) if blocks else "(no prior agents)"

        summarize_prompt = (
            "You are the next agent in a relay. Please summarize the previous participants' reasoning, "
            "writing 1–2 concise sentences for EACH participant listed below. "
            "Return STRICTLY a JSON object with the following schema:\n"
            "{\n"
            '  "summaries": {\n'
            '    "<agent_name>": "<your concise summary for that agent>",\n'
            "    ...\n"
            "  }\n"
            "}\n\n"
            "Participants' materials:\n"
            f"{material}\n"
        )
        raw_sum = _safe_agent_call(agents[name], summarize_prompt, max_new_tokens=plan_tokens)
        sum_text = _extract_text(raw_sum).strip()

        # 尝试解析 JSON；解析失败则降级为空 summaries
        new_summ = {}
        try:
            obj = json.loads(sum_text)
            if isinstance(obj, dict) and isinstance(obj.get("summaries", None), dict):
                # 仅保留 prior_agents 的键，避免模型胡乱扩展
                new_summ = {k: v for k, v in obj["summaries"].items() if k in prior_agents and isinstance(v, str)}
        except Exception:
            new_summ = {}

        # 写日志：当前模型的“对前序的总结”
        _append_log_threadsafe(log_path, format_log_entry(file, qid, round_num, name, "summary", json.dumps(new_summ, ensure_ascii=False)))

        # 将新总结合入累积包（覆盖旧的 summaries）
        bundle = _merge_summaries(bundle, new_summ)

        # === (2) 生成“自己的观点” Top-N（独立随机源） ===
        # 供模型参考的“目前总览”（把 summaries 和 opinions 都简要给一下）
        overview_parts = []
        if bundle["summaries"]:
            overview_parts.append("Current summaries:\n" + "\n".join([f"- {k}: {v}" for k, v in bundle["summaries"].items()]))
        if bundle["opinions"]:
            # 只把名称列出来，避免 prompt 过长；真的需要可改成截断文本展示
            overview_parts.append("Existing opinions from: " + ", ".join(sorted(bundle["opinions"].keys())))
        overview = "\n\n".join(overview_parts) if overview_parts else "(no history)"

        opinion_prompt = (
            f"You have the following question:\n{prompt}\n\n"
            "Context from previous participants (summaries and who has spoken):\n"
            f"{overview}\n\n"
            "Now provide YOUR OWN VIEW.\n"
            "- Begin with 'yes' or 'no' on the first line.\n"
            "- Then give a short justification (2–4 sentences max).\n"
        )

        device_str = _detect_device_str(agents[name])
        gens = _make_generators(n_samples, device_str=device_str)

        texts = []
        for i, g in enumerate(gens, 1):
            gen_kwargs = dict(
                max_new_tokens=answer_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generator=g,
            )
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty

            raw_i = _safe_agent_call(agents[name], opinion_prompt, **gen_kwargs)
            txt_i = _extract_text(raw_i).strip()
            texts.append(txt_i)

            _append_log_threadsafe(log_path, format_log_entry(file, qid, round_num, name, f"opinion_{i:02d}", txt_i))

        yn_list = [_extract_yes_no(t) for t in texts]

        # 将“自己的观点（选第一个样本作为代表）”写入累积包 opinions[name]
        rep_opinion = texts[0] if texts else ""
        bundle["opinions"][name] = rep_opinion

        # 传给下一棒：序列化为 JSON 文本（确保跨进程/落日志可视）
        payload_out = json.dumps(bundle, ensure_ascii=False)

        return name, payload_out, yn_list, texts

    with concurrent.futures.ThreadPoolExecutor() as exe:
        for name, payload_out, yn_list, texts in exe.map(_work, names):
            next_plans_state[name] = payload_out
            answers[name] = yn_list
            explanations[name] = texts

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
