#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# CONFIG
# =====================================================

AGENTS = {
    "agent1": ("Qwen/Qwen3-4B-Thinking-2507", 0),
    "agent2": ("google/gemma-3n-E2B-it", 1),
    "agent3": ("meta-llama/Llama-3.2-3B-Instruct", 2),
}

JUDGE_MODEL = ("ytu-ce-cosmos/Turkish-Gemma-9b-T1", 3)

MAX_NEW_TOKENS_AGENT = 1024
MAX_NEW_TOKENS_JUDGE = 1024

# =====================================================
# DATASET PARSER
# =====================================================

def split_questions(text: str):
    pattern = re.compile(r"(?m)^Question\s+(\d+)")
    matches = list(pattern.finditer(text))

    blocks = []
    for i, m in enumerate(matches):
        qid = int(m.group(1))
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end].strip()
        blocks.append((qid, block))
    return blocks


def find_all_files(root: Path):
    files = []
    for i in range(1, 6):
        d = root / f"Batch{i}"
        if d.exists():
            files.extend(sorted(d.glob("*.txt")))
    return sorted(files)


# =====================================================
# MODEL LOADER
# =====================================================

def load_model(model_name, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# =====================================================
# AGENT WORKER
# =====================================================

def agent_worker(name, model_name, gpu_id, task_q, result_q):

    tokenizer, model = load_model(model_name, gpu_id)
    print(f"[INIT] {name} on GPU {gpu_id}")

    while True:
        item = task_q.get()
        if item is None:
            break

        qid, question = item

        prompt = question + "\nThink briefly and provide your reasoning in ONLY 2-3 sentences, then end with 'Final Answer: yes/no'. DO NOT analyze the examples again.\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_AGENT,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        result_q.put((name, qid, text))


# =====================================================
# JUDGE WORKER
# =====================================================

def build_judge_prompt(question, outputs):

    return f"""
You are a strict medical reasoning judge.

1) Determine the correct answer (yes or no).
2) Evaluate whether each agent's reasoning and final answer are correct with one sentence.

Return JSON ONLY:

{{
  "final_answer": "yes" or "no",
  "agent_evaluation": {{
    "agent1": true/false,
    "agent2": true/false,
    "agent3": true/false
  }}
}}

QUESTION:
{question}

=== AGENT1 ===
{outputs["agent1"]}

=== AGENT2 ===
{outputs["agent2"]}

=== AGENT3 ===
{outputs["agent3"]}
"""


def safe_parse_json(text):
    # 找到第一个 { 和与之匹配的最后一个 }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        clean_json = match.group()
        # 如果模型输出了多个重复的 JSON，我们只取第一个
        if clean_json.count('{"final_answer"') > 1:
            clean_json = clean_json.split('}')[0] + '}'
        try:
            return json.loads(clean_json)
        except:
            pass
    return {"parse_error": True, "raw_output": text[:200]}


def judge_worker(model_name, gpu_id, task_q, result_q):

    tokenizer, model = load_model(model_name, gpu_id)
    print(f"[INIT] JUDGE on GPU {gpu_id}")

    while True:
        item = task_q.get()
        if item is None:
            break

        qid, question, outputs = item

        prompt = build_judge_prompt(question, outputs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_JUDGE,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        result = safe_parse_json(text)

        result_q.put((qid, result))


# =====================================================
# MAIN
# =====================================================

def main(data_root: str, out_root: str):

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_file = out_dir / "predictions.jsonl"

    manager = mp.Manager()
    result_q = manager.Queue()

    # Start agent processes
    agent_queues = {}
    processes = []

    for name, (model, gpu) in AGENTS.items():
        q = manager.Queue()
        agent_queues[name] = q
        p = mp.Process(target=agent_worker, args=(name, model, gpu, q, result_q))
        p.start()
        processes.append(p)

    # Start judge process
    judge_q = manager.Queue()
    judge_p = mp.Process(
        target=judge_worker,
        args=(JUDGE_MODEL[0], JUDGE_MODEL[1], judge_q, result_q)
    )
    judge_p.start()

    files = find_all_files(Path(data_root))

    for file_path in files:

        text = file_path.read_text()
        questions = split_questions(text)

        for qid, block in questions:

            # Resume support
            if pred_file.exists():
                done_ids = set()
                with open(pred_file) as f:
                    for line in f:
                        data = json.loads(line)
                        done_ids.add((data["file"], data["question_id"]))
                if (str(file_path), qid) in done_ids:
                    continue

            # Send to agents
            for q in agent_queues.values():
                q.put((qid, block))

            outputs = {}

            # Collect 3 agent outputs
            for _ in range(3):
                name, rid, text = result_q.get()
                outputs[name] = text

            # =====================================================
            # 在 main 函数中修改这部分：
            # =====================================================

            # ... 前面的代码 (Collect 3 agent outputs)

            # 清洗 Agents 的输出，防止裁判过载
            cleaned_outputs = {}
            for name, raw_text in outputs.items():
                # 1. 移除 Qwen 等模型自带的 <think> 标签内容
                cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
                
                # 2. 如果文本太长（超过 300 字），提取最后一部分结论
                # 很多模型在最后会说 "The final answer is: yes"
                if len(cleaned_text) > 500:
                    # 寻找结论性关键词
                    last_indices = [cleaned_text.rfind(kw) for kw in ["Final Answer", "Answer:", "answer is"]]
                    max_idx = max(last_indices)
                    if max_idx != -1:
                        cleaned_text = "..." + cleaned_text[max_idx:]
                    else:
                        cleaned_text = "..." + cleaned_text[-300:] # 兜底：只截取最后 300 字
                        
                cleaned_outputs[name] = cleaned_text.strip()

            # 发送清洗后的输出给裁判
            judge_q.put((qid, block, cleaned_outputs))

            # Get judge result
            while True:
                rid, judge_result = result_q.get()
                if rid == qid:
                    break

            row = {
                "file": str(file_path),
                "question_id": qid,
                "question": block,
                "agents": outputs,
                "judge": judge_result
            }

            with open(pred_file, "a") as f:
                f.write(json.dumps(row) + "\n")

            print(f"[DONE] {file_path.name} Q{qid}")

    # Shutdown
    for q in agent_queues.values():
        q.put(None)
    judge_q.put(None)

    for p in processes:
        p.join()
    judge_p.join()

    print("All complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", default="runs")
    args = parser.parse_args()

    main(args.data_root, args.out_root)