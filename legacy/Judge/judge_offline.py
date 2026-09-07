import os
import re
import json
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置区 =================
INPUT_JSONL = "runs/20260228_125154/predictions.jsonl" 
OUTPUT_JSONL = "runs/20260228_125154/predictions_rejudged_fixed.jsonl"

# 坚持使用你指定的 Gemma-9b 模型
JUDGE_MODEL_NAME = "ytu-ce-cosmos/Turkish-Gemma-9b-T1"
GPU_ID = 3  
MAX_NEW_TOKENS_JUDGE = 1024 
# ==========================================

def clean_agent_text(text):
    """
    深度清洗：切除 Agent 的心路历程，只给裁判看核心结论。
    """
    if not isinstance(text, str): return ""
    
    # 1. 移除 Qwen 的 <think> 标签内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 移除指令冲突产生的废话（例如：Don't give explanations...）
    text = text.replace("Don't give explanations, reply with only “yes” or “no”.", "")
    
    # 3. 寻找最后的结论
    match = re.search(r'(?:final answer|answer)[:\s]*(yes|no)', text, re.I)
    if match:
        # 只给裁判看结论及其前后 100 个字符
        start = max(0, match.start() - 100)
        return "..." + text[start:match.end() + 20].strip()
    
    # 4. 兜底：取最后 300 字
    return "..." + text[-300:].strip()

def build_judge_prompt(question, outputs):
    """
    采用 Pre-filling 策略：强制模型从 JSON 的第一个键开始写。
    """
    a1 = clean_agent_text(outputs.get("agent1", ""))
    a2 = clean_agent_text(outputs.get("agent2", ""))
    a3 = clean_agent_text(outputs.get("agent3", ""))

    # 注意：Assistant 结尾直接注入了 {"final_answer":
    return f"""<|im_start|>system
You are a medical expert judge. Your task is to analyze the agent responses and determine the final medical truth.
<|im_end|>
<|im_start|>user
[QUESTION]
{question}

[AGENT ANSWERS]
Agent 1: {a1}
Agent 2: {a2}
Agent 3: {a3}

[TASK]
Output a valid JSON reflecting the truth.
<|im_end|>
<|im_start|>assistant
{{"final_answer": """ 

def safe_parse_json(generated_text):
    """
    补齐我们强行注入的开头，然后解析 JSON。
    """
    full_json = '{"final_answer": ' + generated_text
    match = re.search(r'(\{.*?\})', full_json, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return {"parse_error": True, "raw_output": full_json[:300]}

def load_judge():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.bfloat16 # 建议使用 bf16 减少显存占用并提升速度
    )
    return tokenizer, model

def main():
    tokenizer, model = load_judge()
    print(f"🚀 Judge Loaded (Gemma-9b) on GPU {GPU_ID}")

    # 断点续传
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try: done_ids.add(json.loads(line)["question_id"])
                except: pass

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            qid = data['question_id']
            if qid in done_ids: continue

            print(f"⚖️ Re-judging Q{qid}...")

            prompt = build_judge_prompt(data["question"], data["agents"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_JUDGE,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 只取模型生成的新内容
            res_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            data["judge"] = safe_parse_json(res_text)
            data["judge_raw_output"] = '{"final_answer": ' + res_text

            f_out.write(json.dumps(data) + "\n")
            f_out.flush()

    print(f"✅ 评审完成！新文件保存至: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()