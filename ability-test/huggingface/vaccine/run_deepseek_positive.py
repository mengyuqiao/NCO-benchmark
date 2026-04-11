#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline
from pathlib import Path
from datetime import datetime
import csv, re, time, os

# ========= 可调参数（支持环境变量覆盖） =========
# MODEL  = os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
MODEL  = os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
GLOB   = os.getenv("GLOB", "Question/positive/covid_positive_questions_v*.txt")     # 默认匹配 v1~v5
OUTDIR = Path(os.getenv("OUTDIR", "./results_deepseek"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Hugging Face 生成配置
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "4096")),
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),  # 每题生成10条回答
    "do_sample": True,
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("TOP_P", "0.95")),
    "return_full_text": False,
}

# ========= CUDA 控制 =========
DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()

if cuda_visible:
    print(f"[INFO] Using CUDA_VISIBLE_DEVICES={cuda_visible} with device_map=auto")
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        device_map="auto",
        torch_dtype="auto",
    )
else:
    print(f"[INFO] Using single GPU device id = {DEVICE_ID}")
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        device=DEVICE_ID,
        torch_dtype="auto",
    )

# ========= 工具函数 =========
def iter_questions_from_file(path: Path):
    """按 Question N 分块"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r'\n(?=Question\s+\d+)', text, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'Question\s+(\d+)', part, flags=re.IGNORECASE)
        qid = m.group(1) if m else "unknown"
        yield qid, part

# ========= 主逻辑 =========
def main():
    files = sorted(Path(".").glob(GLOB), reverse=True)  # v5 → v1
    if not files:
        print(f"[FATAL] No files match: {GLOB}")
        return

    print(f"📦 Found {len(files)} files: {[f.name for f in files]}")
    print(f"🧠 Using model: {MODEL}")
    print(f"⚙️  num_return_sequences = {GEN_CFG['num_return_sequences']}")
    print(f"🧾 Output dir: {OUTDIR.resolve()}\n")

    for file in files:
        questions = list(iter_questions_from_file(file))
        print(f"==== {file.name}: {len(questions)} questions ====")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = OUTDIR / f"{file.stem}_multi10_{ts}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file","question_id","prompt","response_id","response"])

            for idx, (qid, qtext) in enumerate(questions, start=1):
                prompt = qtext.strip()

                # DeepSeek 支持 messages 对话格式
                messages = [{"role": "user", "content": prompt}]
                try:
                    outputs = pipe(messages, **GEN_CFG)
                    for j, out in enumerate(outputs, start=1):
                        writer.writerow([file.name, qid, prompt, j, out["generated_text"]])
                    print(f"✅ {file.name} Q{idx}: got {len(outputs)} responses")
                except Exception as e:
                    writer.writerow([file.name, qid, prompt, "-", f"[ERROR] {e}"])
                    print(f"⚠️  {file.name} Q{idx} failed: {e}")
                f.flush()
                time.sleep(0.1)
        print(f"🎯 Saved {out_csv.name}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()
