#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer
from pathlib import Path
from datetime import datetime
import csv, re, time, os

# ========= 可调参数（支持环境变量覆盖） =========
# MODEL  = os.getenv("MODEL", "tiiuae/Falcon3-1B-Instruct")
MODEL  = os.getenv("MODEL", "tiiuae/Falcon3-7B-Instruct")
GLOB   = os.getenv("GLOB", "Question/NCO/covid_nco_questions_v*.txt")     # 默认匹配 v1~v5
OUTDIR = Path(os.getenv("OUTDIR", "./results_falcon3_7B"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# ======== 生成参数 ========
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "256")),
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),  # ✅ 每题生成10条
    "do_sample": True,
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("TOP_P", "0.9")),
    "return_full_text": False,      # 关键：避免模型把输入复读出来
    "repetition_penalty": 1.1,      # 轻度惩罚复读
}

# ======== 模型加载 ========
DEVICE_ID = int(os.getenv("DEVICE_ID", "4"))
cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()

print(f"[INFO] Using model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

if cuda_visible:
    print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible}, using device_map=auto")
    pipe = pipeline("text-generation", model=MODEL, device_map="auto", torch_dtype="auto")
else:
    print(f"[INFO] Using single GPU device id = {DEVICE_ID}")
    pipe = pipeline("text-generation", model=MODEL, device=DEVICE_ID, torch_dtype="auto")

# ======== Chat 封装 ========
SYSTEM_MSG = (
    "You are a concise medical reasoning assistant. "
    "Answer every question only with 'yes' or 'no'."
)

def make_prompt(question_text: str):
    """将问题包装成 chat 模板"""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question_text.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ======== 工具函数 ========
def iter_questions_from_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r'\n(?=Question\s+\d+)', text, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'Question\s+(\d+)', part, flags=re.IGNORECASE)
        qid = m.group(1) if m else "unknown"
        yield qid, part

# ======== 主逻辑 ========
def main():
    files = sorted(Path(".").glob(GLOB), reverse=False)
    if not files:
        print(f"[FATAL] No files match pattern: {GLOB}")
        sys.exit(1)

    print(f"📦 Found {len(files)} files: {[f.name for f in files]}")
    print(f"🧾 Output dir: {OUTDIR.resolve()}")
    print(f"⚙️  Generation config: {GEN_CFG}\n")

    for file in files:
        questions = list(iter_questions_from_file(file))
        print(f"==== {file.name}: {len(questions)} questions ====")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = OUTDIR / f"{file.stem}_stable_{ts}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "question_id", "prompt", "response_id", "response"])

            for idx, (qid, qtext) in enumerate(questions, start=1):
                prompt = make_prompt(qtext)
                try:
                    outputs = pipe(prompt, **GEN_CFG)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    for j, out in enumerate(outputs, start=1):
                        text = out.get("generated_text") or out.get("text") or str(out)
                        text = text.strip()
                        writer.writerow([file.name, qid, qtext.strip(), j, text])
                    print(f"✅ {file.name} Q{idx}: got {len(outputs)} responses")
                except Exception as e:
                    writer.writerow([file.name, qid, qtext.strip(), "-", f"[ERROR] {e}"])
                    print(f"⚠️  {file.name} Q{idx} failed: {e}")
                f.flush()
                time.sleep(0.1)

        print(f"🎯 Saved: {out_csv.name}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()
