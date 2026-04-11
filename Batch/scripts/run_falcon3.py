#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer
from pathlib import Path
from datetime import datetime
import csv, re, time, os, sys

# ========= 可调参数（支持环境变量覆盖） =========
MODEL = os.getenv("MODEL", "tiiuae/Falcon3-7B-Instruct")

# 目录结构：questions/Batch1..Batch5/medical_questions_v1..v5.txt
QUESTIONS_ROOT = Path(os.getenv("QUESTIONS_ROOT", "questions"))

# 输出结构：results/<MODEL_TAG>/Batch1..Batch5/
MODEL_TAG = os.getenv("MODEL_TAG", "falcon3_7B")
OUTROOT = Path(os.getenv("OUTROOT", "results"))

# ======== 生成参数 ========
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "256")),
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),
    "do_sample": True,
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("TOP_P", "0.9")),
    "return_full_text": False,
    "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.1")),
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
QUESTION_HEADER = re.compile(r'(?=Question\s+\d+)', flags=re.IGNORECASE)

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

def list_batches(root: Path):
    """找 Batch1..Batch5（也兼容更多 Batch）"""
    if not root.exists():
        print(f"[FATAL] QUESTIONS_ROOT not found: {root.resolve()}")
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("batch")])

def list_versions(batch_dir: Path):
    """Batch 下找 medical_questions_v1..v5（严格 v1-v5）"""
    files = []
    for v in range(1, 6):
        f = batch_dir / f"medical_questions_v{v}.txt"
        if not f.exists():
            raise FileNotFoundError(f"[FATAL] Missing file: {f}")
        files.append((v, f))
    return files

# ======== 主逻辑 ========
def main():
    batches = list_batches(QUESTIONS_ROOT)
    if not batches:
        print(f"[FATAL] No Batch folders found under: {QUESTIONS_ROOT.resolve()}")
        sys.exit(1)

    print(f"📦 Found {len(batches)} batches under: {QUESTIONS_ROOT.resolve()}")
    print(f"🧾 Output root: {(OUTROOT / MODEL_TAG).resolve()}")
    print(f"⚙️  Generation config: {GEN_CFG}\n")

    for batch_dir in batches:
        batch_name = batch_dir.name  # Batch1..Batch5
        version_files = list_versions(batch_dir)

        # 每个 batch 输出到 results/<MODEL_TAG>/<BatchX>/
        out_dir = OUTROOT / MODEL_TAG / batch_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"==== {batch_name}: {len(version_files)} files (v1-v5) ====")

        for v, file in version_files:
            questions = list(iter_questions_from_file(file))
            print(f"  -> {file.name}: {len(questions)} questions")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = out_dir / f"{file.stem}_stable_multi{GEN_CFG['num_return_sequences']}_{ts}.csv"

            with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["model_tag","batch","version","file","question_id","prompt","response_id","response"])

                for idx, (qid, qtext) in enumerate(questions, start=1):
                    prompt = make_prompt(qtext)

                    try:
                        outputs = pipe(prompt, **GEN_CFG)
                        if not isinstance(outputs, list):
                            outputs = [outputs]

                        for j, out in enumerate(outputs, start=1):
                            text_out = out.get("generated_text") or out.get("text") or str(out)
                            text_out = text_out.strip()
                            writer.writerow([MODEL_TAG, batch_name, f"v{v}", file.name, qid, qtext.strip(), j, text_out])

                        print(f"    ✅ {batch_name} {file.name} Q{idx}: got {len(outputs)} responses")
                    except Exception as e:
                        writer.writerow([MODEL_TAG, batch_name, f"v{v}", file.name, qid, qtext.strip(), "-", f"[ERROR] {e}"])
                        print(f"    ⚠️  {batch_name} {file.name} Q{idx} failed: {e}")

                    fcsv.flush()
                    time.sleep(0.05)

            print(f"    🎯 Saved: {out_csv.relative_to(OUTROOT)}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()