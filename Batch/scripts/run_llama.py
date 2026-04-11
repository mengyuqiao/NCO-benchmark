#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline
from pathlib import Path
from datetime import datetime
import csv, re, time, os, sys

# ========= 可调参数（支持环境变量覆盖） =========
MODEL = os.getenv("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# 目录结构：questions/Batch1..Batch5/medical_questions_v1..v5.txt
QUESTIONS_ROOT = Path(os.getenv("QUESTIONS_ROOT", "questions"))

# 输出结构：results/<MODEL_TAG>/Batch1..Batch5/
MODEL_TAG = os.getenv("MODEL_TAG", "llama")
OUTROOT = Path(os.getenv("OUTROOT", "results"))

# Hugging Face 生成配置
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "256")),
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),
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

# ========= 主逻辑 =========
def main():
    batches = list_batches(QUESTIONS_ROOT)
    if not batches:
        print(f"[FATAL] No Batch folders found under: {QUESTIONS_ROOT.resolve()}")
        sys.exit(1)

    print(f"📦 Found {len(batches)} batches under: {QUESTIONS_ROOT.resolve()}")
    print(f"🧠 Using model: {MODEL}")
    print(f"🏷️  MODEL_TAG: {MODEL_TAG}")
    print(f"⚙️  num_return_sequences = {GEN_CFG['num_return_sequences']}")
    print(f"🧾 Output root: {(OUTROOT / MODEL_TAG).resolve()}\n")

    for batch_dir in batches:
        batch_name = batch_dir.name
        version_files = list_versions(batch_dir)

        # 每个 batch 输出到 results/<MODEL_TAG>/<BatchX>/
        out_dir = OUTROOT / MODEL_TAG / batch_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"==== {batch_name}: {len(version_files)} files (v1-v5) ====")

        for v, file in version_files:
            questions = list(iter_questions_from_file(file))
            print(f"  -> {file.name}: {len(questions)} questions")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = out_dir / f"{file.stem}_multi{GEN_CFG['num_return_sequences']}_{ts}.csv"

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model_tag","batch","version","file","question_id","prompt","response_id","response"])

                for idx, (qid, qtext) in enumerate(questions, start=1):
                    prompt = qtext.strip()
                    try:
                        outputs = pipe(prompt, **GEN_CFG)
                        if not isinstance(outputs, list):
                            outputs = [outputs]

                        for j, out in enumerate(outputs, start=1):
                            writer.writerow([MODEL_TAG, batch_name, f"v{v}", file.name, qid, prompt, j, out["generated_text"]])

                        print(f"    ✅ {batch_name} {file.name} Q{idx}: got {len(outputs)} responses")
                    except Exception as e:
                        writer.writerow([MODEL_TAG, batch_name, f"v{v}", file.name, qid, prompt, "-", f"[ERROR] {e}"])
                        print(f"    ⚠️  {batch_name} {file.name} Q{idx} failed: {e}")

                    f.flush()
                    time.sleep(0.05)

            print(f"    🎯 Saved: {out_csv.relative_to(OUTROOT)}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()