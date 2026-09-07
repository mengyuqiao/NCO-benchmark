#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline
from pathlib import Path
from datetime import datetime
import csv, re, time, os

# ========= 可调参数（支持环境变量覆盖） =========
# MODEL  = os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
MODEL  = os.getenv("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

QUESTIONS_ROOT = Path(
    os.getenv(
        "QUESTIONS_ROOT",
        str(REPO_ROOT / "Batch" / "questions")
    )
)
MODEL_TAG = os.getenv("MODEL_TAG", "deepseek")
OUTROOT = Path(
    os.getenv(
        "OUTROOT",
        str(REPO_ROOT / "results")
    )
)
NUM_RUNS = int(os.getenv("NUM_RUNS", "10"))
OUTROOT.mkdir(parents=True, exist_ok=True)

# Hugging Face 生成配置
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "2048")),
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": float(os.getenv("TEMPERATURE", "1.0")),
    "top_p": float(os.getenv("TOP_P", "1.0")),
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

def list_batches(root: Path):
    if not root.exists():
        print(f"[FATAL] QUESTIONS_ROOT not found: {root.resolve()}")
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("batch")])

def list_versions(batch_dir: Path):
    files = []
    for v in range(1, 6):
        f = batch_dir / f"medical_questions_v{v}.txt"
        if not f.exists():
            raise FileNotFoundError(f"[FATAL] Missing file: {f}")
        files.append((v, f))
    return files

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
    batches = list_batches(QUESTIONS_ROOT)
    if not batches:
        print(f"[FATAL] No Batch folders found under: {QUESTIONS_ROOT.resolve()}")
        return

    print(f"📦 Found {len(batches)} batches under: {QUESTIONS_ROOT.resolve()}")
    print(f"🧠 Using model: {MODEL}")
    print(f"🏷️  MODEL_TAG: {MODEL_TAG}")
    print(f"⚙️  num_return_sequences = {GEN_CFG['num_return_sequences']}")
    print(f"🧾 Output root: {OUTROOT.resolve()}\n")

    for batch_dir in batches:
        batch_name = batch_dir.name  # Batch1..Batch5
        version_files = list_versions(batch_dir)

        # 每个 batch 输出到 results/<model_tag>/<batch>/
        out_dir = OUTROOT / MODEL_TAG / batch_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"==== {batch_name}: {len(version_files)} files (v1-v5) ====")

        for v, file in version_files:
            questions = list(iter_questions_from_file(file))
            print(f"  -> {file.name}: {len(questions)} questions")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = out_dir / f"{file.stem}_{NUM_RUNS}runs_{ts}.csv"

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "batch",
                    "version",
                    "file",
                    "run_id",
                    "question_id",
                    "prompt",
                    "response"
                ])

                for run_id in range(1, NUM_RUNS + 1):
                    print(f"    ▶ Independent run {run_id}/{NUM_RUNS}")

                    for idx, (qid, qtext) in enumerate(questions, start=1):
                        prompt = qtext.strip()
                        messages = [{"role": "user", "content": prompt}]

                        try:
                            outputs = pipe(messages, **GEN_CFG)

                            if not isinstance(outputs, list):
                                outputs = [outputs]

                            if len(outputs) != 1:
                                raise RuntimeError(
                                    f"Expected exactly one response, got {len(outputs)}"
                                )

                            out = outputs[0]

                            writer.writerow([
                                batch_name,
                                f"v{v}",
                                file.name,
                                run_id,
                                qid,
                                prompt,
                                out["generated_text"]
                            ])

                            print(
                                f"      ✅ run={run_id} "
                                f"{batch_name} {file.name} Q{idx}"
                            )

                        except Exception as e:
                            writer.writerow([
                                batch_name,
                                f"v{v}",
                                file.name,
                                run_id,
                                qid,
                                prompt,
                                f"[ERROR] {e}"
                            ])

                            print(
                                f"      ⚠️ run={run_id} "
                                f"{batch_name} {file.name} Q{idx} failed: {e}"
                            )

                        f.flush()
                        time.sleep(0.05)

            print(f"    🎯 Saved {out_csv.relative_to(OUTROOT)}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()
