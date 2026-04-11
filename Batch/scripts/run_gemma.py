#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer, set_seed
from pathlib import Path
from datetime import datetime
import csv, re, time, os, sys

# ========= 可调参数（支持环境变量覆盖） =========
MODEL = os.getenv("MODEL", "google/gemma-3-4b-it")

# 目录结构：questions/Batch1..Batch5/medical_questions_v1..v5.txt
QUESTIONS_ROOT = Path(os.getenv("QUESTIONS_ROOT", "questions"))

# 输出结构：results/<MODEL_TAG>/Batch1..Batch5/
MODEL_TAG = os.getenv("MODEL_TAG", "gemma3")
OUTROOT = Path(os.getenv("OUTROOT", "results"))

# 生成配置（短输出+不回显）
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "3")),
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),
    "do_sample": True,
    "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    "top_p": float(os.getenv("TOP_P", "1.0")),
    "return_full_text": False,
}

# 随机种子（可复现）
SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ========= CUDA 控制 =========
DEVICE_ID = int(os.getenv("DEVICE_ID", "1"))
cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()

print(f"[BOOT] Model={MODEL}")
print(f"[BOOT] MODEL_TAG={MODEL_TAG}")
print(f"[BOOT] GEN_CFG={GEN_CFG}")
print(f"[BOOT] SEED={SEED}")

# ====== 初始化 tokenizer 与 pipeline ======
tok = AutoTokenizer.from_pretrained(MODEL)

if cuda_visible:
    print(f"[INFO] Using CUDA_VISIBLE_DEVICES={cuda_visible} with device_map=auto")
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        tokenizer=tok,
        device_map="auto",
        torch_dtype="auto",
    )
else:
    print(f"[INFO] Using single GPU device id = {DEVICE_ID}")
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        tokenizer=tok,
        device=DEVICE_ID,
        torch_dtype="auto",
    )

# ========= 工具函数 =========
def iter_questions_from_file(path: Path):
    """
    稳健的块提取器：
    - 每个块从行首 'Question <num>'（可带 #）开始
    - 一直到下一个 'Question' 行或文件结束
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    pattern = re.compile(
        r"(?mi)^Question\s*#?\s*(\d+).*?(?=^Question\s*#?\s*\d+|\Z)",
        re.DOTALL
    )

    for m in pattern.finditer(text):
        qid = m.group(1)
        block = m.group(0).strip()
        yield qid, block

def to_chat_text(user_prompt: str) -> str:
    """
    用 chat 模板强约束输出只给 yes/no：
    - system 明确：只输出一个 token
    - user 文本后追加 'Answer:' 引导只补一个词
    """
    messages = [
        {"role": "system", "content": "Reply with only one token: yes or no."},
        {"role": "user", "content": user_prompt.strip() + "\nAnswer:"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def take_yes_no(s: str) -> str:
    """兜底：从生成文本中提取第一个 yes/no；都没有就返回 'no'。"""
    if not isinstance(s, str):
        return "no"
    m = re.search(r"\b(yes|no)\b", s.lower())
    return m.group(1) if m else "no"

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
    print(f"🧾 Output root: {(OUTROOT / MODEL_TAG).resolve()}")
    print(f"⚙️  num_return_sequences = {GEN_CFG['num_return_sequences']}\n")

    for batch_dir in batches:
        batch_name = batch_dir.name
        version_files = list_versions(batch_dir)

        out_dir = OUTROOT / MODEL_TAG / batch_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"==== {batch_name}: {len(version_files)} files (v1-v5) ====")

        for v, file in version_files:
            questions = list(iter_questions_from_file(file))
            if not questions:
                print(f"[WARN] No questions parsed from {file}")
                continue

            print(f"  -> {file.name}: {len(questions)} questions")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = out_dir / f"{file.stem}_multi{GEN_CFG['num_return_sequences']}_{ts}.csv"

            with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([
                    "model_tag","batch","version",
                    "file","question_id","prompt",
                    "response_id","response","extracted_answer"
                ])

                for idx, (qid, qtext) in enumerate(questions, start=1):
                    prompt_chat = to_chat_text(qtext)

                    try:
                        outputs = pipe(prompt_chat, **GEN_CFG)
                        if isinstance(outputs, dict):
                            outputs = [outputs]

                        got = 0
                        for j, out in enumerate(outputs, start=1):
                            gen = out.get("generated_text", "")
                            yn = take_yes_no(gen)
                            writer.writerow([
                                MODEL_TAG, batch_name, f"v{v}",
                                file.name, qid, qtext.strip(),
                                j, gen, yn
                            ])
                            got += 1

                        print(f"    ✅ {batch_name} {file.name} Q{idx} (Question {qid}): got {got} responses")

                    except Exception as e:
                        writer.writerow([
                            MODEL_TAG, batch_name, f"v{v}",
                            file.name, qid, qtext.strip(),
                            "-", f"[ERROR] {e}", ""
                        ])
                        print(f"    ⚠️  {batch_name} {file.name} Q{idx} (Question {qid}) failed: {e}")

                    fcsv.flush()
                    time.sleep(0.05)

            print(f"    🎯 Saved: {out_csv.relative_to(OUTROOT)}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()