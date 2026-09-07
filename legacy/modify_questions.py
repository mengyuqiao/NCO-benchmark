from pathlib import Path
import re

# ===== 改成你的根目录 =====
ROOT = Path(r".")

BATCHES = [f"Batch{i}" for i in range(1, 6)]
VERSIONS = [f"v{i}" for i in range(1, 6)]

Q_LINE_RE = re.compile(r"^\s*Question\s+\d+\s*$", re.IGNORECASE)
QUESTION_TAG_RE = re.compile(r"^\s*\[Question\]\s*$", re.IGNORECASE)

def clean_questions_text(text: str) -> str:
    lines = text.splitlines()
    out = []

    in_question_block = False     # 已进入某个 Question 块（读到 "Question N"）
    skipping_until_question_tag = False  # 正在跳过，直到遇到 [Question]

    for line in lines:
        if Q_LINE_RE.match(line):
            # 新 question 开始：写入 "Question N"
            out.append(line.strip())
            in_question_block = True
            skipping_until_question_tag = True
            continue

        if in_question_block and skipping_until_question_tag:
            # 我们要一直跳过，直到遇到 [Question]
            if QUESTION_TAG_RE.match(line):
                skipping_until_question_tag = False  # [Question] 本行也删掉
            # 不输出任何内容
            continue

        # 其它情况：原样输出
        out.append(line)

    # 末尾统一用 \n，避免 Windows/Unix 混乱
    return "\n".join(out).rstrip() + "\n"


def process_file(src_path: Path, dst_path: Path):
    text = src_path.read_text(encoding="utf-8")
    cleaned = clean_questions_text(text)
    dst_path.write_text(cleaned, encoding="utf-8")
    print(f"✅ wrote: {dst_path}")

def main():
    for batch in BATCHES:
        batch_dir = ROOT / batch
        if not batch_dir.exists():
            print(f"❌ Missing batch dir: {batch_dir}")
            continue

        for v in VERSIONS:
            src = batch_dir / f"medical_questions_{v}.txt"
            dst = batch_dir / f"new_medical_questions_{v}.txt"
            if not src.exists():
                print(f"❌ Missing source file: {src}")
                continue
            process_file(src, dst)

if __name__ == "__main__":
    main()