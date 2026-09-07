#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from g4f.client import Client
from pathlib import Path
import re, csv, time
from datetime import datetime
import os

# ===== 配置 =====
FILES = [Path(f"medical_questions_v{i}_nco.txt") for i in range(1, 6)]  # v1~v5
FILES = list(reversed(FILES))  # ← 让脚本从 v5 开始跑到 v1
MODELS = ['gpt-4o-mini', 'gpt-4o-mini-tts', 'gpt-4.1-mini', 'gpt-4.1-nano']
OUTPUT_DIR = Path(".")
RETRIES_PER_MODEL = 2   # 每个模型重试次数
SLEEP_BETWEEN = 0.2     # 请求间隔秒数

# ===== 工具函数 =====
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

def try_model(client, model: str, question: str):
    """调用模型，带重试"""
    for i in range(1, RETRIES_PER_MODEL + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                timeout=60,
                n=1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            print(f"⚠️  {model} attempt {i} failed: {err}")
            time.sleep(1.5 * i)
    return None

# ===== 主逻辑 =====
def main():
    client = Client()
    files = [f for f in FILES if f.exists()]
    if not files:
        print("[FATAL] No medical_questions_v*_q31-60.txt found!")
        return

    pid = os.getpid()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"results_fallback_q31_60_{ts}_{pid}.csv"
    print(f"🧾 Output file: {out_csv}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "question_id", "model_used", "response"])

        for f_idx, path in enumerate(files, start=1):
            qs = list(iter_questions_from_file(path))
            print(f"\n==== File [{f_idx}/{len(files)}]: {path.name} | {len(qs)} questions ====")

            for q_idx, (qid, qtext) in enumerate(qs, start=1):
                print(f"[{path.name}] Question {q_idx}/{len(qs)} ...", end=" ")

                answered = False
                for model in MODELS:
                    print(f"→ {model}", end=" ", flush=True)
                    ans = try_model(client, model, qtext)
                    if ans:
                        print(f"✅ success with {model}")
                        writer.writerow([path.name, qid, model, ans])
                        f.flush()
                        answered = True
                        break
                    else:
                        print(f"❌", end=" ", flush=True)
                if not answered:
                    writer.writerow([path.name, qid, "all_failed", "[no successful response]"])
                    f.flush()
                time.sleep(SLEEP_BETWEEN)

    print(f"\n🎉 Done. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
