from transformers import pipeline, AutoTokenizer, set_seed
from pathlib import Path
from datetime import datetime
import csv, re, time, os

# ========= 可调参数（支持环境变量覆盖） =========
# 建议使用 Instruct 版；VL-Instruct 也可文本推理
MODEL  = os.getenv("MODEL",  "tiiuae/Falcon3-7B-Instruct")
GLOB   = os.getenv("GLOB", "Positive Outcome/medical_questions_v*.txt")     # 默认匹配 v1~v5
OUTDIR = Path(os.getenv("OUTDIR", "./results_falcon3"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# 生成配置（改为短输出+不回显）
GEN_CFG = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "3")),        # 只补极少 token
    "num_return_sequences": int(os.getenv("NUM_RETURN_SEQUENCES", "10")),  # 每题生成10条回答
    "do_sample": True, # 默认关闭采样（更稳）
    "temperature": 0.3,
    "top_p": float(os.getenv("TOP_P", "1.0")),
    "return_full_text": False,                                       # 关键：不回显输入
}

# 随机种子（可复现）
SEED = int(os.getenv("SEED", "42"))
set_seed(SEED)

# ========= CUDA 控制 =========
DEVICE_ID = int(os.getenv("DEVICE_ID", "1"))
cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()

print(f"[BOOT] Model={MODEL}")
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
    - 不受文件开头是否有说明文字影响
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行

    # 放宽格式：允许 "Question 1" / "Question #1"
    pattern = re.compile(
        r"(?mi)^Question\s*#?\s*(\d+).*?(?=^Question\s*#?\s*\d+|\Z)",
        re.DOTALL
    )

    for m in pattern.finditer(text):
        qid = m.group(1)
        block = m.group(0).strip()
        yield qid, block

def log_first_qids(file_path: Path, questions, k=5):
    preview = [qid for qid, _ in questions[:k]]
    print(f"[CHECK] {file_path.name} first qids: {preview}")

def to_chat_text(user_prompt: str) -> str:
    """
    用 chat 模板强约束输出只给 yes/no：
    - system 明确：只输出一个 token
    - user 文本后追加 'Answer:' 引导只补一个词
    - 不回显输入（return_full_text=False）
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

    os.environ["PYTHONHASHSEED"] = str(SEED)

    for file in files:
        questions = list(iter_questions_from_file(file))
        if not questions:
            print(f"[WARN] No questions parsed from {file.name}")
            continue

        log_first_qids(file, questions, k=5)
        print(f"==== {file.name}: {len(questions)} questions ====")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = OUTDIR / f"{file.stem}_multi10_{ts}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 多一列 extracted_answer，便于后处理
            writer.writerow(["file","question_id","prompt","response_id","response","extracted_answer"])

            for idx, (qid, qtext) in enumerate(questions, start=1):
                # 每次只喂一题（qtext 已经是切好的单题）
                prompt_chat = to_chat_text(qtext)

                try:
                    outputs = pipe(prompt_chat, **GEN_CFG)
                    if isinstance(outputs, dict):
                        outputs = [outputs]

                    got = 0
                    for j, out in enumerate(outputs, start=1):
                        gen = out.get("generated_text", "")
                        yn = take_yes_no(gen)
                        writer.writerow([file.name, qid, qtext.strip(), j, gen, yn])
                        got += 1

                    print(f"✅ {file.name} Q{idx} (Question {qid}): got {got} responses")

                except Exception as e:
                    writer.writerow([file.name, qid, qtext.strip(), "-", f"[ERROR] {e}", ""])
                    print(f"⚠️  {file.name} Q{idx} (Question {qid}) failed: {e}")

                f.flush()
                time.sleep(0.05)

        print(f"🎯 Saved {out_csv.name}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()
