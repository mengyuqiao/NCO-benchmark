import re
from pathlib import Path
import pandas as pd

# ======================
# 路径配置：按你实际改
# ======================
RESULTS_ROOT = Path(r"./results")   # 里面是 deepseek/falcon/gemma/qwen...
GOLD_ROOT    = Path(r"./results")          # 里面放 batch1_medical_answers.txt ... batch5_...

BATCHES = [f"Batch{i}" for i in range(1, 6)]
VERSION_RE = re.compile(r"_v([1-5])_", re.IGNORECASE)

# 如果 CSV 有 run 列，优先使用；否则按 “每题10条” 推断
RUN_COL_CANDIDATES = ["run", "run_id", "response_id", "choice_id", "candidate_id", "seq_id", "gen_id"]

# 题号列：有的话更稳；没有就按每题10条推断
QID_COL_CANDIDATES = ["question_id", "qid", "q_id", "idx", "index", "problem_id"]
QTEXT_COL_CANDIDATES = ["question", "prompt", "query", "input", "instruction"]


def norm_yesno(x):
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return None


def infer_version_from_filename(fname: str):
    m = VERSION_RE.search(fname)
    return f"v{m.group(1)}" if m else None


def find_first_existing_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_gold_answers(batch: str):
    num = batch.replace("Batch", "")  # "1".."5"
    gold_path = GOLD_ROOT / f"Batch{num}_medical_answers.txt"
    if not gold_path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到 gold 文件: {gold_path}")

    lines = gold_path.read_text(encoding="utf-8").splitlines()
    gold = [norm_yesno(x) for x in lines]
    if any(g is None for g in gold):
        bad = [i for i, g in enumerate(gold) if g is None][:10]
        raise ValueError(f"[ERROR] gold 文件存在非 yes/no 行，示例行号: {bad}")
    return gold_path, gold


def infer_run_ids(df):
    run_col = find_first_existing_col(df, RUN_COL_CANDIDATES)
    if run_col is None:
        # 兜底：每题10条 -> run = (row_index % 10) + 1
        run_ids = (pd.Series(range(len(df)), index=df.index) % 10 + 1).astype(int)
        return "row_mod10", run_ids

    s = df[run_col].astype(str)
    extracted = s.str.extract(r"(\d+)", expand=False)

    if extracted.isna().all():
        # 兜底：还是按每题10条
        run_ids = (pd.Series(range(len(df)), index=df.index) % 10 + 1).astype(int)
        return f"{run_col}(unparsed)->row_mod10", run_ids

    run_ids = extracted.fillna("1").astype(int)
    # 若是 0..9，则映射到 1..10
    if run_ids.min() == 0 and run_ids.max() <= 9:
        run_ids = run_ids + 1
    return run_col, run_ids


def infer_question_index(df):
    """
    返回每行对应的题号 q_idx（从0开始），用于对齐 gold 行号
    """
    qid_col = find_first_existing_col(df, QID_COL_CANDIDATES)
    if qid_col is not None:
        q = pd.to_numeric(df[qid_col], errors="coerce")
        if q.notna().all():
            q = q.astype(int)
            if q.min() == 1:  # 常见 1-based
                q = q - 1
            return qid_col, q

    qtext_col = find_first_existing_col(df, QTEXT_COL_CANDIDATES)
    if qtext_col is not None:
        codes, _ = pd.factorize(df[qtext_col].astype(str), sort=False)
        return qtext_col, pd.Series(codes, index=df.index)

    # 兜底：每题10条 -> q = row_index // 10
    q_idx = (pd.Series(range(len(df)), index=df.index) // 10).astype(int)
    return "row_block10", q_idx


def confusion_counts(pred_series, gold_series):
    """
    pred/gold 都是 'yes'/'no' 的 Series（已过滤有效行）
    以 yes 为 positive
    """
    pred_yes = pred_series == "yes"
    gold_yes = gold_series == "yes"

    tp = int((pred_yes & gold_yes).sum())
    fp = int((pred_yes & ~gold_yes).sum())
    fn = int((~pred_yes & gold_yes).sum())
    tn = int((~pred_yes & ~gold_yes).sum())
    return tp, fp, fn, tn


def safe_div(a, b):
    return (a / b) if b != 0 else 0.0


# ======================
# 主流程：为每个模型生成/覆盖 accuracy.csv（包含四指标）
# ======================
model_dirs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]

for model_dir in model_dirs:
    model_name = model_dir.name
    rows = []

    print(f"\n==============================")
    print(f"Model: {model_name}")
    print(f"Root:  {model_dir}")
    print(f"==============================")

    for batch in BATCHES:
        batch_dir = model_dir / batch
        if not batch_dir.exists():
            print(f"[WARN] 跳过不存在目录: {batch_dir}")
            continue

        gold_path, gold = load_gold_answers(batch)

        for csv_path in batch_dir.glob("*.csv"):
            version = infer_version_from_filename(csv_path.name)
            if version is None:
                print(f"  [SKIP] 无法识别 v1..v5: {csv_path.name}")
                continue

            df = pd.read_csv(csv_path)

            if "answer" not in df.columns:
                raise ValueError(f"[ERROR] {csv_path.name} 没有 answer 列（先跑提取 answer 的脚本）")

            pred = df["answer"].apply(norm_yesno)

            run_src, run_ids = infer_run_ids(df)
            q_src, q_idx = infer_question_index(df)

            gold_series = q_idx.map(lambda i: gold[i] if 0 <= i < len(gold) else None)

            valid = pred.notna() & gold_series.notna()
            if valid.sum() == 0:
                raise ValueError(f"[ERROR] {csv_path} 无可评估样本（answer 或 gold 对齐失败）")

            # 按 run 分组算四指标
            for run_id in sorted(run_ids.unique()):
                mask = valid & (run_ids == run_id)
                total = int(mask.sum())
                if total == 0:
                    continue

                p_run = pred[mask]
                g_run = gold_series[mask]

                tp, fp, fn, tn = confusion_counts(p_run, g_run)

                accuracy = safe_div(tp + tn, tp + tn + fp + fn)
                precision = safe_div(tp, tp + fp)
                recall = safe_div(tp, tp + fn)
                f1 = safe_div(2 * precision * recall, precision + recall)

                rows.append({
                    "model": model_name,
                    "batch": batch,
                    "version": version,
                    "run": int(run_id),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "total": total,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "gold_file": gold_path.name,
                    "run_source": run_src,
                    "q_source": q_src,
                    "csv_file": csv_path.name
                })

    out_df = pd.DataFrame(rows).sort_values(["batch", "version", "run"])
    out_path = model_dir / "accuracy.csv"   # 按你现有命名覆盖
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")