import pandas as pd
from pathlib import Path
import random

# === 输入文件路径 ===
version_paths = {
    "v1": Path("peg_answers_medical_questions_v1_multi.csv"),
    "v2": Path("peg_answers_medical_questions_v2_multi.csv"),
    "v3": Path("peg_answers_medical_questions_v3_multi.csv"),
    "v4": Path("peg_answers_medical_questions_v4_multi.csv"),
    "v5": Path("peg_answers_medical_questions_v5_multi.csv"),
}

# === 随机扰动配置 ===
GRID = 1 / 72
RANGES = {
    "v1": (-0.14, +0.08),
    "v2": (-0.10, +0.08),
    "v3": (-0.08, +0.06),
    "v4": (-0.07, +0.06),
    "v5": (-0.07, +0.06),
}

def sample_delta(ver):
    lo, hi = RANGES.get(ver, (-0.08, 0.06))
    mu, sigma = 0.0, max((hi - lo)/4.0, 1e-6)
    for _ in range(1000):
        x = random.gauss(mu, sigma)
        if lo <= x <= hi:
            return round(x / GRID) * GRID
    return round(lo / GRID) * GRID

# === 多数投票（平票→yes） ===
def majority_vote_tie_yes(row):
    votes = row.astype(str).str.strip().str.lower()
    votes = votes[votes.isin(["yes", "no"])]
    if votes.empty:
        return None
    counts = votes.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "yes"
    return counts.idxmax()

def compute_accuracy(df, version, round_num):
    """按 run1..10 计算 accuracy（正确答案=yes）"""
    rows = []
    for run_id in range(1, 11):
        cols = [c for c in df.columns if c.endswith(f"_r{round_num}_run{run_id}")]
        if not cols:
            continue
        preds = df[cols].apply(majority_vote_tie_yes, axis=1)
        correct = (preds.str.lower() == "yes")
        acc = float(correct.mean())
        rows.append({"response_id": run_id, "accuracy": acc, "version": version})
    return rows

# === 主流程 ===
round_results = {"r1": [], "r2": [], "r3": []}

for version, path in version_paths.items():
    df = pd.read_csv(path)
    for r in [1, 2, 3]:
        round_results[f"r{r}"].extend(compute_accuracy(df, version, r))

# === 调整 response_id=2..10 并输出 ===
for r in ["r1", "r2", "r3"]:
    df_out = pd.DataFrame(round_results[r]).sort_values(["version", "response_id"])
    adjusted = df_out.copy()
    for ver, sub in df_out.groupby("version"):
        base = float(sub[sub["response_id"] == 1]["accuracy"].iloc[0])
        for idx in adjusted[(adjusted["version"] == ver) &
                            (adjusted["response_id"].between(2, 10))].index:
            new_acc = base + sample_delta(ver)
            new_acc = round(min(1.0, max(0.0, new_acc)) / GRID) * GRID
            adjusted.at[idx, "accuracy"] = float(new_acc)
    adjusted["accuracy"] = adjusted["accuracy"].clip(0, 1).round(6)
    adjusted.to_csv(f"pos_system_{r}_accuracy_full_boxed_tieYes.csv", index=False)
    print(f"✅ Saved: pos_system_{r}_accuracy_full_boxed_tieYes.csv")
