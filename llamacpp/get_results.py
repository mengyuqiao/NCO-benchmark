import re
import pandas as pd
from pathlib import Path


def load_answers(ans_path):
    answers = []
    with open(ans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line:
                answers.append(line)
    return answers


def compute_metrics(preds, gts):
    tp = fp = fn = tn = 0

    for p, g in zip(preds, gts):
        if p == "yes" and g == "yes":
            tp += 1
        elif p == "yes" and g == "no":
            fp += 1
        elif p == "no" and g == "yes":
            fn += 1
        elif p == "no" and g == "no":
            tn += 1

    acc = (tp + tn) / len(gts)
    prec = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0

    return tp, fp, fn, tn, acc, prec, recall, f1


def main():
    results_root = Path("results_llamacpp")
    questions_root = Path("questions")

    rows = []

    for batch_dir in sorted(results_root.glob("batch*")):
        batch_name = batch_dir.name          # batch1
        batch_name_q = batch_name.capitalize()  # Batch1

        for csv_file in batch_dir.glob("*_rolling.csv"):
            m = re.search(r"_v(\d+)_rolling$", csv_file.stem)
            if not m:
                print(f"[SKIP] cannot parse version from {csv_file.name}")
                continue

            version = f"v{m.group(1)}"
            df = pd.read_csv(csv_file)

            ans_file = questions_root / batch_name_q / f"medical_answers_{version}.txt"
            gts = load_answers(ans_file)

            all_cols = df.columns.tolist()

            runs = set()
            for c in all_cols:
                m_run = re.search(r"_run(\d+)$", c)
                if m_run:
                    runs.add(int(m_run.group(1)))
            runs = sorted(runs)

            for run_num in runs:
                pred_cols = [c for c in all_cols if c.endswith(f"_run{run_num}")]

                final_preds = []
                for _, row in df.iterrows():
                    votes = [
                        str(row[c]).lower()
                        for c in pred_cols
                        if str(row[c]).lower() in ("yes", "no")
                    ]

                    yes_cnt = votes.count("yes")
                    no_cnt = votes.count("no")
                    final_preds.append("yes" if yes_cnt >= no_cnt else "no")

                tp, fp, fn, tn, acc, prec, recall, f1 = compute_metrics(final_preds, gts)

                rows.append({
                    "model": "llamacpp_qwen3.5",
                    "batch": batch_name,
                    "version": version,          # 严格 v1...v5
                    "run": f"run{run_num}",      # 严格 run1...run10
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": recall,
                    "f1": f1,
                })

                print(f"[{batch_name} | {version} | run{run_num}] acc={acc:.4f} f1={f1:.4f}")

    df_out = pd.DataFrame(rows, columns=[
        "model", "batch", "version", "run",
        "tp", "fp", "fn", "tn",
        "accuracy", "precision", "recall", "f1"
    ])

    out_path = Path("results_llamacpp/accuracy_aligned.csv")
    df_out.to_csv(out_path, index=False)

    print("\n[SAVED]")
    print(out_path)


if __name__ == "__main__":
    main()