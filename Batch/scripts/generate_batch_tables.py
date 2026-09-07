from pathlib import Path
import pandas as pd

INPUT = Path("results/evaluation/per_batch_run_metrics.csv")
OUT = Path("results/evaluation/batch_specific_tables")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_RUNS = 10

METRICS = [
    ("f1", "F1"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]

VERSIONS = ["v1", "v2", "v3", "v4", "v5"]

VERSION_NAMES = {
    "v1": "Plain",
    "v2": "System prompt",
    "v3": "2-shot",
    "v4": "5-shot",
    "v5": "10-shot",
}

BATCH_NAMES = {
    "Batch1": "Gastrointestinal & Nutritional",
    "Batch2": "Psychiatric & Behavioral",
    "Batch3": "Musculoskeletal",
    "Batch4": "Ophthalmologic",
    "Batch5": "Cardiovascular",
}

# Display order matching the manuscript.
MODEL_ORDER = [
    "Claude",
    "DeepSeek",
    "Falcon3",
    "Gemini3",
    "Gemma",
    "GPT5",
    "Grok",
    "Llama",
    "Qwen",
    "Claude-Panel",
    "Gemini-Panel",
    "Mixed-Panel",
    "Qwen-Panel",
    "Arbitration",
]

df = pd.read_csv(INPUT)

required = {
    "model", "batch", "version", "run_id",
    "accuracy", "precision", "recall", "f1"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# ----------------------------------------------------------------------
# Strict coverage check
# ----------------------------------------------------------------------
counts = (
    df.groupby(["model", "batch", "version"])["run_id"]
      .nunique()
      .reset_index(name="n_runs")
)

bad = counts[counts["n_runs"] != EXPECTED_RUNS]

if not bad.empty:
    print("\nERROR: incomplete batch-specific coverage:")
    print(bad.to_string(index=False))
    raise SystemExit(
        "\nRefusing to generate final tables because some "
        "model/batch/version cells do not contain exactly 10 runs."
    )

print("Coverage check passed: every observed model/batch/version has 10 runs.")

unknown_models = sorted(set(df["model"]) - set(MODEL_ORDER))
if unknown_models:
    print("\nWARNING: model labels not in MODEL_ORDER:")
    for m in unknown_models:
        print(" ", m)
    print(
        "\nThese rows will still be included, but check whether their names "
        "should be normalized to manuscript labels."
    )

markdown_parts = []
table_no = 5

for batch in BATCH_NAMES:
    batch_df = df[df["batch"] == batch]

    for metric, metric_label in METRICS:

        summary = (
            batch_df
            .groupby(["model", "version"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        summary["formatted"] = summary.apply(
            lambda r: f'{r["mean"]:.2f} ({r["std"]:.2f})',
            axis=1
        )

        table = summary.pivot(
            index="model",
            columns="version",
            values="formatted"
        )

        # Preserve manuscript order, then append any unexpected names.
        present = list(table.index)
        ordered = [m for m in MODEL_ORDER if m in present]
        ordered += [m for m in present if m not in ordered]
        table = table.reindex(ordered)

        table = table.reindex(columns=VERSIONS)
        table.columns = [VERSION_NAMES[v] for v in VERSIONS]

        csv_path = OUT / f"Table_{table_no:02d}_{batch}_{metric}.csv"
        table.to_csv(csv_path)

        caption = (
            f"Supplementary Table {table_no}. {metric_label} "
            f"(mean and SD in parentheses) for {BATCH_NAMES[batch]} "
            f"outcomes ({batch}) across five prompt types."
        )

        markdown_parts.append(caption)
        markdown_parts.append("")
        markdown_parts.append(table.to_markdown())
        markdown_parts.append("")
        markdown_parts.append("")

        print(f"[OK] {csv_path}")
        table_no += 1

md_path = OUT / "batch_specific_tables.md"
md_path.write_text("\n".join(markdown_parts), encoding="utf-8")

print()
print("Generated 20 batch-specific tables.")
print(f"Markdown: {md_path}")
print(f"CSV directory: {OUT}")