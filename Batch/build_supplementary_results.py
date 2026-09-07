#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_INDIVIDUAL = (
    REPO_ROOT
    / "results"
    / "evaluation"
    / "per_run_metrics.csv"
)

DEFAULT_ROLLING = (
    REPO_ROOT
    / "PEG"
    / "results"
    / "rolling_review"
    / "evaluation"
    / "per_run_metrics.csv"
)

DEFAULT_ARBITRATION = (
    REPO_ROOT
    / "PEG"
    / "results"
    / "arbitration"
    / "evaluation"
    / "per_run_metrics.csv"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "supplementary_results.csv"
)


EXPECTED_RUNS = set(
    range(1, 11)
)

EXPECTED_VERSIONS = {
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
}


PROMPT_ORDER = {
    "v1": 1,
    "v2": 2,
    "v3": 3,
    "v4": 4,
    "v5": 5,
}


CONFIGURATION_ORDER = [
    # Individual models
    "Claude",
    "DeepSeek",
    "Falcon3",
    "Gemini3",
    "Gemma",
    "GPT5",
    "Grok",
    "Llama",
    "Qwen",

    # Rolling review
    "Claude-Panel",
    "Gemini-Panel",
    "Mixed-Panel",
    "Qwen-Panel",

    # Arbitration
    "Arbitration",
]


EXPECTED_BY_ARCHITECTURE = {
    "individual": {
        "Claude",
        "DeepSeek",
        "Falcon3",
        "Gemini3",
        "Gemma",
        "GPT5",
        "Grok",
        "Llama",
        "Qwen",
    },

    "rolling_review": {
        "Claude-Panel",
        "Gemini-Panel",
        "Mixed-Panel",
        "Qwen-Panel",
    },

    "arbitration": {
        "Arbitration",
    },
}


# ============================================================
# Name normalization
# ============================================================

INDIVIDUAL_NAME_MAP = {
    "claude": "Claude",

    "deepseek": "DeepSeek",

    "falcon": "Falcon3",
    "falcon3": "Falcon3",

    "gemini": "Gemini3",
    "gemini3": "Gemini3",

    "gemma": "Gemma",

    "gpt5": "GPT5",
    "gpt-5": "GPT5",

    "grok": "Grok",

    "llama": "Llama",
    "llama3": "Llama",

    "qwen": "Qwen",
}


PANEL_NAME_MAP = {
    "claude-panel": "Claude-Panel",
    "claude_panel": "Claude-Panel",

    "gemini-panel": "Gemini-Panel",
    "gemini_panel": "Gemini-Panel",

    "mixed-panel": "Mixed-Panel",
    "mixed_panel": "Mixed-Panel",

    "qwen-panel": "Qwen-Panel",
    "qwen_panel": "Qwen-Panel",

    "arbitration": "Arbitration",
}


def normalize_name(
    value,
    architecture,
):

    raw = str(
        value
    ).strip()

    key = (
        raw
        .lower()
        .replace(" ", "")
    )

    if architecture == "individual":

        if key in INDIVIDUAL_NAME_MAP:
            return INDIVIDUAL_NAME_MAP[
                key
            ]

    else:

        # Preserve "-" and "_" here.
        panel_key = (
            raw
            .strip()
            .lower()
        )

        if panel_key in PANEL_NAME_MAP:
            return PANEL_NAME_MAP[
                panel_key
            ]

    raise ValueError(
        "Unexpected configuration name "
        f"'{value}' for architecture "
        f"'{architecture}'."
    )


# ============================================================
# Input
# ============================================================

REQUIRED_METRIC_COLUMNS = {
    "model",
    "version",
    "prompt_type",
    "run_id",

    "n_total",
    "n_parseable",
    "n_unparseable",
    "parse_rate",

    "nco_total",
    "pco_total",

    "tp_nco",
    "fp_nco",
    "fn_nco",
    "tn_pco",

    "unparseable_nco",
    "unparseable_pco",

    "accuracy",
    "precision",
    "recall",
    "f1",
}


def load_source(
    path: Path,
    architecture: str,
):

    if not path.exists():
        raise FileNotFoundError(
            "Required evaluator output "
            f"does not exist:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    missing = (
        REQUIRED_METRIC_COLUMNS
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} is missing required "
            f"columns: {sorted(missing)}"
        )

    df = df.copy()

    df[
        "architecture"
    ] = architecture

    df[
        "configuration"
    ] = [
        normalize_name(
            value,
            architecture,
        )
        for value
        in df["model"]
    ]

    df[
        "configuration_type"
    ] = (
        "individual"
        if architecture
        == "individual"
        else
        "multi_agent"
    )

    df[
        "positive_class"
    ] = "NCO (no)"

    return df


# ============================================================
# Validation
# ============================================================

def validate_configuration_sets(
    combined,
):

    errors = []

    for (
        architecture,
        expected,
    ) in EXPECTED_BY_ARCHITECTURE.items():

        observed = set(
            combined.loc[
                combined[
                    "architecture"
                ]
                == architecture,
                "configuration",
            ]
        )

        missing = (
            expected
            - observed
        )

        extra = (
            observed
            - expected
        )

        if missing:
            errors.append(
                f"{architecture}: "
                f"missing configurations "
                f"{sorted(missing)}"
            )

        if extra:
            errors.append(
                f"{architecture}: "
                f"unexpected configurations "
                f"{sorted(extra)}"
            )

    if errors:
        raise ValueError(
            "Configuration validation "
            "failed:\n  - "
            + "\n  - ".join(errors)
        )


def validate_runs(
    combined,
):

    errors = []

    for configuration in (
        CONFIGURATION_ORDER
    ):

        sub = combined[
            combined[
                "configuration"
            ]
            == configuration
        ]

        versions = set(
            sub[
                "version"
            ].astype(str)
        )

        if (
            versions
            != EXPECTED_VERSIONS
        ):
            errors.append(
                f"{configuration}: "
                f"versions="
                f"{sorted(versions)}, "
                "expected v1-v5"
            )

        for version in sorted(
            EXPECTED_VERSIONS
        ):

            run_df = sub[
                sub[
                    "version"
                ]
                == version
            ]

            runs = set(
                pd.to_numeric(
                    run_df[
                        "run_id"
                    ],
                    errors="raise",
                )
                .astype(int)
            )

            if (
                runs
                != EXPECTED_RUNS
            ):
                errors.append(
                    f"{configuration} "
                    f"{version}: "
                    f"runs="
                    f"{sorted(runs)}, "
                    "expected 1-10"
                )

    if errors:
        raise ValueError(
            "Independent-run validation "
            "failed:\n  - "
            + "\n  - ".join(
                errors[:100]
            )
        )


def validate_complete_benchmark(
    combined,
):

    bad = combined[
        combined[
            "n_total"
        ]
        != 54
    ]

    if not bad.empty:

        sample = bad[
            [
                "configuration",
                "version",
                "run_id",
                "n_total",
            ]
        ].head(20)

        raise ValueError(
            "Per-run metrics must be "
            "computed over all 54 outcomes.\n"
            + sample.to_string(
                index=False
            )
        )


def validate_duplicates(
    combined,
):

    keys = [
        "architecture",
        "configuration",
        "version",
        "run_id",
    ]

    duplicate = (
        combined
        .duplicated(
            keys,
            keep=False,
        )
    )

    if duplicate.any():

        sample = (
            combined.loc[
                duplicate,
                keys,
            ]
            .head(30)
        )

        raise ValueError(
            "Duplicate per-run metric rows "
            "detected:\n"
            + sample.to_string(
                index=False
            )
        )


def validate_row_count(
    combined,
):

    expected = 700

    observed = len(
        combined
    )

    if observed != expected:
        raise ValueError(
            "Unexpected number of "
            "supplementary result rows: "
            f"{observed}; expected "
            f"{expected}."
        )


# ============================================================
# Output
# ============================================================

OUTPUT_COLUMNS = [
    "configuration",
    "configuration_type",
    "architecture",

    "version",
    "prompt_type",
    "run_id",

    "positive_class",

    "n_total",
    "n_parseable",
    "n_unparseable",
    "parse_rate",

    "nco_total",
    "pco_total",

    "tp_nco",
    "fp_nco",
    "fn_nco",
    "tn_pco",

    "unparseable_nco",
    "unparseable_pco",

    "accuracy",
    "precision",
    "recall",
    "f1",
]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical "
            "supplementary_results.csv"
        )
    )

    parser.add_argument(
        "--individual",
        type=Path,
        default=DEFAULT_INDIVIDUAL,
    )

    parser.add_argument(
        "--rolling",
        type=Path,
        default=DEFAULT_ROLLING,
    )

    parser.add_argument(
        "--arbitration",
        type=Path,
        default=DEFAULT_ARBITRATION,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    individual = load_source(
        args.individual,
        "individual",
    )

    rolling = load_source(
        args.rolling,
        "rolling_review",
    )

    arbitration = load_source(
        args.arbitration,
        "arbitration",
    )

    combined = pd.concat(
        [
            individual,
            rolling,
            arbitration,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Hard validation.
    # --------------------------------------------------------

    validate_configuration_sets(
        combined
    )

    validate_duplicates(
        combined
    )

    validate_runs(
        combined
    )

    validate_complete_benchmark(
        combined
    )

    validate_row_count(
        combined
    )

    # --------------------------------------------------------
    # Deterministic ordering.
    # --------------------------------------------------------

    configuration_order = {
        name: index
        for index, name
        in enumerate(
            CONFIGURATION_ORDER
        )
    }

    combined[
        "_configuration_order"
    ] = combined[
        "configuration"
    ].map(
        configuration_order
    )

    combined[
        "_prompt_order"
    ] = combined[
        "version"
    ].map(
        PROMPT_ORDER
    )

    combined = combined.sort_values(
        [
            "_configuration_order",
            "_prompt_order",
            "run_id",
        ]
    )

    output = combined[
        OUTPUT_COLUMNS
    ].copy()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    print("=" * 72)
    print(
        "supplementary_results.csv"
    )
    print("=" * 72)

    print(
        f"Saved: {args.output.resolve()}"
    )

    print(
        f"Rows:  {len(output)}"
    )

    print()

    print(
        "Individual rows:   "
        f"{len(individual)}"
    )

    print(
        "Rolling rows:      "
        f"{len(rolling)}"
    )

    print(
        "Arbitration rows:  "
        f"{len(arbitration)}"
    )

    print()

    print(
        "Configurations:    "
        f"{output['configuration'].nunique()}"
    )

    print(
        "Prompt types:      "
        f"{output['version'].nunique()}"
    )

    print(
        "Runs/config/prompt: 10"
    )

    print(
        "Outcomes/run:      54"
    )

    print()

    print(
        "✅ All completeness and "
        "consistency checks passed."
    )


if __name__ == "__main__":
    main()
