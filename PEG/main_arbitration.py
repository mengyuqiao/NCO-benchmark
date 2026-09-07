#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional

from arbitration_core import (
    PANEL_SIZE,
    run_arbitration,
)
from model_loader import load_agents
from question_loader import load_question_blocks


# ============================================================
# Paths
# ============================================================

PEG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PEG_DIR.parent

DEFAULT_PANEL_CONFIG = (
    PEG_DIR
    / "panels"
    / "mixed.json"
)

DEFAULT_QUESTIONS_ROOT = (
    REPO_ROOT
    / "Batch"
    / "questions"
)

DEFAULT_OUT_ROOT = (
    PEG_DIR
    / "results"
    / "arbitration"
    / "Arbitration"
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-aligned "
            "five-agent + Claude arbitration experiment."
        )
    )

    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL_CONFIG,
        help=(
            "Five-agent arbitration panel. "
            "Default: PEG/panels/mixed.json"
        ),
    )

    parser.add_argument(
        "--questions-root",
        type=Path,
        default=DEFAULT_QUESTIONS_ROOT,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help=(
            "Override panel num_runs. "
            "Use 1 for smoke tests."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        choices=range(1, 6),
        metavar="1-5",
        default=None,
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=range(1, 6),
        metavar="1-5",
        default=None,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Config helpers
# ============================================================

def load_json(
    path: Path,
) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_panel(
    config: Dict,
):
    agents = config.get(
        "agents",
        []
    )

    if len(agents) != PANEL_SIZE:
        raise ValueError(
            "Arbitration requires exactly "
            f"{PANEL_SIZE} panel agents."
        )

    agent_ids = [
        str(
            x.get(
                "agent_id",
                ""
            )
        ).strip()
        for x in agents
    ]

    if any(
        not x
        for x in agent_ids
    ):
        raise ValueError(
            "Every panel agent requires agent_id."
        )

    if len(set(agent_ids)) != PANEL_SIZE:
        raise ValueError(
            "All five panel agent IDs "
            "must be unique."
        )


def resolve_model_id(
    spec: Dict,
) -> str:
    model_id = str(
        spec.get(
            "model_id",
            ""
        )
    ).strip()

    if model_id:
        return model_id

    env_name = str(
        spec.get(
            "model_id_env",
            ""
        )
    ).strip()

    if env_name:
        value = os.getenv(
            env_name,
            ""
        ).strip()

        if value:
            return value

        raise RuntimeError(
            f"Missing required environment "
            f"variable: {env_name}"
        )

    raise ValueError(
        f"No model ID configured for "
        f"{spec.get('agent_id')}."
    )


def build_claude_judge_spec():
    """
    Exact Claude model ID is supplied at runtime.

    Required:
        CLAUDE_MODEL_ID
        ANTHROPIC_API_KEY

    We do not guess the historical Claude checkpoint.
    """

    return {
        "agent_id":
            "claude_judge",

        "provider":
            "anthropic",

        "model_id_env":
            "claude-sonnet-4-6",
    }


# ============================================================
# Benchmark discovery
# ============================================================

def discover_question_files(
    root: Path,
    batch_filter: Optional[int],
    version_filter: Optional[int],
):
    paths = []

    batches = (
        [batch_filter]
        if batch_filter is not None
        else range(1, 6)
    )

    versions = (
        [version_filter]
        if version_filter is not None
        else range(1, 6)
    )

    for batch_id in batches:
        batch_dir = (
            root
            / f"Batch{batch_id}"
        )

        if not batch_dir.exists():
            raise FileNotFoundError(
                f"Missing benchmark directory: "
                f"{batch_dir}"
            )

        for version in versions:
            path = (
                batch_dir
                / f"medical_questions_v"
                  f"{version}.txt"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing benchmark file: "
                    f"{path}"
                )

            paths.append(path)

    return paths


def infer_batch_version(
    path: Path,
):
    batch = path.parent.name

    version_number = (
        path.stem
        .rsplit("_v", 1)[-1]
    )

    return (
        batch,
        f"v{version_number}",
    )


# ============================================================
# Resume helpers
# ============================================================

def completed_keys(
    path: Path,
):
    done = set()

    if not path.exists():
        return done

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

                if (
                    row.get("status")
                    == "completed"
                ):
                    done.add((
                        int(
                            row["run_id"]
                        ),
                        str(
                            row["question_id"]
                        ),
                    ))

            except Exception:
                continue

    return done


# ============================================================
# Output helpers
# ============================================================

def append_jsonl(
    path: Path,
    record: Dict,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_csv_header_if_needed(
    path: Path,
):
    if path.exists():
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch",
            "version",
            "run_id",
            "question_id",
            "source_file",
            "original_prompt",

            "agent1_id",
            "agent1_response",

            "agent2_id",
            "agent2_response",

            "agent3_id",
            "agent3_response",

            "agent4_id",
            "agent4_response",

            "agent5_id",
            "agent5_response",

            "judge_id",
            "judge_response",
            "judge_input_tokens",
            "judge_output_tokens",

            "final_response",
        ])


def append_csv(
    path: Path,
    batch: str,
    version: str,
    question_id: str,
    source_file: str,
    original_prompt: str,
    result: Dict,
):
    outputs = result[
        "panel_outputs"
    ]

    if len(outputs) != PANEL_SIZE:
        raise RuntimeError(
            "Expected exactly five "
            "panel outputs."
        )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            batch,
            version,
            result["run_id"],
            question_id,
            source_file,
            original_prompt,

            outputs[0]["agent_id"],
            outputs[0]["response"],

            outputs[1]["agent_id"],
            outputs[1]["response"],

            outputs[2]["agent_id"],
            outputs[2]["response"],

            outputs[3]["agent_id"],
            outputs[3]["response"],

            outputs[4]["agent_id"],
            outputs[4]["response"],
            
            result["judge_id"],
            result["judge_response"],

            (
                result.get(
                    "judge_usage"
                ) or {}
            ).get(
                "input_tokens"
            ),

            (
                result.get(
                    "judge_usage"
                ) or {}
            ).get(
                "output_tokens"
            ),

            result["final_response"],
        ])


def write_manifest(
    out_dir: Path,
    panel_config: Dict,
    num_runs: int,
):
    resolved_agents = []

    for position, spec in enumerate(
        panel_config["agents"],
        start=1,
    ):
        resolved_agents.append({
            "position":
                position,

            "agent_id":
                spec["agent_id"],

            "provider":
                spec["provider"],

            "model_id":
                resolve_model_id(
                    spec
                ),

            "device":
                spec.get(
                    "device"
                ),
        })

    manifest = {
        "architecture":
            "arbitration",

        "num_runs":
            num_runs,

        "panel_name":
            panel_config.get(
                "panel_name",
                "Mixed-Panel"
            ),

        "panel_agents":
            resolved_agents,

        "judge": {
            "agent_id":
                "claude_judge",

            "provider":
                "anthropic",

            "model_id":
                "claude-sonnet-4-6",
        },

        "generation_config": {
            "max_new_tokens":
                2048,

            "do_sample":
                True,

            "temperature":
                1.0,

            "top_p":
                1.0,
        },
    }

    with (
        out_dir
        / "run_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    panel_config = load_json(
        args.panel
    )

    validate_panel(
        panel_config
    )

    num_runs = (
        args.num_runs
        if args.num_runs is not None
        else int(
            panel_config.get(
                "num_runs",
                10
            )
        )
    )

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be positive."
        )

    judge_spec = (
        build_claude_judge_spec()
    )

    print("=" * 72)
    print("NCO Arbitration")
    print("=" * 72)

    print(
        "Panel:       "
        f"{panel_config.get('panel_name')}"
    )

    print(
        f"Panel size:  "
        f"{len(panel_config['agents'])}"
    )

    print(
        f"Runs:        "
        f"{num_runs}"
    )

    print()
    print("Independent panel agents:")

    for position, spec in enumerate(
        panel_config["agents"],
        start=1,
    ):
        print(
            f"  {position}: "
            f"{spec['agent_id']} | "
            f"{spec['provider']} | "
            f"{spec.get('model_id')}"
        )

    print()
    print(
        "Judge:       "
        "claude_judge | anthropic | "
        "$CLAUDE_MODEL_ID"
    )

    # --------------------------------------------------------
    # Structural validation can run without API credentials.
    # --------------------------------------------------------

    if args.validate_only:
        print()

        if not os.getenv(
            "CLAUDE_MODEL_ID"
        ):
            print(
                "ℹ️ Runtime model ID still needs:"
            )
            print(
                "   CLAUDE_MODEL_ID"
            )

        if not os.getenv(
            "ANTHROPIC_API_KEY"
        ):
            print(
                "ℹ️ Runtime credential still needs:"
            )
            print(
                "   ANTHROPIC_API_KEY"
            )

        print()
        print(
            "✅ Arbitration configuration "
            "is structurally valid."
        )

        return

    # --------------------------------------------------------
    # Require Claude runtime configuration.
    # --------------------------------------------------------

    
    if not os.getenv(
        "ANTHROPIC_API_KEY"
    ):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set."
        )

    # --------------------------------------------------------
    # Resolve panel model IDs.
    # --------------------------------------------------------

    resolved_panel_specs = []

    for spec in panel_config[
        "agents"
    ]:
        resolved = dict(
            spec
        )

        resolved[
            "model_id"
        ] = resolve_model_id(
            spec
        )

        resolved_panel_specs.append(
            resolved
        )

    # --------------------------------------------------------
    # Load the five Mixed-Panel agents.
    # --------------------------------------------------------

    panel_agents = load_agents(
        resolved_panel_specs
    )

    agent_order = [
        spec["agent_id"]
        for spec in resolved_panel_specs
    ]

    # --------------------------------------------------------
    # Load the Claude judge.
    # --------------------------------------------------------

    resolved_judge_spec = dict(
        judge_spec
    )

    resolved_judge_spec[
        "model_id"
    ] = resolve_model_id(
        judge_spec
    )

    judge_agents = load_agents(
        [
            resolved_judge_spec
        ]
    )

    judge = judge_agents[
        "claude_judge"
    ]

    # --------------------------------------------------------
    # Output directory.
    # --------------------------------------------------------

    out_dir = args.out_root

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_manifest(
        out_dir=out_dir,
        panel_config=panel_config,
        num_runs=num_runs,
    )

    # --------------------------------------------------------
    # Benchmark files.
    # --------------------------------------------------------

    question_files = (
        discover_question_files(
            root=args.questions_root,
            batch_filter=args.batch,
            version_filter=args.version,
        )
    )

    total_completed = 0
    total_failed = 0

    for file_index, question_path in enumerate(
        question_files,
        start=1,
    ):
        batch, version = (
            infer_batch_version(
                question_path
            )
        )

        questions = load_question_blocks(
            question_path
        )

        if not questions:
            raise RuntimeError(
                f"No questions parsed from "
                f"{question_path}"
            )

        file_tag = (
            f"{batch}_{version}"
        )

        raw_path = (
            out_dir
            / f"{file_tag}_raw.jsonl"
        )

        csv_path = (
            out_dir
            / f"{file_tag}_responses.csv"
        )

        error_path = (
            out_dir
            / f"{file_tag}_errors.jsonl"
        )

        write_csv_header_if_needed(
            csv_path
        )

        done = completed_keys(
            raw_path
        )

        print()
        print(
            f"[{file_index}/"
            f"{len(question_files)}] "
            f"{batch} {version} | "
            f"{len(questions)} questions"
        )

        for run_id in range(
            1,
            num_runs + 1,
        ):
            print(
                f"  ▶ Independent run "
                f"{run_id}/{num_runs}"
            )

            for (
                question_id,
                original_prompt
            ) in questions:

                key = (
                    run_id,
                    str(question_id),
                )

                if key in done:
                    print(
                        f"    [SKIP] "
                        f"run={run_id} "
                        f"Q{question_id}"
                    )
                    continue

                print(
                    f"    [RUN] "
                    f"run={run_id} "
                    f"Q{question_id}"
                )

                try:
                    result = run_arbitration(
                        original_prompt=
                            original_prompt,

                        agents=
                            panel_agents,

                        agent_order=
                            agent_order,

                        judge=
                            judge,

                        judge_id=
                            "claude_judge",

                        run_id=
                            run_id,
                    )

                    record = {
                        "status":
                            "completed",

                        "architecture":
                            "arbitration",

                        "panel_name":
                            panel_config.get(
                                "panel_name",
                                "Mixed-Panel"
                            ),

                        "batch":
                            batch,

                        "version":
                            version,

                        "source_file":
                            str(
                                question_path
                            ),

                        "question_id":
                            str(
                                question_id
                            ),

                        "original_prompt":
                            original_prompt,

                        **result,
                    }

                    append_jsonl(
                        raw_path,
                        record,
                    )

                    append_csv(
                        path=
                            csv_path,

                        batch=
                            batch,

                        version=
                            version,

                        question_id=
                            str(
                                question_id
                            ),

                        source_file=
                            str(
                                question_path
                            ),

                        original_prompt=
                            original_prompt,

                        result=
                            result,
                    )

                    done.add(
                        key
                    )

                    total_completed += 1

                except Exception as exc:
                    total_failed += 1

                    print(
                        f"    [ERROR] "
                        f"run={run_id} "
                        f"Q{question_id}: "
                        f"{exc}"
                    )

                    append_jsonl(
                        error_path,
                        {
                            "status":
                                "error",

                            "architecture":
                                "arbitration",

                            "batch":
                                batch,

                            "version":
                                version,

                            "run_id":
                                run_id,

                            "question_id":
                                str(
                                    question_id
                                ),

                            "error":
                                repr(
                                    exc
                                ),
                        },
                    )

        print(
            f"  Raw trace: "
            f"{raw_path}"
        )

        print(
            f"  Flat CSV: "
            f"{csv_path}"
        )

    print()
    print("=" * 72)
    print("Arbitration summary")
    print("=" * 72)
    print(
        f"Completed: {total_completed}"
    )
    print(
        f"Failed:    {total_failed}"
    )

    if total_failed:
        print(
            "⚠️ Experiment finished "
            "with failed items."
        )
    else:
        print(
            "✅ Arbitration experiment "
            "completed without errors."
        )


if __name__ == "__main__":
    main()
