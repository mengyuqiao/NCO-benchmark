#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from model_loader import load_agents
from peg_core import PANEL_SIZE, run_rolling_review
from question_loader import load_question_blocks


# ============================================================
# Paths
# ============================================================

PEG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PEG_DIR.parent

DEFAULT_QUESTIONS_ROOT = (
    REPO_ROOT
    / "Batch"
    / "questions"
)

DEFAULT_OUT_ROOT = (
    PEG_DIR
    / "results"
    / "rolling_review"
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-aligned five-agent "
            "rolling-review experiment."
        )
    )

    parser.add_argument(
        "--panel",
        type=Path,
        required=True,
        help=(
            "Path to a panel JSON configuration, "
            "e.g. PEG/panels/mixed.json"
        ),
    )

    parser.add_argument(
        "--questions-root",
        type=Path,
        default=DEFAULT_QUESTIONS_ROOT,
        help="Canonical benchmark question root.",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output directory.",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help=(
            "Override num_runs in the panel config. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        choices=range(1, 6),
        metavar="1-5",
        help="Run only one benchmark batch.",
    )

    parser.add_argument(
        "--version",
        type=int,
        default=None,
        choices=range(1, 6),
        metavar="1-5",
        help="Run only one prompt version.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate and print the resolved panel "
            "configuration without loading models."
        ),
    )

    return parser.parse_args()


# ============================================================
# Panel configuration
# ============================================================

def load_panel_config(
    panel_path: Path,
) -> Dict:
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel config not found: {panel_path}"
        )

    with panel_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    return config


def resolve_model_id(
    spec: Dict,
) -> str:
    """
    Resolve model_id directly or from model_id_env.

    This resolves only the model identifier.
    API credentials are never stored here.
    """
    model_id = str(
        spec.get(
            "model_id",
            ""
        )
    ).strip()

    if model_id:
        return model_id

    model_id_env = str(
        spec.get(
            "model_id_env",
            ""
        )
    ).strip()

    if model_id_env:
        value = os.getenv(
            model_id_env,
            ""
        ).strip()

        if value:
            return value

        raise RuntimeError(
            f"Environment variable "
            f"{model_id_env} is required "
            f"for agent {spec.get('agent_id')}."
        )

    raise ValueError(
        f"Agent {spec.get('agent_id')} "
        "has neither model_id nor model_id_env."
    )


def validate_panel_config(
    config: Dict,
    require_model_ids: bool = True,
):
    panel_name = str(
        config.get(
            "panel_name",
            ""
        )
    ).strip()

    if not panel_name:
        raise ValueError(
            "panel_name is required."
        )

    architecture = str(
        config.get(
            "architecture",
            ""
        )
    ).strip()

    if architecture != "rolling_review":
        raise ValueError(
            "This runner only supports "
            "architecture='rolling_review'."
        )

    agents = config.get(
        "agents",
        []
    )

    if len(agents) != PANEL_SIZE:
        raise ValueError(
            f"{panel_name} must contain exactly "
            f"{PANEL_SIZE} agents; "
            f"found {len(agents)}."
        )

    agent_ids = [
        str(
            spec.get(
                "agent_id",
                ""
            )
        ).strip()
        for spec in agents
    ]

    if any(
        not aid
        for aid in agent_ids
    ):
        raise ValueError(
            "Every panel agent needs agent_id."
        )

    if len(set(agent_ids)) != PANEL_SIZE:
        raise ValueError(
            "All five agent IDs must be unique."
        )

    supported_providers = {
        "hf",
        "huggingface",
        "hf_multimodal",
        "huggingface_multimodal",
        "anthropic",
        "claude",
        "gemini",
        "google",
    }

    for spec in agents:
        aid = spec["agent_id"]

        provider = str(
            spec.get(
                "provider",
                ""
            )
        ).strip().lower()

        if provider not in supported_providers:
            raise ValueError(
                f"Unsupported provider "
                f"'{provider}' for {aid}."
            )

        if provider in {
            "hf",
            "huggingface",
            "hf_multimodal",
            "huggingface_multimodal",
        }:
            if "device" not in spec:
                raise ValueError(
                    f"HF agent {aid} "
                    "requires device."
                )

        if require_model_ids:
            resolve_model_id(spec)

    num_runs = int(
        config.get(
            "num_runs",
            10
        )
    )

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be positive."
        )


def resolved_panel_manifest(
    config: Dict,
) -> Dict:
    """
    Produce a reproducibility manifest containing
    the exact model IDs used at runtime.

    API keys are never included.
    """
    resolved_agents = []

    for position, spec in enumerate(
        config["agents"],
        start=1,
    ):
        resolved_agents.append({
            "position": position,
            "agent_id": spec["agent_id"],
            "provider": spec["provider"],
            "model_id": resolve_model_id(
                spec
            ),
            "device": spec.get(
                "device"
            ),
        })

    return {
        "panel_name":
            config["panel_name"],

        "architecture":
            config["architecture"],

        "agents":
            resolved_agents,
    }


# ============================================================
# Benchmark discovery
# ============================================================

def discover_question_files(
    root: Path,
    batch_filter: Optional[int],
    version_filter: Optional[int],
) -> List[Path]:
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
# Resume / output helpers
# ============================================================

def completed_keys(
    jsonl_path: Path,
):
    done = set()

    if not jsonl_path.exists():
        return done

    with jsonl_path.open(
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
            "panel_name",
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

            "final_agent_id",
            "final_response",
        ])


def append_csv(
    path: Path,
    panel_name: str,
    batch: str,
    version: str,
    question_id: str,
    source_file: str,
    original_prompt: str,
    result: Dict,
):
    trace = result["trace"]

    if len(trace) != PANEL_SIZE:
        raise RuntimeError(
            f"Expected {PANEL_SIZE} "
            f"trace entries, received "
            f"{len(trace)}."
        )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            panel_name,
            batch,
            version,
            result["run_id"],
            question_id,
            source_file,
            original_prompt,

            trace[0]["agent_id"],
            trace[0]["response"],

            trace[1]["agent_id"],
            trace[1]["response"],

            trace[2]["agent_id"],
            trace[2]["response"],

            trace[3]["agent_id"],
            trace[3]["response"],

            trace[4]["agent_id"],
            trace[4]["response"],

            result["final_agent_id"],
            result["final_response"],
        ])


def write_manifest(
    panel_dir: Path,
    panel_config_path: Path,
    config: Dict,
    num_runs: int,
):
    manifest = (
        resolved_panel_manifest(
            config
        )
    )

    manifest.update({
        "num_runs":
            num_runs,

        "panel_config":
            str(
                panel_config_path.resolve()
            ),

        "generation_config": {
            "max_new_tokens": 2048,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
        },
    })

    path = (
        panel_dir
        / "run_manifest.json"
    )

    with path.open(
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

    config = load_panel_config(
        args.panel
    )

    # In validate-only mode we allow Claude/Gemini
    # model environment variables to still be absent.
    validate_panel_config(
        config,
        require_model_ids=(
            not args.validate_only
        ),
    )

    panel_name = config[
        "panel_name"
    ]

    num_runs = (
        args.num_runs
        if args.num_runs is not None
        else int(
            config.get(
                "num_runs",
                10
            )
        )
    )

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be positive."
        )

    print("=" * 72)
    print("NCO Rolling Review")
    print("=" * 72)
    print(
        f"Panel:        "
        f"{panel_name}"
    )
    print(
        f"Architecture: "
        f"{config['architecture']}"
    )
    print(
        f"Panel size:   "
        f"{len(config['agents'])}"
    )
    print(
        f"Runs:         "
        f"{num_runs}"
    )
    print()

    print("Agent order:")

    for position, spec in enumerate(
        config["agents"],
        start=1,
    ):
        model_display = (
            spec.get("model_id")
            or
            f"${spec.get('model_id_env')}"
        )

        print(
            f"  {position}: "
            f"{spec['agent_id']} | "
            f"{spec['provider']} | "
            f"{model_display}"
        )

    if args.validate_only:
        print()
        print(
            "✅ Panel configuration "
            "is structurally valid."
        )

        unresolved = [
            spec.get(
                "model_id_env"
            )
            for spec
            in config["agents"]
            if (
                not spec.get("model_id")
                and
                not os.getenv(
                    spec.get(
                        "model_id_env",
                        ""
                    )
                )
            )
        ]

        unresolved = sorted(
            set(
                x
                for x in unresolved
                if x
            )
        )

        if unresolved:
            print(
                "ℹ️ Runtime model IDs "
                "still need:"
            )

            for name in unresolved:
                print(
                    f"   {name}"
                )

        return

    # --------------------------------------------------------
    # Resolve exact runtime model IDs.
    # --------------------------------------------------------

    resolved_config = dict(
        config
    )

    resolved_config["agents"] = []

    for spec in config["agents"]:
        new_spec = dict(
            spec
        )

        new_spec["model_id"] = (
            resolve_model_id(
                spec
            )
        )

        resolved_config[
            "agents"
        ].append(
            new_spec
        )

    # --------------------------------------------------------
    # Load models / API clients.
    # --------------------------------------------------------

    agents = load_agents(
        resolved_config["agents"]
    )

    agent_order = [
        spec["agent_id"]
        for spec
        in resolved_config["agents"]
    ]

    # --------------------------------------------------------
    # Output directory.
    # --------------------------------------------------------

    panel_dir = (
        args.out_root
        / panel_name
    )

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_manifest(
        panel_dir=panel_dir,
        panel_config_path=args.panel,
        config=resolved_config,
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

    for file_index, question_path in enumerate(
        question_files,
        start=1,
    ):
        batch, version = (
            infer_batch_version(
                question_path
            )
        )

        questions = (
            load_question_blocks(
                question_path
            )
        )

        if not questions:
            raise RuntimeError(
                f"No questions parsed from "
                f"{question_path}"
            )

        file_tag = (
            f"{batch}_{version}"
        )

        jsonl_path = (
            panel_dir
            / f"{file_tag}_raw.jsonl"
        )

        csv_path = (
            panel_dir
            / f"{file_tag}_responses.csv"
        )

        error_path = (
            panel_dir
            / f"{file_tag}_errors.jsonl"
        )

        write_csv_header_if_needed(
            csv_path
        )

        done = completed_keys(
            jsonl_path
        )

        print()
        print(
            f"[{file_index}/"
            f"{len(question_files)}] "
            f"{batch} {version} | "
            f"{len(questions)} questions"
        )

        # ----------------------------------------------------
        # Independent runs.
        # ----------------------------------------------------

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
                    result = (
                        run_rolling_review(
                            original_prompt=
                                original_prompt,

                            agents=
                                agents,

                            agent_order=
                                agent_order,

                            run_id=
                                run_id,
                        )
                    )

                    record = {
                        "status":
                            "completed",

                        "architecture":
                            "rolling_review",

                        "panel_name":
                            panel_name,

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
                        jsonl_path,
                        record,
                    )

                    append_csv(
                        path=csv_path,
                        panel_name=panel_name,
                        batch=batch,
                        version=version,
                        question_id=str(
                            question_id
                        ),
                        source_file=str(
                            question_path
                        ),
                        original_prompt=
                            original_prompt,
                        result=result,
                    )

                    done.add(
                        key
                    )

                except Exception as exc:
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
                                "rolling_review",

                            "panel_name":
                                panel_name,

                            "batch":
                                batch,

                            "version":
                                version,

                            "question_id":
                                str(
                                    question_id
                                ),

                            "run_id":
                                run_id,

                            "error":
                                repr(exc),
                        },
                    )

        print(
            f"  ✓ Raw trace: "
            f"{jsonl_path}"
        )

        print(
            f"  ✓ Flat CSV: "
            f"{csv_path}"
        )

    print()
    print(
        "✅ Rolling-review experiment "
        "completed."
    )


if __name__ == "__main__":
    main()
