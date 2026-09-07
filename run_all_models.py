#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = REPO_ROOT / "Batch" / "scripts"


RUNNERS = {
    # Local models
    "deepseek": {
        "type": "local",
        "script": SCRIPT_ROOT / "run_deepseek.py",
    },
    "falcon3": {
        "type": "local",
        "script": SCRIPT_ROOT / "run_falcon3.py",
    },
    "gemma": {
        "type": "local",
        "script": SCRIPT_ROOT / "run_gemma.py",
    },
    "llama": {
        "type": "local",
        "script": SCRIPT_ROOT / "run_llama.py",
    },
    "qwen": {
        "type": "local",
        "script": SCRIPT_ROOT / "run_qwen.py",
    },

    # API models
    "claude": {
        "type": "api",
        "script": SCRIPT_ROOT / "run_claude.py",
    },
    "gemini": {
        "type": "api",
        "script": SCRIPT_ROOT / "run_gemini_api.py",
    },
    "gpt5": {
        "type": "api",
        "script": SCRIPT_ROOT / "run_gpt5.py",
    },
    "grok": {
        "type": "api",
        "script": SCRIPT_ROOT / "run_grok.py",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-aligned nine-model "
            "individual experiment."
        )
    )

    scope = parser.add_mutually_exclusive_group()

    scope.add_argument(
        "--local-only",
        action="store_true",
        help="Run only the five local open-source models.",
    )

    scope.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the four API models.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(RUNNERS),
        default=None,
        help=(
            "Run only selected models, e.g. "
            "--models llama qwen claude"
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        choices=range(1, 6),
        default=None,
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=range(1, 6),
        default=None,
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate model runners/configuration "
            "without performing inference."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--allow-model-redirect",
        action="store_true",
        help=(
            "Debugging only: allow current xAI redirect "
            "for retired grok-4-0709. Redirected results "
            "must NOT be used as historical paper results."
        ),
    )

    return parser.parse_args()


def selected_models(args):
    if args.models:
        return args.models

    if args.local_only:
        return [
            name
            for name, spec in RUNNERS.items()
            if spec["type"] == "local"
        ]

    if args.api_only:
        return [
            name
            for name, spec in RUNNERS.items()
            if spec["type"] == "api"
        ]

    return list(RUNNERS.keys())


def build_command(
    model_name,
    args,
):
    script = RUNNERS[
        model_name
    ]["script"]

    command = [
        sys.executable,
        str(script),
    ]

    # API runners support these CLI filters.
    #
    # The revised local runners are expected to use
    # their canonical Batch/questions configuration;
    # only forward arguments they actually support.
    is_api = (
        RUNNERS[
            model_name
        ]["type"]
        == "api"
    )

    if is_api:

        if args.batch is not None:
            command += [
                "--batch",
                str(args.batch),
            ]

        if args.version is not None:
            command += [
                "--version",
                str(args.version),
            ]

        if args.num_runs is not None:
            command += [
                "--num-runs",
                str(args.num_runs),
            ]

        if args.validate_only:
            command.append(
                "--validate-only"
            )

        if (
            model_name == "grok"
            and
            args.allow_model_redirect
        ):
            command.append(
                "--allow-model-redirect"
            )

    return command


def main():
    args = parse_args()

    models = selected_models(
        args
    )

    print("=" * 72)
    print("NCO Benchmark — Individual Models")
    print("=" * 72)

    print(
        "Canonical benchmark:"
    )

    print(
        f"  {REPO_ROOT / 'Batch' / 'questions'}"
    )

    print()

    print(
        f"Selected models ({len(models)}):"
    )

    for model in models:
        spec = RUNNERS[
            model
        ]

        print(
            f"  - {model:<10} "
            f"[{spec['type']}]"
        )

    print()

    for model in models:

        script = RUNNERS[
            model
        ]["script"]

        if not script.exists():
            raise FileNotFoundError(
                f"Missing runner: {script}"
            )

        command = build_command(
            model,
            args,
        )

        print(
            f"[RUN] {model}"
        )

        print(
            "      "
            + " ".join(command)
        )

        if args.dry_run:
            continue

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

    print()

    if args.dry_run:
        print(
            "✅ Dry run completed."
        )
    elif args.validate_only:
        print(
            "✅ Selected configurations validated."
        )
    else:
        print(
            "✅ Selected individual-model "
            "experiments completed."
        )


if __name__ == "__main__":
    main()
