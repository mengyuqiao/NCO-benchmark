#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from pathlib import Path


TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 2048
DEFAULT_NUM_RUNS = 10

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
QUESTIONS_ROOT = REPO_ROOT / "Batch" / "questions"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


# ============================================================
# Benchmark parsing
# ============================================================

def load_question_blocks(path: Path):
    text = path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"(?m)^Question\s+(\d+)"
    )

    matches = list(pattern.finditer(text))

    if not matches:
        raise ValueError(
            f"No questions found in {path}"
        )

    blocks = []

    for i, match in enumerate(matches):
        qid = int(match.group(1))

        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        block = text[start:end].strip()

        blocks.append(
            (qid, block)
        )

    return blocks


def discover_question_files(
    batch_filter=None,
    version_filter=None,
):
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

    files = []

    for batch in batches:
        for version in versions:
            path = (
                QUESTIONS_ROOT
                / f"Batch{batch}"
                / f"medical_questions_v{version}.txt"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing benchmark file: {path}"
                )

            files.append(
                (batch, version, path)
            )

    return files


# ============================================================
# Provider implementations
# ============================================================

class AnthropicBackend:

    def __init__(self, model_id):
        from anthropic import Anthropic

        api_key = os.getenv(
            "ANTHROPIC_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set."
            )

        self.model_id = model_id

        self.client = Anthropic(
            api_key=api_key
        )

    def generate(self, prompt):

        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text_parts = []

        for block in message.content:
            if getattr(
                block,
                "type",
                None,
            ) == "text":
                text_parts.append(
                    block.text
                )

        text = "".join(
            text_parts
        ).strip()

        if not text:
            raise RuntimeError(
                "Anthropic returned empty text."
            )

        usage = getattr(
            message,
            "usage",
            None,
        )

        return {
            "raw_response":
                text,

            "model_id_reported":
                getattr(
                    message,
                    "model",
                    self.model_id,
                ),

            "input_tokens":
                getattr(
                    usage,
                    "input_tokens",
                    None,
                ),

            "output_tokens":
                getattr(
                    usage,
                    "output_tokens",
                    None,
                ),
        }


class GeminiBackend:

    def __init__(self, model_id):

        from google import genai
        from google.genai import types

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.model_id = model_id
        self.types = types

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(self, prompt):

        response = (
            self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    candidate_count=1,
                    max_output_tokens=
                        MAX_NEW_TOKENS,
                    temperature=
                        TEMPERATURE,
                    top_p=
                        TOP_P,
                ),
            )
        )

        text = (
            getattr(
                response,
                "text",
                None,
            )
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "Gemini returned empty text."
            )

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        return {
            "raw_response":
                text,

            "model_id_reported":
                (
                    getattr(
                        response,
                        "model_version",
                        None,
                    )
                    or self.model_id
                ),

            "input_tokens":
                getattr(
                    usage,
                    "prompt_token_count",
                    None,
                ),

            "output_tokens":
                getattr(
                    usage,
                    "candidates_token_count",
                    None,
                ),
        }


class OpenAIBackend:

    def __init__(self, model_id):

        from openai import OpenAI

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        self.model_id = model_id

        self.client = OpenAI(
            api_key=api_key
        )

    def generate(self, prompt):

        response = (
            self.client.responses.create(
                model=self.model_id,
                input=prompt,
                max_output_tokens=
                    MAX_NEW_TOKENS,
                temperature=
                    TEMPERATURE,
                top_p=
                    TOP_P,
            )
        )

        text = (
            getattr(
                response,
                "output_text",
                None,
            )
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "OpenAI returned empty text."
            )

        usage = getattr(
            response,
            "usage",
            None,
        )

        return {
            "raw_response":
                text,

            "model_id_reported":
                getattr(
                    response,
                    "model",
                    self.model_id,
                ),

            "input_tokens":
                getattr(
                    usage,
                    "input_tokens",
                    None,
                ),

            "output_tokens":
                getattr(
                    usage,
                    "output_tokens",
                    None,
                ),
        }


class XAIBackend:

    def __init__(
        self,
        model_id,
        allow_redirect=False,
    ):

        from openai import OpenAI

        api_key = os.getenv(
            "XAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY is not set."
            )

        self.model_id = model_id
        self.allow_redirect = allow_redirect

        # grok-4-0709 was retired on 2026-05-15.
        #
        # xAI now redirects requests using that slug
        # to grok-4.3. Such redirected generations
        # are NOT exact reproductions of the paper model.
        if (
            self.model_id
            == "grok-4-0709"
            and
            not self.allow_redirect
        ):
            raise RuntimeError(
                "grok-4-0709 is retired and xAI "
                "currently redirects this model slug "
                "to grok-4.3. Exact paper-model "
                "reproduction is therefore unavailable. "
                "Use historical raw outputs for the "
                "paper result. "
                "--allow-model-redirect is for "
                "debugging only."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def generate(self, prompt):

        response = (
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=
                    TEMPERATURE,
                top_p=
                    TOP_P,
                max_tokens=
                    MAX_NEW_TOKENS,
                n=1,
            )
        )

        text = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "xAI returned empty text."
            )

        usage = getattr(
            response,
            "usage",
            None,
        )

        reported = getattr(
            response,
            "model",
            self.model_id,
        )

        return {
            "raw_response":
                text,

            "model_id_reported":
                reported,

            "input_tokens":
                (
                    getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    )
                    if usage is not None
                    else None
                ),

            "output_tokens":
                (
                    getattr(
                        usage,
                        "completion_tokens",
                        None,
                    )
                    if usage is not None
                    else None
                ),
        }


def build_backend(
    provider,
    model_id,
    allow_redirect=False,
):

    if provider == "anthropic":
        return AnthropicBackend(
            model_id
        )

    if provider == "gemini":
        return GeminiBackend(
            model_id
        )

    if provider == "openai":
        return OpenAIBackend(
            model_id
        )

    if provider == "xai":
        return XAIBackend(
            model_id,
            allow_redirect=
                allow_redirect,
        )

    raise ValueError(
        f"Unsupported provider: {provider}"
    )


# ============================================================
# Output
# ============================================================

CSV_FIELDS = [
    "provider",
    "model_id_requested",
    "model_id_reported",
    "batch",
    "version",
    "run_id",
    "question_id",
    "raw_response",
    "input_tokens",
    "output_tokens",
    "source_file",
]


def completed_keys(path):

    done = set()

    if not path.exists():
        return done

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            done.add(
                (
                    int(row["run_id"]),
                    int(row["question_id"]),
                )
            )

    return done


def append_csv(
    path,
    row,
):

    first_write = (
        not path.exists()
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        if first_write:
            writer.writeheader()

        writer.writerow(row)


def append_error(
    path,
    row,
):

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# CLI
# ============================================================

def run_cli(
    experiment_name,
    provider,
    model_id,
):

    parser = argparse.ArgumentParser()

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
        default=DEFAULT_NUM_RUNS,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--allow-model-redirect",
        action="store_true",
        help=(
            "Debugging only. Never use redirected "
            "model output for paper reproduction."
        ),
    )

    args = parser.parse_args()

    if args.num_runs <= 0:
        raise ValueError(
            "num-runs must be positive."
        )

    files = discover_question_files(
        batch_filter=args.batch,
        version_filter=args.version,
    )

    print("=" * 72)
    print("NCO API Individual Model")
    print("=" * 72)

    print(
        f"Experiment: {experiment_name}"
    )

    print(
        f"Provider:   {provider}"
    )

    print(
        f"Model ID:   {model_id}"
    )

    print(
        f"Temperature:{TEMPERATURE}"
    )

    print(
        f"Top-p:      {TOP_P}"
    )

    print(
        f"Max tokens: {MAX_NEW_TOKENS}"
    )

    print(
        f"Runs:       {args.num_runs}"
    )

    print(
        f"Files:      {len(files)}"
    )

    if args.validate_only:

        print()

        for batch, version, path in files:

            questions = (
                load_question_blocks(
                    path
                )
            )

            print(
                f"  Batch{batch} v{version}: "
                f"{len(questions)} questions"
            )

        print()
        print(
            "✅ Configuration and benchmark "
            "files validated."
        )

        return

    backend = build_backend(
        provider=provider,
        model_id=model_id,
        allow_redirect=
            args.allow_model_redirect,
    )

    model_root = (
        args.out_root
        / experiment_name
    )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_completed = 0
    total_failed = 0

    for (
        batch,
        version,
        path,
    ) in files:

        questions = (
            load_question_blocks(
                path
            )
        )

        batch_dir = (
            model_root
            / f"Batch{batch}"
        )

        batch_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_csv = (
            batch_dir
            / (
                f"{experiment_name}_"
                f"v{version}_"
                f"{args.num_runs}runs.csv"
            )
        )

        error_file = (
            batch_dir
            / (
                f"{experiment_name}_"
                f"v{version}_errors.jsonl"
            )
        )

        done = completed_keys(
            out_csv
        )

        print()
        print(
            f"Batch{batch} v{version} | "
            f"{len(questions)} questions"
        )

        for run_id in range(
            1,
            args.num_runs + 1,
        ):

            print(
                f"  ▶ Independent run "
                f"{run_id}/{args.num_runs}"
            )

            for (
                question_id,
                prompt,
            ) in questions:

                key = (
                    run_id,
                    question_id,
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
                        backend.generate(
                            prompt
                        )
                    )

                    append_csv(
                        out_csv,
                        {
                            "provider":
                                provider,

                            "model_id_requested":
                                model_id,

                            "model_id_reported":
                                result[
                                    "model_id_reported"
                                ],

                            "batch":
                                f"Batch{batch}",

                            "version":
                                f"v{version}",

                            "run_id":
                                run_id,

                            "question_id":
                                question_id,

                            "raw_response":
                                result[
                                    "raw_response"
                                ],

                            "input_tokens":
                                result[
                                    "input_tokens"
                                ],

                            "output_tokens":
                                result[
                                    "output_tokens"
                                ],

                            "source_file":
                                str(path),
                        },
                    )

                    done.add(key)

                    total_completed += 1

                except Exception as exc:

                    total_failed += 1

                    print(
                        f"    [ERROR] "
                        f"run={run_id} "
                        f"Q{question_id}: "
                        f"{exc}"
                    )

                    append_error(
                        error_file,
                        {
                            "provider":
                                provider,

                            "model_id_requested":
                                model_id,

                            "batch":
                                f"Batch{batch}",

                            "version":
                                f"v{version}",

                            "run_id":
                                run_id,

                            "question_id":
                                question_id,

                            "error":
                                repr(exc),
                        },
                    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    print(
        f"Completed: {total_completed}"
    )

    print(
        f"Failed:    {total_failed}"
    )

    if total_failed:
        print(
            "⚠️ Finished with failures."
        )
    else:
        print(
            "✅ Completed without errors."
        )
