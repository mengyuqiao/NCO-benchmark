#!/usr/bin/env python3

from run_api_common import run_cli


if __name__ == "__main__":
    run_cli(
        experiment_name="gemini",
        provider="gemini",
        model_id="gemini-3.1-pro-preview",
    )
