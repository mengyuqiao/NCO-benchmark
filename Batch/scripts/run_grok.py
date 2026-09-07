#!/usr/bin/env python3

from run_api_common import run_cli


if __name__ == "__main__":
    run_cli(
        experiment_name="grok",
        provider="xai",
        model_id="grok-4-0709",
    )
