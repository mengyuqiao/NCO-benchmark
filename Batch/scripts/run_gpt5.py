#!/usr/bin/env python3

from run_api_common import run_cli


if __name__ == "__main__":
    run_cli(
        experiment_name="gpt5",
        provider="openai",
        model_id="gpt-5-chat-latest",
    )
