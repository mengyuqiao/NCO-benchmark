# NCO-benchmark

A benchmark for evaluating how Large Language Models (LLMs) respond to medical questions designed to elicit either **non-compliant outcomes (NCO)** or **positive outcomes**. The framework runs multiple open-source models side-by-side across a battery of question sets, repeats each run over multiple rounds for statistical stability, and logs results for downstream analysis.

## Overview

NCO-benchmark stress-tests medical reasoning and safety behavior of LLMs using two complementary question suites:

- **NCO questions** — prompts crafted to probe situations where a model might produce a non-compliant or undesired medical response.
- **Positive questions** — counterpart medical prompts where a well-behaved model should produce helpful, compliant answers.

Each suite contains 5 versions (`v1` through `v5`), enabling progression-style or robustness-style evaluation across prompt variants.

## Repository Structure

```
NCO-benchmark/
├── Questions/
│   ├── NCO/                  # NCO medical questions, v1–v5 (72 questions)
│   └── Positive/             # Positive medical questions, v1–v5 (27 questions)
├── Batch/                    # Batched-question dataset (see below)
├── PEG/                      # Group-level LLM evaluation (see below)
├── ability-test/             # Single-model ability tests
├── llamacpp/                 # llama.cpp-based runners / integration
├── log_utils.py              # Logging helpers
├── model_loader.py           # Hugging Face model loading utilities
├── question_loader.py        # Loads question files into runnable prompts
├── peg_core.py               # Core PEG evaluation logic
├── main_multi_model.py       # Main entry point — runs all models over all questions
├── run_all_models.py         # Launcher that configures models/devices and calls main
└── README.md
```

## Models Evaluated

The default launcher (`run_all_models.py`) benchmarks the following Hugging Face models:

- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `tiiuae/Falcon3-7B-Instruct`
- `google/gemma-3-4b-it`
- `Qwen/Qwen2.5-7B-Instruct`

Each model is pinned to its own CUDA device (GPUs `0`–`3` by default) so all four can be evaluated in parallel.

## Requirements

- Python 3.9+
- CUDA-capable GPUs (4 recommended for the default configuration)
- PyTorch with CUDA support
- `transformers`, `accelerate`, and other Hugging Face dependencies
- Sufficient disk space and VRAM to host the four models above
- Hugging Face account + token for any gated models (e.g. Gemma)

Install dependencies with:

```bash
pip install torch transformers accelerate
```

(A `requirements.txt` can be added as the project stabilizes.)

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/mengyuqiao/NCO-benchmark.git
   cd NCO-benchmark
   ```

2. Authenticate with Hugging Face if needed:

   ```bash
   huggingface-cli login
   ```

3. Run the full benchmark across all configured models and question sets:

   ```bash
   python run_all_models.py
   ```

   This will:
   - Set `CUDA_VISIBLE_DEVICES` to GPUs 0–3
   - Load all four models
   - Run each model against the 10 question files (5 NCO + 5 Positive)
   - Repeat each run for `NUM_ROUNDS = 3` rounds
   - Delegate actual execution to `main_multi_model.py`

## Configuration

Key variables are defined at the top of `run_all_models.py` and passed as environment variables to `main_multi_model.py`:

| Variable           | Description                                        | Default         |
| ------------------ | -------------------------------------------------- | --------------- |
| `HF_MODELS`        | Comma-separated list of Hugging Face model IDs     | 4 models above  |
| `HF_DEVICES`       | Comma-separated CUDA device indices                | `0,1,2,3`       |
| `NUM_ROUNDS`       | Number of evaluation rounds per (model, question) | `3`             |
| `QUESTION_PATHS`   | Comma-separated list of question file paths       | 10 files        |

Edit `run_all_models.py` to add/remove models, change device assignments, adjust round counts, or point at a different question suite.

## Question Suites

Questions live under `Questions/` as plain text files, one prompt per line (or per block, depending on the loader). The default run evaluates:

- `Questions/NCO/nco_v{1..5}_questions.txt`
- `Questions/Positive/medical_questions_v{1..5}.txt`

You can add your own suites by dropping new `.txt` files into either folder and extending `QUESTION_PATHS` in `run_all_models.py`.

## Outputs

Generated outputs include:

- **`results/`** — per-model, per-question-set result files containing model responses and any computed scores.
- **`logs/`** — run logs produced via `log_utils.py`, useful for debugging and auditing which prompts were sent to which model.

## Modules

- **`model_loader.py`** — Loads Hugging Face models and tokenizers onto specified devices.
- **`question_loader.py`** — Parses question files and yields prompts to the evaluation loop.
- **`peg_core.py`** — Implements the PEG (Prompt Evaluation Graph) evaluation flow used to score or classify model outputs.
- **`log_utils.py`** — Shared logging setup for reproducible run records.
- **`main_multi_model.py`** — Orchestrates the multi-model, multi-round, multi-question evaluation.
- **`run_all_models.py`** — Thin launcher that sets environment variables and kicks off `main_multi_model.py`.

## Notes

- Some inline comments in the source are in Chinese; the code itself is language-agnostic.
- The default configuration assumes a 4-GPU machine. On a single-GPU or CPU-only box, reduce `HF_MODELS` and `DEVICES` accordingly in `run_all_models.py`.
- Gated models (e.g. Gemma) require accepting the model license on Hugging Face before first run.

## License

No license file is currently provided in the repository. If you intend to use or redistribute this benchmark, please contact the repository owner.

## Contact

Repository maintained by [@mengyuqiao](https://github.com/mengyuqiao). Issues and pull requests are welcome via the [GitHub repo](https://github.com/mengyuqiao/NCO-benchmark).