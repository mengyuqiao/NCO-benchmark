# NCO Benchmark

This repository contains the benchmark, inference pipelines, multi-agent implementations, evaluation code, and released result artifacts for studying whether large language models can distinguish **negative control outcomes (NCOs)** from **positive control outcomes (PCOs)** in a causal-reasoning task comparing GLP-1 receptor agonists (GLP-1RA) with SGLT2 inhibitors (SGLT2i).

The study evaluates:

- nine individual LLMs;
- four five-agent rolling-review panels;
- a five-agent arbitration architecture with a Claude judge; and
- additional compute-enhanced single-model baselines used to examine whether multi-agent gains can be explained by test-time compute alone.

The canonical paper benchmark and canonical paper-aligned implementations are explicitly identified below. Several historical/development directories are retained for provenance but should not be used to reproduce the reported experiments.

---

## 1. Canonical benchmark

The benchmark used for the paper is:

```text
Batch/questions/
├── Batch1/
├── Batch2/
├── Batch3/
├── Batch4/
└── Batch5/
```

It contains **54 outcomes** divided into five clinical groups:

| Batch | Clinical group | Outcomes | NCO (`no`) | PCO (`yes`) |
|---|---|---:|---:|---:|
| Batch1 | Gastrointestinal & Nutritional | 12 | 7 | 5 |
| Batch2 | Psychiatric & Behavioral | 11 | 6 | 5 |
| Batch3 | Musculoskeletal | 10 | 7 | 3 |
| Batch4 | Ophthalmologic | 11 | 6 | 5 |
| Batch5 | Cardiovascular | 10 | 4 | 6 |
| **Total** |  | **54** | **30** | **24** |

Each outcome is evaluated under five prompt conditions:

| Version | Prompt condition |
|---|---|
| `v1` | Plain |
| `v2` | System prompt |
| `v3` | 2-shot |
| `v4` | 5-shot |
| `v5` | 10-shot |

Each batch contains one `medical_questions_v*.txt` file and a matching `medical_answers_v*.txt` file for each prompt version.

### Data provenance

The clinical cohort description and aggregate outcome-incidence estimates used in the benchmark prompts were provided directly by Penn Medicine. The underlying patient-level EHR data and institution-specific extraction procedures are not included in this repository.

### Terminology

The original experimental prompts use the legacy term `OOI` ("Positive Outcome") for the positive-control class. The manuscript and supplementary material use `PCO` ("positive control outcome"). The original prompt wording is retained to preserve the exact experimental inputs; `OOI` and `PCO` refer to the same class here.

Under the paper's evaluation convention:

- `no` = **negative control outcome (NCO)**;
- `yes` = **positive control outcome (PCO)**; and
- **NCO (`no`) is treated as the positive class** when computing Precision, Recall, and F1.

---

## 2. Models

### Individual models

| Display name | Exact model identifier | Access |
|---|---|---|
| Claude | `claude-sonnet-4-6` | Anthropic API |
| Gemini3 | `gemini-3.1-pro-preview` | Gemini API |
| GPT5 | `gpt-5-chat-latest` | OpenAI API |
| Grok | `grok-4-0709` | xAI API |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Hugging Face |
| Qwen | `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face |
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face |
| Falcon3 | `tiiuae/Falcon3-7B-Instruct` | Hugging Face |
| Gemma | `google/gemma-3-4b-it` | Hugging Face |

`grok-4-0709` is the historical identifier used in the study. The endpoint may no longer reproduce the original model because the model has since been retired or redirected by the provider. Redirected outputs must not be represented as exact reproductions of the historical Grok experiment.

API credentials are supplied through environment variables and are never stored in the repository.

---

## 3. Main inference settings

The paper-aligned individual and multi-agent experiments use:

```text
temperature       = 1.0
top_p             = 1.0
max_new_tokens    = 2048
independent runs  = 10
```

An independent run is a separate model generation or complete multi-agent trajectory. Multiple return sequences from one generation call are not treated as independent paper runs.

The machine-readable paper configuration is also recorded in:

```text
configs/paper.yaml
```

---

## 4. Individual-model experiments

The top-level launcher is:

```bash
python run_all_models.py
```

Useful options include:

```bash
# Inspect commands without inference
python run_all_models.py --dry-run

# Run only local Hugging Face models
python run_all_models.py --local-only

# Run only API models
python run_all_models.py --api-only

# Select models
python run_all_models.py --models llama qwen claude
```

Model-specific runners are under:

```text
Batch/scripts/
├── run_deepseek.py
├── run_falcon3.py
├── run_gemma.py
├── run_llama.py
├── run_qwen.py
├── run_claude.py
├── run_gemini_api.py
├── run_gpt5.py
├── run_grok.py
└── run_api_common.py
```

The current local runners write explicit `run_id` and `question_id` fields together with the original prompt, full model response, and parsed answer.

### API credentials

Set only the keys required for the models being executed:

```bash
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export XAI_API_KEY="..."
```

---

## 5. Multi-agent experiments

The canonical multi-agent code is under `PEG/`.

### 5.1 Rolling review

Rolling review uses five agents sequentially:

```text
Original prompt
     │
     ▼
 Agent 1
     │ complete response
     ▼
 Agent 2
     │ complete revised/affirmed response
     ▼
 Agent 3
     │
     ▼
 Agent 4
     │
     ▼
 Agent 5  ──> final panel response
```

Agent 1 receives the original benchmark prompt. Agents 2–5 receive the original prompt plus the complete response from the immediately preceding agent and are asked to critique, revise, or affirm it. The fifth agent's response is the final panel response.

Canonical implementation:

```text
PEG/peg_core.py
PEG/model_loader.py
PEG/main_multi_model.py
```

Panel definitions:

```text
PEG/panels/
├── mixed.json
├── claude.json
├── gemini.json
└── qwen.json
```

The four reported rolling-review panels are:

- **Mixed-Panel**: Llama + Qwen + Gemma + Falcon3 + DeepSeek;
- **Claude-Panel**: five Claude agents;
- **Gemini-Panel**: five Gemini agents; and
- **Qwen-Panel**: five logical Qwen agents.

Example:

```bash
python PEG/main_multi_model.py --panel PEG/panels/mixed.json
```

A minimal smoke test is:

```bash
python PEG/main_multi_model.py \
  --panel PEG/panels/mixed.json \
  --batch 1 \
  --version 1 \
  --num-runs 1
```

Rolling-review outputs include the response from every agent, the final agent response, explicit run/question identifiers, and the original prompt.

### 5.2 Arbitration

Arbitration uses five independent agents followed by a Claude judge:

```text
Agent 1 ─┐
Agent 2 ─┤
Agent 3 ─┼──> Claude Sonnet 4.6 judge ──> final response
Agent 4 ─┤
Agent 5 ─┘
```

The five agents independently receive the same original benchmark prompt and do not see one another's outputs. The judge receives the original prompt and all five complete responses simultaneously.

Canonical implementation:

```text
PEG/arbitration_core.py
PEG/main_arbitration.py
```

Example:

```bash
python PEG/main_arbitration.py \
  --batch 1 \
  --version 1 \
  --num-runs 1
```

Claude is accessed through the Anthropic API. The API key is provided through `ANTHROPIC_API_KEY` and is not stored in the repository.

---

## 6. Evaluation

The canonical evaluator is:

```text
Batch/get_accuracy.py
```

It computes:

- Accuracy;
- Precision;
- Recall;
- F1;
- parse rate; and
- explicit parse failures.

### Evaluation rules

1. NCO (`no`) is the positive class.
2. Independent runs are identified only by explicit `run_id` values.
3. `response_id` and row order are not used to infer independent runs.
4. Binary parsing requires an explicit yes/no decision; incidental yes/no tokens inside free-form reasoning are not sufficient.
5. Unparseable outputs are retained and are not silently converted to `yes` or `no`.
6. Accuracy uses the complete benchmark denominator; an unparseable output is counted as incorrect.
7. For each independent run, metrics are computed over the **full 54-outcome benchmark**.
8. The reported mean and **sample standard deviation** are then computed across the ten independent runs.

Example:

```bash
python Batch/get_accuracy.py \
  --results-root results \
  --gold-root Batch/questions \
  --expected-runs 10
```

The evaluator writes:

```text
evaluation/
├── item_predictions.csv
├── parse_failures.csv
├── per_batch_run_metrics.csv
├── per_run_metrics.csv
└── summary_metrics.csv
```

`--allow-incomplete` is intended only for debugging partial runs and should not be used for final paper tables.

---

## 7. Supplementary and batch-specific results

The paper reports aggregate metrics over the complete benchmark for each independent run. To make the aggregation and variability analysis transparent, batch-level results can also be reported separately for all five batches.

The batch-specific supplementary release should be stored under:

```text
results/batch_specific/
├── supplementary_results.xlsx
└── tables/
    ├── deepseek/
      ├── Batch1/
         ├── medical_questions_xxx.csv
         ├── ...
      ├── Batch2/
      ├── ...
    ├── falcon3_7B/
    ├── gemma3/
    └── llama/
    └── qwen_8B/
```

The 20 batch-specific tables correspond to:

```text
5 batches × 4 metrics (F1, Accuracy, Precision, Recall) = 20 tables
```

---

## 8. Compute-enhanced single-model baselines

To examine whether multi-agent gains are explained only by greater test-time computation, additional single-model experiments were conducted on **Qwen** and **Llama** using the same benchmark and five prompt conditions.

The released compute-parity result archive contains four inference strategies:

| Config ID | Strategy | Released setting |
|---|---|---|
| `parallel_sampling` | Parallel stochastic sampling + majority vote | 32 samples per item/run; `max_new_tokens=16` per sample |
| `self_consistency_cot` | Self-consistency chain-of-thought + majority vote | 8 trajectories per item/run; `max_new_tokens=1024` |
| `extended_ttc` | Extended test-time computation | 1 trajectory; `max_new_tokens=8192` |
| `doubled_budget` | Doubled direct-answer generation budget | 1 trajectory; `max_new_tokens=6` |

All four use `temperature=1.0` and `top_p=1.0`, and each configuration is evaluated over ten independent runs.

The final compute-enhanced release should be stored under:

```text
results/compute_parity/
├── metrics_detail.csv
├── metrics_summary.csv
├── full_grid.log
└── raw/
    └── grid/
        ├── Qwen3-VL-8B-Instruct/
        │   ├── parallel_sampling/
        │   ├── self_consistency_cot/
        │   ├── extended_ttc/
        │   └── doubled_budget/
        └── Llama-3.1-8B-Instruct/
            ├── parallel_sampling/
            ├── self_consistency_cot/
            ├── extended_ttc/
            └── doubled_budget/
```

Within each terminal `model/config/version/batch/` directory:

- `calls.jsonl` stores each model call, including the prompt version, run, sample index, seed, generation settings, token counts, raw output, and parsed answer;
- `item_votes.jsonl` stores the per-item aggregate decision, including number of samples, unparseable count, and majority-vote result.

`metrics_detail.csv` contains run-level batch metrics and confusion counts. `metrics_summary.csv` contains the ten run values together with their mean and sample standard deviation. `full_grid.log` records grid completion and verification information.

These reviewer-response experiments are reported separately from the original manuscript baselines; they do not replace the original individual-model results.

---

## 9. Result-file data dictionary

### Individual-model raw CSVs

Current runners use the following core fields:

| Field | Meaning |
|---|---|
| `model_tag` | Model/configuration label |
| `batch` | `Batch1`–`Batch5` |
| `version` | `v1`–`v5` |
| `file` | Source benchmark file |
| `run_id` | Independent run index |
| `question_id` | Outcome/question index within the batch |
| `prompt` | Complete benchmark prompt |
| `response` | Complete raw model response |
| `extracted_answer` | Parsed binary answer when available |

### Rolling-review CSVs

Core fields include `panel_name`, `batch`, `version`, `run_id`, `question_id`, `source_file`, `original_prompt`, five `(agent_id, agent_response)` pairs, `final_agent_id`, and `final_response`.

### Arbitration CSVs

Core fields include `batch`, `version`, `run_id`, `question_id`, `source_file`, `original_prompt`, five independent `(agent_id, agent_response)` pairs, judge metadata/response, judge token usage, and `final_response`.

### Canonical evaluator outputs

- `item_predictions.csv`: item-level gold label, parsed prediction, parse method, and correctness;
- `parse_failures.csv`: responses for which no valid binary prediction could be extracted;
- `per_batch_run_metrics.csv`: metrics for each model × batch × prompt × run;
- `per_run_metrics.csv`: full-benchmark metrics for each model × prompt × run;
- `summary_metrics.csv`: mean and sample SD across independent runs.

---

## 10. Repository layout

```text
NCO-benchmark/
├── Batch/
│   ├── questions/                  # CANONICAL benchmark
│   ├── scripts/                    # Individual-model runners/utilities
│   ├── get_accuracy.py             # CANONICAL evaluator
│   └── build_supplementary_results.py
│
├── PEG/
│   ├── panels/                     # Rolling-review panel definitions
│   ├── peg_core.py                 # CANONICAL rolling-review logic
│   ├── model_loader.py             # Local/API model adapters
│   ├── main_multi_model.py         # Rolling-review runner
│   ├── arbitration_core.py         # CANONICAL arbitration logic
│   └── main_arbitration.py         # Arbitration runner
│
├── configs/
│   └── paper.yaml                  # Paper configuration summary
│
├── results/
│   ├── batch_specific/             # Batch-level supplementary artifacts
│   └── compute_parity/             # Compute-enhanced single-model artifacts
│
├── run_all_models.py               # Nine-model individual launcher
├── LICENSE
└── README.md
```

---

## 11. Historical/development code

The repository contains older experimental and development artifacts, including directories such as:

```text
Questions/
PEG/Questions/
PEG/Judge/
llamacpp/
ability-test/
```

and older duplicate top-level implementation files.

These are retained only for development history/provenance. They are **not** the canonical implementation used to reproduce the final paper configuration.

In particular:

- use `Batch/questions/` rather than older question copies;
- use `PEG/peg_core.py` + `PEG/main_multi_model.py` for rolling review;
- use `PEG/arbitration_core.py` + `PEG/main_arbitration.py` for arbitration; and
- use `Batch/get_accuracy.py` for final evaluation.

The legacy local-judge code under `PEG/Judge/` does not define the final arbitration architecture reported in the manuscript.

---

## 12. Reproducibility checklist

For paper reproduction, use:

1. `Batch/questions/` as the benchmark source;
2. the exact model identifiers listed above;
3. `temperature=1.0`, `top_p=1.0`, and the experiment-specific token budget;
4. ten explicit independent runs;
5. complete raw responses with explicit `run_id` and `question_id` metadata;
6. the canonical rolling-review/arbitration implementations under `PEG/`; and
7. `Batch/get_accuracy.py` for the final manuscript metrics.

Historical API availability can change. If an exact historical provider model is no longer available, results from a redirected or replacement model should be labeled as a new reproduction rather than as the original paper output.

---

## 13. Software environment

The local Hugging Face pipeline depends primarily on PyTorch and Transformers; result processing uses pandas. API experiments additionally require the corresponding provider SDKs (`anthropic`, `google-genai`, and `openai`).

The project has been exercised with a CUDA-enabled PyTorch environment and Transformers-based Hugging Face model loading. Exact package compatibility can depend on the selected checkpoints and GPU/CUDA installation.

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Contact

Repository maintained by [@mengyuqiao](https://github.com/mengyuqiao/NCO-benchmark).
