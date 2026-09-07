# NCO Benchmark

This repository contains the benchmark, inference pipelines, multi-agent implementations, evaluation code, and released result artifacts for studying whether large language models can distinguish **negative control outcomes (NCOs)** from **positive control outcomes (PCOs)** in a causal-reasoning task comparing GLP-1 receptor agonists (GLP-1RA) with SGLT2 inhibitors (SGLT2i).

The study evaluates:

- nine individual LLMs;
- four five-agent rolling-review panels;
- a five-agent arbitration architecture with a Claude judge; and
- additional compute-enhanced single-model baselines examining whether the observed multi-agent gains can be explained by test-time compute alone.

The canonical benchmark and paper-aligned implementations are identified explicitly below. Historical and development artifacts are retained separately under `legacy/` and should not be used to reproduce the final paper configuration.

---

## 1. Canonical benchmark

The canonical benchmark is:

```text
Batch/questions/
├── Batch1/
├── Batch2/
├── Batch3/
├── Batch4/
└── Batch5/
```

The benchmark contains **54 outcomes** divided into five clinical groups:

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

Each batch contains aligned `medical_questions_v*.txt` and `medical_answers_v*.txt` files.

### Data provenance

The clinical cohort description and aggregate outcome-incidence estimates used in the benchmark prompts were provided directly by Penn Medicine. The underlying patient-level EHR data and institution-specific data-extraction procedures are not included in this repository.

### Terminology

The historical experimental prompts use the term `OOI` ("Positive Outcome") for the positive-control class. The manuscript and supplementary material use `PCO` ("positive control outcome"). The original prompt wording is retained for reproducibility; `OOI` and `PCO` refer to the same class in this benchmark.

Under the evaluation convention used in the study:

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

`grok-4-0709` is the historical model identifier used in the study. Because provider-side model availability and routing can change, a redirected or replacement endpoint should be treated as a new reproduction rather than as the exact historical Grok run.

API credentials are supplied through environment variables and are never stored in the repository.

---

## 3. Main inference configuration

The paper-aligned individual-model and multi-agent experiments use:

```text
temperature      = 1.0
top_p            = 1.0
max_new_tokens   = 2048
independent runs = 10
```

An independent run corresponds to a separate model generation or complete multi-agent trajectory. Multiple samples returned within a single inference strategy are not reinterpreted as independent paper runs.


---

## 4. Individual-model experiments

The top-level launcher for the nine individual models is:

```bash
python run_all_models.py
```

Examples:

```bash
# Inspect commands without running inference
python run_all_models.py --dry-run

# Run local Hugging Face models only
python run_all_models.py --local-only

# Run API models only
python run_all_models.py --api-only

# Select specific models
python run_all_models.py --models llama qwen claude
```

Model-specific runners are located under:

```text
Batch/scripts/
```

including the runners for DeepSeek, Falcon3, Gemma, Llama, Qwen, Claude, Gemini, GPT-5, and Grok.

The current runners record explicit `run_id` and `question_id` fields together with the complete benchmark prompt, full model response, and parsed binary answer.

### API credentials

Set only the credentials required for the models being executed:

```bash
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export XAI_API_KEY="..."
```

Credentials must not be committed to the repository.

---

## 5. Multi-agent experiments

The canonical multi-agent implementation is under:

```text
PEG/
```

### 5.1 Rolling review

Rolling review consists of five sequential agents.

```text
Original prompt
      │
      ▼
   Agent 1
      │ complete response
      ▼
   Agent 2
      │ revised or affirmed response
      ▼
   Agent 3
      │
      ▼
   Agent 4
      │
      ▼
   Agent 5 ──> final panel response
```

Agent 1 receives the original benchmark prompt. Agents 2–5 receive the original prompt together with the complete response from the immediately preceding agent and are asked to critique, revise, or affirm that response. The fifth agent's response is used as the final panel response.

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

The four reported panels are:

- **Mixed-Panel**: Llama + Qwen + Gemma + Falcon3 + DeepSeek
- **Claude-Panel**: five Claude agents
- **Gemini-Panel**: five Gemini agents
- **Qwen-Panel**: five logical Qwen agents

Run the four panels with:

```bash
python PEG/main_multi_model.py --panel PEG/panels/mixed.json
python PEG/main_multi_model.py --panel PEG/panels/claude.json
python PEG/main_multi_model.py --panel PEG/panels/gemini.json
python PEG/main_multi_model.py --panel PEG/panels/qwen.json
```

A minimal smoke test is:

```bash
python PEG/main_multi_model.py \
  --panel PEG/panels/mixed.json \
  --batch 1 \
  --version 1 \
  --num-runs 1
```

Rolling-review outputs record the original prompt, the complete response produced by each of the five agents, explicit run/question identifiers, and the final fifth-agent response.

### 5.2 Arbitration

Arbitration uses five independent agents followed by a Claude judge.

```text
Agent 1 ─┐
Agent 2 ─┤
Agent 3 ─┼──> Claude Sonnet 4.6 judge ──> final response
Agent 4 ─┤
Agent 5 ─┘
```

The five agents independently receive the same original benchmark prompt and do not observe one another's responses. The judge receives the original prompt and all five complete agent responses simultaneously.

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

The arbitration judge is fixed to:

```text
claude-sonnet-4-6
```

Claude is accessed through the Anthropic API using `ANTHROPIC_API_KEY`; API credentials are not stored in the repository.

---

## 6. Evaluation

The canonical evaluator is:

```text
Batch/get_accuracy.py
```

It computes:

- Accuracy
- Precision
- Recall
- F1
- parse rate
- explicit parse failures

### Evaluation rules

1. NCO (`no`) is treated as the positive class.
2. Independent runs are identified using explicit `run_id` values.
3. `response_id` and row order are not used to infer independent runs.
4. Binary parsing requires an explicit yes/no decision; incidental yes/no tokens inside free-form reasoning are insufficient.
5. Unparseable outputs are retained rather than silently converted to either class.
6. Accuracy uses the complete benchmark denominator, so an unparseable response is counted as incorrect.
7. For each independent run, metrics are computed over the **complete 54-outcome benchmark**.
8. Reported means and **sample standard deviations** are then computed across the ten independent runs.

Example:

```bash
python Batch/get_accuracy.py \
  --results-root results \
  --gold-root Batch/questions \
  --expected-runs 10
```

The evaluator produces:

```text
evaluation/
├── item_predictions.csv
├── parse_failures.csv
├── per_batch_run_metrics.csv
├── per_run_metrics.csv
└── summary_metrics.csv
```

`--allow-incomplete` is intended only for debugging partial runs and should not be used for final paper evaluation.

---

## 7. Batch-specific supplementary results

The manuscript reports performance over the complete 54-outcome benchmark within each independent run and then reports the mean and sample standard deviation across ten runs.

To make the batch-level variation transparent, the released supplementary workbook is:

```text
results/batch_specific/supplementary_results.xlsx
```

It contains the batch-specific results used to report:

```text
5 batches × 4 metrics
= 20 batch-specific supplementary tables
```

The four metrics are:

```text
F1
Accuracy
Precision
Recall
```

These batch-level tables are supplementary analyses and do not change the full-benchmark aggregation procedure used for the main reported metrics.

---

## 8. Compute-enhanced single-model baselines

Additional experiments evaluate whether improvements from multi-agent collaboration can be explained solely by additional single-model test-time computation.

These experiments use two open-source models from the main study:

```text
Qwen/Qwen3-VL-8B-Instruct
meta-llama/Llama-3.1-8B-Instruct
```

under all five benchmark prompt conditions.

Four inference strategies are released:

| Config ID | Strategy | Experimental setting |
|---|---|---|
| `parallel_sampling` | Parallel stochastic sampling + majority vote | 32 samples; `max_new_tokens=16` per sample |
| `self_consistency_cot` | Self-consistency chain-of-thought + majority vote | 8 trajectories; `max_new_tokens=1024` per trajectory |
| `extended_ttc` | Extended test-time computation | 1 trajectory; `max_new_tokens=8192` |
| `doubled_budget` | Doubled direct-answer budget | 1 trajectory; baseline 3 tokens increased to 6 |

All configurations use:

```text
temperature = 1.0
top_p       = 1.0
runs        = 10
```

The experiment runner is:

```text
Batch/scripts/run_compute_parity.py
```

Example:

```bash
python Batch/scripts/run_compute_parity.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --task GLP \
  --batch Batch1 \
  --prompt-version v1 \
  --config parallel_sampling \
  --runs 10 \
  --gpu 0
```

The same runner supports:

```text
parallel_sampling
self_consistency_cot
extended_ttc
doubled_budget
```

and both Qwen and Llama.

Released compute-parity results are under:

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

- `calls.jsonl` contains individual model calls, including model/configuration information, prompt version, run, sample index, seed, generation settings, token counts, raw output, and parsed answer.
- `item_votes.jsonl` contains the aggregate item-level decision for each run, including sample counts, unparseable counts, and majority-vote result.

`metrics_detail.csv` contains run-level batch metrics and confusion counts. `metrics_summary.csv` contains the ten run values together with their aggregate means and sample standard deviations. `full_grid.log` records execution and verification information.

These experiments were added as compute-enhanced single-model baselines and are reported separately from the original individual-model experiments.

---

## 9. Result-file data dictionary

### Individual-model outputs

Current runners use the following core fields:

| Field | Meaning |
|---|---|
| `model_tag` | Model/configuration identifier |
| `batch` | `Batch1`–`Batch5` |
| `version` | `v1`–`v5` |
| `file` | Source benchmark file |
| `run_id` | Independent run index |
| `question_id` | Outcome/question identifier |
| `prompt` | Complete benchmark prompt |
| `response` | Complete raw model response |
| `extracted_answer` | Parsed binary answer, when available |

### Rolling-review outputs

Core fields include:

```text
panel_name
batch
version
run_id
question_id
source_file
original_prompt
agent IDs and complete agent responses
final_agent_id
final_response
```

### Arbitration outputs

Core fields include:

```text
batch
version
run_id
question_id
source_file
original_prompt
five independent agent responses
judge metadata
judge response
judge token usage
final_response
```

### Canonical evaluator outputs

- `item_predictions.csv`: gold label, parsed prediction, parse method, and item-level correctness
- `parse_failures.csv`: responses from which no valid binary decision could be extracted
- `per_batch_run_metrics.csv`: metrics for each model × batch × prompt × run
- `per_run_metrics.csv`: metrics over the complete benchmark for each model × prompt × run
- `summary_metrics.csv`: mean and sample standard deviation across independent runs

---

## 10. Repository layout

```text
NCO-benchmark/
├── Batch/
│   ├── questions/                  # Canonical benchmark
│   ├── scripts/                    # Individual and compute-parity runners
│   ├── get_accuracy.py             # Canonical evaluator
│   └── build_supplementary_results.py
│
├── PEG/
│   ├── panels/                     # Rolling-review panel definitions
│   ├── peg_core.py                 # Canonical rolling-review logic
│   ├── model_loader.py             # Local/API model adapters
│   ├── main_multi_model.py         # Rolling-review runner
│   ├── arbitration_core.py         # Canonical arbitration logic
│   └── main_arbitration.py         # Arbitration runner
│
├── configs/
│   └── paper.yaml                  # Paper configuration summary
│
├── results/
│   ├── batch_specific/             # Batch-level supplementary results
│   └── compute_parity/             # Compute-enhanced baselines and raw outputs
│
├── legacy/                         # Historical/development artifacts
│
├── run_all_models.py               # Nine-model individual launcher
├── LICENSE
└── README.md
```

---

## 11. Historical and development artifacts

Older experimental implementations, duplicate benchmark copies, development evaluators, local-judge prototypes, plotting utilities, and other historical artifacts are retained under:

```text
legacy/
```

They are preserved for provenance only and are **not** the canonical paper implementation.

For reproduction of the paper-aligned configuration, use:

```text
Benchmark:
Batch/questions/

Individual-model runners:
Batch/scripts/

Rolling review:
PEG/peg_core.py
PEG/main_multi_model.py
PEG/panels/

Arbitration:
PEG/arbitration_core.py
PEG/main_arbitration.py

Evaluation:
Batch/get_accuracy.py
```

In particular, historical local-judge implementations under `legacy/` do not define the final five-agent-plus-Claude arbitration architecture reported in the manuscript.

---

## 12. Reproducibility checklist

For reproduction of the paper-aligned experiments:

1. Use `Batch/questions/` as the benchmark source.
2. Use the exact model identifiers listed in this README.
3. Use the inference settings associated with the experiment being reproduced.
4. Use ten explicitly identified independent runs for the reported main metrics.
5. Preserve explicit `run_id` and `question_id` metadata.
6. Preserve complete raw model responses rather than only parsed binary decisions.
7. Use the canonical rolling-review and arbitration implementations under `PEG/`.
8. Use `Batch/get_accuracy.py` for paper-aligned evaluation.

Historical provider models may become unavailable. A redirected or replacement API model should be labeled as a new reproduction and should not be represented as an exact historical run.

---

## 13. Software environment

The local inference pipeline primarily depends on:

```text
Python
PyTorch
Transformers
pandas
```

API experiments additionally require the corresponding provider SDKs, including Anthropic, Gemini, OpenAI, and xAI-compatible clients as applicable.

The compute-parity runner supports both Transformers and vLLM execution paths. Exact GPU, CUDA, and package compatibility can depend on the selected checkpoint.

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Contact

Repository maintained by [@mengyuqiao](https://github.com/mengyuqiao/NCO-benchmark).
