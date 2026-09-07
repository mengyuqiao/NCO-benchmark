#!/usr/bin/env python3
"""
Compute-matched single-model baseline runner (B1-B4), matching the released compute-parity experiment.

Data source, per user decision on this branch (2026-08-31): --task selects a
data FAMILY, it does NOT filter by NCO vs PCO. For --task GLP, --batch BatchN
selects Batch/questions/BatchN/medical_questions_v{version}.txt, which mixes
NCO and PCO items in one file (an organ-system batch), with per-item gold
label read from the aligned medical_answers_v{version}.txt (line i -> the
i-th "Question" block). Every item in the batch is run -- nothing is
filtered by gold -- and each logged call records its own "gold" field so
downstream metrics scripts can combine NCO+PCO within the batch (CLAUDE.md
constraint #4: NCO = positive class, computed on the combined set).

Confirmed organ-system mapping (matched against Supplementary Table 1 by
NCO+PCO item counts, then disambiguated by outcome content where counts
tied -- see conversation on this branch, 2026-08-31):
    Batch1 = GI      (7 NCO + 5 PCO)
    Batch2 = Psych   (6 NCO + 5 PCO)
    Batch3 = MSK      (7 NCO + 3 PCO)
    Batch4 = Ophtho  (6 NCO + 5 PCO)
    Batch5 = CV      (4 NCO + 6 PCO)

--task COVID is NOT implemented. There is no organ-system-batched COVID
dataset in this repo (only flat, single-class files under
ability-test/huggingface/vaccine/Question/{NCO,positive}/) -- the same "combine
NCO+PCO in one batch" structure used for GLP does not exist for COVID. Rather
than invent a batching scheme, this raises NotImplementedError. Ask the user
before deciding how COVID should be split into scoring batches.

B1/B4 prompt stripping (decided on this branch, 2026-08-31): upstream's
Batch/questions/ items (commits 06338fa, then cf55426) now mandate
reasoning-before-answer in both [Instruction] and [Your Task], for every
prompt version. B1 (parallel_sampling) and B4 (doubled_budget) send their
own "answer only" system message (DIRECT_SYSTEM_PROMPT below); leaving the
item's own reasoning mandate in the user content would make the logged
prompt contradict the logged system message and not match what the model
actually does. strip_reasoning_requirement() below removes it
programmatically (drops [Your Task], swaps any reasoning-mentioning
[Instruction] bullet for a direct-answer one) rather than hand-maintaining
a separate prompt copy, so it keeps working if upstream edits the wording
again. Applies to both B1 and B4 since they share the same direct-answer
system prompt and code path (DIRECT_ANSWER_CONFIGS).

Four configs (CLAUDE.md "Current task" table), token budget fixed at 8192
generated tokens/item unless noted:
    parallel_sampling     direct yes/no, max_new_tokens=16,  k=32 (see direct_k
                           note in resolve_config -- NOT token_budget//16)
    self_consistency_cot  reasoning+answer, max_new_tokens=1024, k=8192//1024=8
    extended_ttc          single trajectory, max_new_tokens=8192, k=1
    doubled_budget        existing single-model config (Batch/scripts/run_qwen.py:
                           system-prompt yes/no, MAX_NEW_TOKENS default 3) with
                           max_new_tokens doubled to 6, k=1 -- NOT subject to the
                           8192 budget (control showing budget alone isn't enough)

Extraction reuses peg_core._extract_yes_no verbatim (imported, not
reimplemented) so behavior never drifts from the multi-agent pipeline's
parser (yes/no/unparseable three-way, never defaults to "no" -- the historical evaluation protocol).
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_FILE.parent))

from PEG.question_loader import load_question_blocks  # noqa: E402
from compute_parity_logging import CallLogger  # noqa: E402

# Historical parser used by the coordinator-provided compute-parity runs.
# Kept here explicitly so the released runner matches the raw outputs and
# metrics associated with those runs.
_ANSWER_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*[:\-]?\s*\**\s*[\"']?\b(yes|no)\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def _extract_yes_no(text: str) -> str:
    marker_matches = list(_ANSWER_MARKER_RE.finditer(text))
    if marker_matches:
        return marker_matches[-1].group(1).lower()
    yn_matches = list(_YES_NO_RE.finditer(text))
    if yn_matches:
        return yn_matches[-1].group(1).lower()
    return "unparseable"

DEFAULT_TOKEN_BUDGET = 8192
DEFAULT_DIRECT_TOKENS = 16
DEFAULT_DIRECT_K = 32
DEFAULT_COT_TOKENS = 1024
DEFAULT_EXTENDED_TOKENS = 8192
DEFAULT_BASE_TOKENS = 3
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

ORGAN_SYSTEM_BY_BATCH = {
    "Batch1": "GI",
    "Batch2": "Psych",
    "Batch3": "MSK",
    "Batch4": "Ophtho",
    "Batch5": "CV",
}

# System-prompt wrapping per config. B1/B4 reuse the exact instruction string
# from Batch/scripts/run_qwen.py's to_chat_text() for fidelity to "the
# existing single-model config". B2/B3 ask for reasoning then a parseable
# final line; peg_core._extract_yes_no takes the FIRST standalone yes/no in
# the text, so a CoT response that says e.g. "there is no prior evidence, but
# ... yes" could misparse -- this is an existing, known limitation of the
# shared extractor, not something this script papers over (CLAUDE.md
# instruction: stay consistent with peg_core, don't reimplement it).
DIRECT_SYSTEM_PROMPT = "Reply with only one token: yes or no. Do not explain your reasoning."
COT_SYSTEM_PROMPT = (
    "Think through the causal reasoning step by step. After your reasoning, "
    "end your response with a new line in exactly this format: "
    "Final answer: yes  OR  Final answer: no."
)
EXTENDED_COT_SYSTEM_PROMPT = (
    "Think through the causal reasoning as thoroughly as you need to. After "
    "your reasoning, end your response with a new line in exactly this "
    "format: Final answer: yes  OR  Final answer: no."
)

CONFIG_SYSTEM_PROMPTS = {
    "parallel_sampling": DIRECT_SYSTEM_PROMPT,
    "self_consistency_cot": COT_SYSTEM_PROMPT,
    "extended_ttc": EXTENDED_COT_SYSTEM_PROMPT,
    "doubled_budget": DIRECT_SYSTEM_PROMPT,
}

# Configs that keep the item's own trailing "Answer: [Your answer, ...]"
# placeholder line and append a bare "Answer:" cue (matches run_qwen.py's
# to_chat_text() exactly) vs. configs that strip that placeholder line so the
# model has room to reason before answering.
DIRECT_ANSWER_CONFIGS = {"parallel_sampling", "doubled_budget"}


def strip_answer_placeholder(qtext: str) -> str:
    lines = qtext.rstrip("\n").split("\n")
    if lines and lines[-1].strip().lower().startswith("answer:"):
        lines = lines[:-1]
    return "\n".join(lines).rstrip()


_YOUR_TASK_SECTION_RE = re.compile(r"\[Your Task\].*?(?=\[Question\]|\Z)", re.IGNORECASE | re.DOTALL)
_INSTRUCTION_SECTION_RE = re.compile(
    r"\[Instruction\].*?(?=\[Few-shot Examples\]|\[Your Task\]|\[Question\]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_BULLET_RE = re.compile(r"^\s*-.*\breason\w*\b.*$", re.IGNORECASE | re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def strip_reasoning_requirement(qtext: str) -> str:
    """For direct-answer configs (B1 parallel_sampling, B4 doubled_budget):
    upstream's item text now mandates reasoning-before-answer in both
    [Instruction] and [Your Task] (commits 06338fa, then cf55426). These
    configs send their own system message telling the model to answer only
    -- if the user content still told it to reason first, the logged prompt
    would contradict the logged system message and wouldn't match what the
    model actually does. Strip the requirement programmatically (not a
    hand-maintained prompt copy) so it stays correct if upstream edits the
    wording again: drop the whole [Your Task] section, and swap any
    [Instruction] bullet that mentions reasoning for a direct-answer one.
    """
    qtext = _YOUR_TASK_SECTION_RE.sub("", qtext)

    def _fix_instruction(m):
        section = m.group(0)
        section = _REASONING_BULLET_RE.sub('- Reply with only "yes" or "no."', section)
        return section

    qtext = _INSTRUCTION_SECTION_RE.sub(_fix_instruction, qtext)
    return _BLANK_RUN_RE.sub("\n\n", qtext).strip()


def build_user_content(qtext: str, config: str) -> str:
    if config in DIRECT_ANSWER_CONFIGS:
        qtext = strip_reasoning_requirement(qtext)
        return qtext.strip() + "\nAnswer:"
    return strip_answer_placeholder(qtext)


def resolve_config(config: str, token_budget: int, direct_tokens: int,
                    cot_tokens: int, extended_tokens: int, base_tokens: int,
                    direct_k: int):
    """Returns (max_new_tokens, k) for one config, per the experiment configuration.

    parallel_sampling's k is NOT derived from token_budget (unlike
    self_consistency_cot). Temperature sweep on this branch (2026-08-31,
    GLP/Batch3/v2/qid=1, temperatures 0.7-2.0, 32+64 samples/temp) found
    zero sampling diversity for this config -- every sample across every
    temperature came back byte-identical. token_budget//direct_tokens=512
    would just repeat that single draw 512 times for no benefit, silently
    eating GPU time while never actually spending the notional 8192-token
    budget (16 tokens/sample x 512 = 8192 max_new_tokens cap, but real
    usage was ~2 tokens/sample -> ~1024 total, far under budget regardless
    of k). direct_k defaults to 32: enough to demonstrate the null-diversity
    finding is stable, not enough to burn compute on a provably flat
    distribution. Report the gap between direct_k's actual token usage and
    the 8192 target explicitly -- it's a result (this config structurally
    can't use the budget), not something to paper over by inflating k.
    """
    if config == "parallel_sampling":
        return direct_tokens, direct_k
    if config == "self_consistency_cot":
        return cot_tokens, token_budget // cot_tokens
    if config == "extended_ttc":
        return extended_tokens, 1
    if config == "doubled_budget":
        return base_tokens * 2, 1
    raise ValueError(f"unknown config: {config}")


def majority_vote(extracted_list):
    """unparseable never votes; 0 valid votes or a tie -> 'unparseable'
    (CLAUDE.md constraint #2: never silently default to a class)."""
    votes = [e for e in extracted_list if e in ("yes", "no")]
    if not votes:
        return "unparseable"
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "unparseable"
    return counts[0][0]


def load_glp_batch(batch: str, version: str):
    root = REPO_ROOT / "Batch" / "questions" / batch
    qfile = root / f"medical_questions_{version}.txt"
    afile = root / f"medical_answers_{version}.txt"
    if not qfile.exists():
        raise FileNotFoundError(qfile)
    if not afile.exists():
        raise FileNotFoundError(afile)

    blocks = load_question_blocks(qfile)
    gold_lines = [
        line.strip().lower()
        for line in afile.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        if line.strip()
    ]
    if len(gold_lines) != len(blocks):
        raise ValueError(
            f"{qfile.name} has {len(blocks)} questions but {afile.name} has "
            f"{len(gold_lines)} answer lines -- cannot align gold labels"
        )
    for g in gold_lines:
        if g not in ("yes", "no"):
            raise ValueError(f"{afile}: non yes/no gold line {g!r}")

    items = []
    for (qid_str, text), gold in zip(blocks, gold_lines):
        items.append({"qid": int(qid_str), "gold": gold, "text": text})
    return items


def load_items(task: str, batch: str, version: str):
    if task == "GLP":
        if batch not in ORGAN_SYSTEM_BY_BATCH:
            raise ValueError(f"--batch must be one of {sorted(ORGAN_SYSTEM_BY_BATCH)}, got {batch!r}")
        return load_glp_batch(batch, version)
    if task == "COVID":
        raise NotImplementedError(
            "COVID has no organ-system-batched dataset in this repo (only flat "
            "NCO-only / positive-only files under "
            "ability-test/huggingface/vaccine/Question/). Ask the user how COVID "
            "items should be grouped into scoring batches before implementing "
            "this -- do not guess a mapping."
        )
    raise ValueError(f"unknown --task {task!r}")


def load_model_and_tokenizer(model_name: str, device):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch = (getattr(config, "architectures", None) or [""])[0]
    # Qwen3-VL is registered under AutoModelForImageTextToText, not
    # AutoModelForCausalLM, even when used text-only (verified against this
    # environment's transformers==5.16.1 model mapping tables). Llama-3.1 is
    # a plain CausalLM. Pick the class from the checkpoint's own config
    # rather than hard-coding per model name.
    model_cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    print(f"[BOOT] {model_name}: architecture={arch!r} -> {model_cls.__name__}", flush=True)

    model = model_cls.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto").to(device)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, device, system_prompt, user_content,
                  max_new_tokens, seed, temperature, top_p):
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # Some checkpoints (e.g. Qwen3-VL, whose "tokenizer" is really a
    # processor) return a BatchEncoding/dict from apply_chat_template even
    # for text-only input, instead of a bare input_ids tensor.
    if isinstance(templated, torch.Tensor):
        input_ids = templated.to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = templated["input_ids"].to(device)
        attention_mask = templated.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids)
    prompt_tokens = input_ids.shape[1]

    # transformers' generate() (this env: 5.16.1) does not accept a
    # `generator=` kwarg (peg_core.py's usage of it goes through a
    # `pipeline()` call, not a direct generate() call -- not applicable
    # here). Reseed the global RNG per call instead: each sample gets its
    # own seed logged, and a full reseed before each call makes the draws
    # independent of call order, which is what the experiment's reproducibility requirement
    # actually requires.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = out_ids[0, prompt_tokens:]
    completion_tokens = int(new_ids.shape[0])
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw_output, int(prompt_tokens), completion_tokens


def load_vllm_engine(model_name: str, max_model_len: int, gpu_memory_utilization: float = 0.85):
    """vLLM feasibility verified on this branch (2026-08-31): vllm==0.28.0
    pins torch==2.13.0, exactly matching this env's installed torch (no
    downgrade); both Qwen3VLForConditionalGeneration and LlamaForCausalLM
    are natively registered in vLLM's model registry at this version;
    load+generate smoke-tested for both models before this function was
    written to depend on it.
    """
    from vllm import LLM

    llm = LLM(model=model_name, trust_remote_code=True,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def generate_batch_vllm(llm, tokenizer, system_prompt, user_contents, max_new_tokens, seeds, temperature, top_p):
    """ALL pending requests for a run -- across every item, every k sample --
    submitted as ONE llm.generate() call, not a loop. This is deliberate,
    not just per-item batching: a k=1 config (extended_ttc, doubled_budget)
    has nothing to batch within one item, so batching only per-item gives
    vLLM zero cross-request parallelism to exploit and the fixed engine
    startup cost dominates (measured on this branch, 2026-08-31: B3 via
    vLLM was slightly SLOWER wall-clock than transformers when only
    batched per-item, because each item's single sample was submitted
    alone). Batching every pending request in the whole run together fixes
    this for k=1 configs too, and is strictly better for k>1 configs.

    Each request still gets its own individually-seeded SamplingParams
    (not vLLM's `n=k`, which would share one seed across a group) so the
    per-sample `seed` field in the JSONL log means exactly what it did
    under the transformers path (the experiment's reproducibility requirement). Token counts
    come straight from vLLM's own output token ids -- no retokenization,
    no heuristic (constraint #3).

    user_contents and seeds must be the same length (one prompt-building
    input, one seed, per request); a repeated item's user_content is fine.
    Returns a list of (raw_output, prompt_tokens, completion_tokens),
    aligned with the input lists.
    """
    from vllm import SamplingParams

    prompts = []
    for user_content in user_contents:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    params_list = [
        SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
        for seed in seeds
    ]
    outputs = llm.generate(prompts, params_list, use_tqdm=False)

    results = []
    for out in outputs:
        prompt_tokens = len(out.prompt_token_ids)
        completion = out.outputs[0]
        completion_tokens = len(completion.token_ids)
        raw_output = completion.text
        results.append((raw_output, prompt_tokens, completion_tokens))
    return results


def process_job(*, task, batch, prompt_version, config, model_name, engine_kind, engine, tokenizer, device,
                 logger, runs, seed_base, temperature, top_p,
                 token_budget, direct_tokens, direct_k, cot_tokens, extended_tokens, base_tokens, limit=None):
    """Process one (task,batch,prompt_version,config,model) job against an
    ALREADY-LOADED engine: load items, resolve the config's (max_new_tokens,
    k), build every pending (qid,run,sample) request across the WHOLE job,
    submit it (batched for vLLM, looped for transformers), log each call,
    then print/write the per-item majority vote. Resumable via `logger`'s
    already-loaded done-cache (CallLogger reads its run_dir's existing
    calls.jsonl on construction).

    This is the one place the generation-and-logging loop lives -- shared
    by run_single_model.py's CLI (one job per invocation, engine loaded
    fresh each time) and grid_worker.py (many jobs per invocation, one
    engine loaded once and reused across all of them). Don't reimplement
    this loop a second time; import and call this instead.

    Returns (n_calls_made, total_completion_tokens, total_prompt_tokens).
    """
    import json
    from datetime import datetime, timezone

    items = load_items(task, batch, prompt_version)
    if limit is not None:
        items = items[:limit]
    print(f"[DATA] task={task} batch={batch} ({ORGAN_SYSTEM_BY_BATCH.get(batch)}) "
          f"version={prompt_version} -> {len(items)} items", flush=True)

    max_new_tokens, k = resolve_config(config, token_budget, direct_tokens, cot_tokens,
                                        extended_tokens, base_tokens, direct_k)
    print(f"[CFG] config={config} max_new_tokens={max_new_tokens} k={k} "
          f"runs={runs} temperature={temperature} top_p={top_p}", flush=True)
    print(f"[RUN_DIR] {logger.run_dir}", flush=True)

    system_prompt = CONFIG_SYSTEM_PROMPTS[config]

    existing = {}
    for rec in logger.read_all():
        existing[(rec["qid"], rec["run"], rec["sample"])] = rec["extracted"]

    total_completion_tokens = 0
    total_prompt_tokens = 0
    n_calls_made = 0
    votes_summary_path = logger.run_dir / "item_votes.jsonl"

    # sample_extracted[(qid, run)] -> list of extracted values, index = sample-1
    sample_extracted = {(item["qid"], run): [None] * k for item in items for run in range(1, runs + 1)}
    user_content_by_qid = {item["qid"]: build_user_content(item["text"], config) for item in items}
    gold_by_qid = {item["qid"]: item["gold"] for item in items}

    # Build ONE flat list of every missing (qid, run, sample) across the
    # WHOLE job. For vLLM this is submitted as a single generate() call --
    # batching only within one item leaves k=1 configs with nothing to
    # batch (see generate_batch_vllm docstring). For transformers this is
    # still processed one call at a time.
    pending = []  # each: (qid, run, sample, seed)
    for item in items:
        qid = item["qid"]
        for run in range(1, runs + 1):
            for sample in range(1, k + 1):
                key = (qid, run, sample)
                if key in existing:
                    sample_extracted[(qid, run)][sample - 1] = existing[key]
                else:
                    seed = seed_base + run * 10000 + sample
                    pending.append((qid, run, sample, seed))

    if pending:
        if engine_kind == "vllm":
            batch_results = generate_batch_vllm(
                engine, tokenizer, system_prompt,
                [user_content_by_qid[qid] for qid, _, _, _ in pending],
                max_new_tokens, [seed for _, _, _, seed in pending],
                temperature, top_p,
            )
        else:
            batch_results = [
                generate_one(engine, tokenizer, device, system_prompt, user_content_by_qid[qid],
                             max_new_tokens, seed, temperature, top_p)
                for qid, _, _, seed in pending
            ]

        for (qid, run, sample, seed), (raw_output, prompt_tokens, completion_tokens) in zip(pending, batch_results):
            extracted = _extract_yes_no(raw_output)
            logger.log_call(
                task=task, batch=batch, qid=qid, gold=gold_by_qid[qid], config=config,
                model=model_name, prompt_version=prompt_version,
                run=run, sample=sample, seed=seed,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                raw_output=raw_output, extracted=extracted,
            )
            existing[(qid, run, sample)] = extracted
            sample_extracted[(qid, run)][sample - 1] = extracted
            total_completion_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens
            n_calls_made += 1

            if n_calls_made % 20 == 0 or n_calls_made == len(pending):
                print(f"[PROGRESS] calls_made={n_calls_made}/{len(pending)} "
                      f"completion_tokens_running_total={total_completion_tokens}", flush=True)

    for item in items:
        qid, gold = item["qid"], item["gold"]
        for run in range(1, runs + 1):
            extracted_list = sample_extracted[(qid, run)]
            vote = majority_vote(extracted_list)
            n_unparseable = sum(1 for e in extracted_list if e == "unparseable")
            print(f"[VOTE] qid={qid} gold={gold} run={run} n_samples={len(extracted_list)} "
                  f"unparseable={n_unparseable} majority_vote={vote} "
                  f"{'CORRECT' if vote == gold else 'WRONG' if vote in ('yes','no') else 'UNSCORED'}",
                  flush=True)

            with votes_summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task": task, "batch": batch, "qid": qid, "gold": gold,
                    "config": config, "model": model_name, "prompt_version": prompt_version,
                    "run": run, "n_samples": len(extracted_list), "n_unparseable": n_unparseable,
                    "majority_vote": vote,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    print(f"\n[SUMMARY] new calls this invocation={n_calls_made} "
          f"total_completion_tokens={total_completion_tokens} total_prompt_tokens={total_prompt_tokens}", flush=True)
    print(f"[SUMMARY] logs: {logger.log_path}", flush=True)
    print(f"[SUMMARY] item votes: {votes_summary_path}", flush=True)

    return n_calls_made, total_completion_tokens, total_prompt_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", choices=["Qwen/Qwen3-VL-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"])
    ap.add_argument("--task", required=True, choices=["GLP", "COVID"])
    ap.add_argument("--batch", required=True, help="e.g. Batch1..Batch5 (GLP organ-system batch)")
    ap.add_argument("--prompt-version", required=True, choices=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--config", required=True,
                     choices=["parallel_sampling", "self_consistency_cot", "extended_ttc", "doubled_budget"])
    ap.add_argument("--runs", type=int, default=10, help="independent repetitions per item")
    ap.add_argument("--seed-base", type=int, default=12345)
    ap.add_argument("--gpu", type=int, required=True, help="CUDA device index")
    ap.add_argument("--run-dir", default=None,
                     help="existing run directory to resume; omit to start a new timestamped one")
    ap.add_argument("--limit", type=int, default=None, help="cap number of items processed (smoke tests)")

    ap.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    ap.add_argument("--direct-tokens", type=int, default=DEFAULT_DIRECT_TOKENS,
                     help="max_new_tokens for parallel_sampling")
    ap.add_argument("--direct-k", type=int, default=DEFAULT_DIRECT_K,
                     help="parallel_sampling's k, NOT derived from token_budget -- see resolve_config docstring "
                          "(temperature sweep found zero sampling diversity, so token_budget//direct_tokens=512 "
                          "would just repeat one draw 512x)")
    ap.add_argument("--cot-tokens", type=int, default=DEFAULT_COT_TOKENS,
                     help="max_new_tokens for self_consistency_cot. 1024, not 512: a natural-length "
                          "probe (GLP/Batch3/v2/qid=1, max_new_tokens=4096, 4 samples) found real "
                          "completion lengths of 471-939 tokens (mean 723) when not cut off. At 512, "
                          "87.5%% of samples hit the cap and 25%% were truncated before ever reaching a "
                          "conclusion (unparseable) -- a systematic bias, not model confusion, and one "
                          "that gets worse on harder items. 1024 covers the observed max with margin, "
                          "drops unparseable to 0%%, and still uses ~74%% of the 8192 budget (k=8 x "
                          "~723 actual tokens/sample). 2048 (k=4) only uses ~35%% of the budget -- too "
                          "wasteful to call compute-matched.")
    ap.add_argument("--extended-tokens", type=int, default=DEFAULT_EXTENDED_TOKENS,
                     help="max_new_tokens for extended_ttc")
    ap.add_argument("--base-tokens", type=int, default=DEFAULT_BASE_TOKENS,
                     help="existing single-model max_new_tokens (Batch/scripts/run_qwen.py default); "
                          "doubled_budget uses 2x this")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--engine", choices=["transformers", "vllm"], default="transformers",
                     help="vllm submits one item's k samples as a single batched call instead of "
                          "looping generate_one() k times -- feasibility verified 2026-08-31 "
                          "(vllm==0.28.0, torch pin matches installed torch exactly, both models "
                          "load and generate). transformers stays the default so existing behavior "
                          "doesn't change under anyone's feet.")
    ap.add_argument("--vllm-max-model-len", type=int, default=16384,
                     help="must cover prompt_tokens + max_new_tokens for whichever config is largest "
                          "(extended_ttc needs the most headroom)")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    device = torch.device("cuda:0")

    logger = CallLogger(config=args.config, base_dir=str(REPO_ROOT / "results" / "compute_parity" / "raw"), run_dir=args.run_dir)

    if args.engine == "vllm":
        engine, tokenizer = load_vllm_engine(args.model, args.vllm_max_model_len, args.vllm_gpu_memory_utilization)
    else:
        engine, tokenizer = load_model_and_tokenizer(args.model, device)

    process_job(
        task=args.task, batch=args.batch, prompt_version=args.prompt_version, config=args.config,
        model_name=args.model, engine_kind=args.engine, engine=engine, tokenizer=tokenizer, device=device,
        logger=logger, runs=args.runs, seed_base=args.seed_base,
        temperature=args.temperature, top_p=args.top_p,
        token_budget=args.token_budget, direct_tokens=args.direct_tokens, direct_k=args.direct_k,
        cot_tokens=args.cot_tokens, extended_tokens=args.extended_tokens, base_tokens=args.base_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
