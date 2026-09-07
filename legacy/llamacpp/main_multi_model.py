# main_multi_model.py
import os
import csv
from pathlib import Path

from question_loader import load_question_blocks
from model_loader import load_model_pipelines
from peg_core import (
    initial_plans_parallel_all,
    relay_ring_round,
)
from log_utils import get_log_path, get_csv_path, write_csv_header, append_csv_row

# ===== 配置（支持环境变量）=====
QUESTION_PATHS = os.getenv(
    "QUESTION_PATHS",
    "Questions/NCO/nco_v1_questions.txt"
).split(",")

# 现在虽然变量名还叫 HF_MODELS，但本质已经是 llama.cpp / GGUF repo
HF_MODELS = os.getenv("HF_MODELS", "").split(",") if os.getenv("HF_MODELS") else [
    "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF::g1",
    "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF::g2",
    "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF::g3",
]

# 仍然保留设备映射，用来告诉每个 agent 用哪个 GPU
HF_DEVICES = os.getenv("HF_DEVICES", "").split(",") if os.getenv("HF_DEVICES") else ["0", "1", "2"]

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
RESUME = os.getenv("RESUME", "1") != "0"
N_SAMPLES = int(os.getenv("N_SAMPLES", "10"))
RUN_TAG = os.getenv("RUN_TAG", "default")

# llama.cpp 配置
GGUF_FILENAME = os.getenv("GGUF_FILENAME", "*.gguf")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "60"))
VERBOSE = os.getenv("LLAMA_VERBOSE", "0") == "1"

def model_repo(model_id: str) -> str:
    return model_id.split("::", 1)[0].strip()

def model_tag(model_id: str) -> str:
    parts = model_id.split("::", 1)
    return parts[1].strip() if len(parts) == 2 else ""

assert len(HF_MODELS) == len(HF_DEVICES), "HF_MODELS 与 HF_DEVICES 数量需一致"

MODEL_DEVICE_MAP = {m: int(d) for m, d in zip(HF_MODELS, HF_DEVICES)}

def load_completed_qids_from_csv(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip() for c in (reader.fieldnames or [])]
            if "qid" not in fieldnames:
                return done

            for row in reader:
                try:
                    qid_val = int(str(row.get("qid", "")).strip())
                    if qid_val > 0:
                        done.add(qid_val)
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARN] resume failed to read CSV ({csv_path.name}): {e}", flush=True)

    return done

def build_header(models, num_rounds, n_samples):
    header = ["file", "qid", "question"]
    for model in models:
        for r in range(1, num_rounds + 1):
            for k in range(1, n_samples + 1):
                header.append(f"{model}_r{r}_run{k}")
    return header

def main():
    print(f"[BOOT] Loading models: {HF_MODELS} with device map {MODEL_DEVICE_MAP}", flush=True)
    print(f"[CFG] NUM_ROUNDS={NUM_ROUNDS}, N_SAMPLES={N_SAMPLES}, RESUME={'ON' if RESUME else 'OFF'}", flush=True)

    agent_ids = HF_MODELS
    repo_list = [model_repo(m) for m in agent_ids]

    agents = load_model_pipelines(
        repo_list=repo_list,
        device_map=MODEL_DEVICE_MAP,
        agent_ids=agent_ids,
        gguf_filename=GGUF_FILENAME,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=VERBOSE,
    )

    for file_idx, qpath in enumerate(QUESTION_PATHS, start=1):
        path = Path(qpath)

        batch_name = "batch_unknown"
        for part in path.parts:
            pl = part.lower()
            if pl.startswith("batch") and pl[5:].isdigit():
                batch_name = pl
                break

        log_dir = Path("logs") / batch_name
        res_dir = Path("results") / batch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        file_key = f"{batch_name}_{path.stem}_{RUN_TAG}_multi"

        log_path = get_log_path(log_dir, file_key)
        csv_path = get_csv_path(res_dir, file_key)

        questions = load_question_blocks(path)
        header = build_header(HF_MODELS, NUM_ROUNDS, N_SAMPLES)

        completed_qids = load_completed_qids_from_csv(csv_path) if RESUME else set()

        if not (RESUME and csv_path.exists()):
            write_csv_header(csv_path, header)

        print(
            f"\n📂 Starting file {file_idx}/{len(QUESTION_PATHS)}: {path.name} "
            f"({len(questions)} questions) | resume={'ON' if RESUME else 'OFF'} "
            f"| completed={len(completed_qids)}",
            flush=True
        )

        skipped, executed = 0, 0

        for qidx, (qid, qtext) in enumerate(questions, start=1):
            if RESUME and (qidx in completed_qids):
                print(f"[SKIP] Q{qidx} already completed (resume)", flush=True)
                skipped += 1
                continue

            print(f"\n[RUN] Q{qidx}: {path.name}", flush=True)
            row = {
                "file": file_key,
                "qid": qidx,
                "question": qtext.replace("\n", " ").strip()
            }

            print(f"[PROCEED] start stage: initial_plan (Q{qidx})", flush=True)
            plans_state = initial_plans_parallel_all(qtext, agents, file_key, qidx, 0, log_path)
            print(f"[SYNC] ✅ initial_plan completed for all models (Q{qidx})", flush=True)

            for rnd in range(1, NUM_ROUNDS + 1):
                print(f"[PROCEED] start stage: relay_round_{rnd} (Q{qidx})", flush=True)

                plans_state, answers, _ = relay_ring_round(
                    qtext=qtext,
                    agents=agents,
                    plans_state=plans_state,
                    model_order=HF_MODELS,
                    file_key=file_key,
                    qidx=qidx,
                    rnd=rnd,
                    log_path=log_path,
                    n_samples=N_SAMPLES,
                )

                print(f"[SYNC] ✅ relay_round_{rnd} completed for all models (Q{qidx})", flush=True)

                for model in HF_MODELS:
                    yn_list = answers.get(model, []) or []

                    if len(yn_list) < N_SAMPLES:
                        yn_list = yn_list + [""] * (N_SAMPLES - len(yn_list))
                    elif len(yn_list) > N_SAMPLES:
                        yn_list = yn_list[:N_SAMPLES]

                    for k, yn in enumerate(yn_list, 1):
                        row[f"{model}_r{rnd}_run{k}"] = yn

            append_csv_row(csv_path, [row.get(col, "") for col in header])
            print(f"[DONE] Q{qidx} results saved → {csv_path.name}", flush=True)
            executed += 1

        print(
            f"\n[SUMMARY] {path.name} | executed={executed}, skipped={skipped}, total={executed+skipped}",
            flush=True
        )

    print("\n✅ multi-model relay finished", flush=True)

if __name__ == "__main__":
    visible = ",".join(sorted(set(HF_DEVICES), key=int))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible)
    print(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    main()