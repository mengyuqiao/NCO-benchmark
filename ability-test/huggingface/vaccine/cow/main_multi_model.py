# main_multi_model.py
import os, json, csv
from pathlib import Path
from question_loader import load_question_blocks
from model_loader import load_model_pipelines
from peg_core import (
    # 新的“环形接力”流程
    initial_plans_parallel_all,
    relay_ring_round,
)
from log_utils import get_log_path, get_csv_path, write_csv_header, append_csv_row

# ===== 配置（支持环境变量）=====
QUESTION_PATHS = os.getenv("QUESTION_PATHS", "../Questions/NCO/covid_nco_questions_v1.txt").split(",")
HF_MODELS = os.getenv("HF_MODELS", "").split(",") if os.getenv("HF_MODELS") else [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "tiiuae/Falcon3-7B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
]
HF_DEVICES = os.getenv("HF_DEVICES", "").split(",") if os.getenv("HF_DEVICES") else ["0","1","2","3"]
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
RESUME = os.getenv("RESUME", "1") != "0"   # 默认开启断点续跑
N_SAMPLES = int(os.getenv("N_SAMPLES", "10"))  # 每个模型每轮一次性生成的采样数（等价跑 N 次）

assert len(HF_MODELS) == len(HF_DEVICES), "HF_MODELS 与 HF_DEVICES 数量需一致"
MODEL_DEVICE_MAP = {m: int(d) for m, d in zip(HF_MODELS, HF_DEVICES)}

def load_completed_qids_from_csv(csv_path: Path) -> set:
    """读取已存在的结果 CSV，收集其中的 qid（整数）用于断点续跑。"""
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
    """
    CSV 头：'file','qid','question', 以及每个模型在每一轮的 run1..runN 列
    例如：{model}_r1_run1, ..., {model}_r1_run10, {model}_r2_run1, ...
    """
    header = ["file", "qid", "question"]
    for model in models:
        for r in range(1, num_rounds + 1):
            for k in range(1, n_samples + 1):
                header.append(f"{model}_r{r}_run{k}")
    return header

def main():
    print(f"[BOOT] Loading models: {HF_MODELS} with device map {MODEL_DEVICE_MAP}", flush=True)
    print(f"[CFG] NUM_ROUNDS={NUM_ROUNDS}, N_SAMPLES={N_SAMPLES}, RESUME={'ON' if RESUME else 'OFF'}", flush=True)
    agents = load_model_pipelines(HF_MODELS, MODEL_DEVICE_MAP)

    for file_idx, qpath in enumerate(QUESTION_PATHS, start=1):
        path = Path(qpath)
        file_key = path.stem + "_multi"
        log_path = get_log_path(Path("logs/"), file_key)
        csv_path = get_csv_path(Path("results/"), file_key)

        questions = load_question_blocks(path)
        header = build_header(HF_MODELS, NUM_ROUNDS, N_SAMPLES)

        # ===== 断点续跑：已完成集合 =====
        completed_qids = load_completed_qids_from_csv(csv_path) if RESUME else set()

        # 写表头：若 RESUME 且 CSV 已存在，则不重写；否则照常写
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
            # 以 qidx 作为判定（与你当前 CSV 写入保持一致）
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

            # ----- Round 0: 每个模型各自 initial_plan（并行一次，写 [initial plan] 日志）-----
            print(f"[PROCEED] start stage: initial_plan (Q{qidx})", flush=True)
            plans_state = initial_plans_parallel_all(qtext, agents, file_key, qidx, 0, log_path)
            print(f"[SYNC] ✅ initial_plan completed for all models (Q{qidx})", flush=True)

            # ----- Rounds 1..N: 环形接力（上一棒 plan -> 我 refine+answer(Top-N) -> 我产出 plan）-----
            for rnd in range(1, NUM_ROUNDS + 1):
                print(f"[PROCEED] start stage: relay_round_{rnd} (Q{qidx})", flush=True)
                plans_state, answers, _ = relay_ring_round(
                    qtext,
                    agents,
                    plans_state,
                    HF_MODELS,
                    file_key,
                    qidx,
                    rnd,
                    log_path,
                    n_samples=N_SAMPLES  # 关键：一次性生成 Top-N
                )
                print(f"[SYNC] ✅ relay_round_{rnd} completed for all models (Q{qidx})", flush=True)

                # 写入该轮 answers（每个模型的 top-N yes/no）
                # answers: dict[str, List[str]]  # 每模型本轮Top-N yes/no
                for model in HF_MODELS:
                    yn_list = answers.get(model, []) or []
                    # 兜底填充/截断，保证长度恰好 N_SAMPLES
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
