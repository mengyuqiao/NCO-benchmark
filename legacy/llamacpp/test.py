import os
import re
import csv
import json
import time
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path


REPO_ID = "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill-GGUF"
FILENAME = "Qwen3.5-9B.Q4_K_M.gguf"

DEFAULT_AGENT_IDS = ["g1", "g2", "g3"]


def read_questions(txt_path: Path) -> list[str]:
    text = txt_path.read_text(encoding="utf-8")
    parts = re.split(r"(?=Question \d+)", text)
    return [p.strip() for p in parts if p.strip()]


def extract_yes_no(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
    else:
        tail = text

    tail_lower = tail.lower()

    # 优先找独立 yes/no
    m = re.search(r"\b(yes|no)\b", tail_lower)
    if m:
        return m.group(1)

    # 兜底：看最后几行
    lines = [line.strip().lower() for line in tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if line == "yes":
            return "yes"
        if line == "no":
            return "no"
        if "yes" in line:
            return "yes"
        if "no" in line:
            return "no"

    return ""


def build_round1_prompt(question_block: str) -> str:
    return (
        f"{question_block}\n\n"
        "Important: Reply with only one word: yes or no."
    )


def build_review_prompt(question_block: str, previous_raw: str, previous_agent: str) -> str:
    return (
        f"{question_block}\n\n"
        f"Previous reviewer ({previous_agent}) answered:\n"
        f"{previous_raw}\n\n"
        "Review the previous answer carefully.\n"
        "You may agree or disagree.\n"
        "Reply with only one word: yes or no."
    )


def worker_main(
    agent_id: str,
    gpu_id: int,
    repo_id: str,
    filename: str,
    n_ctx: int,
    n_gpu_layers: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from llama_cpp import Llama

        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            main_gpu=0,
            verbose=False,
        )

        out_queue.put({
            "type": "ready",
            "agent_id": agent_id,
            "gpu_id": gpu_id,
        })

        while True:
            item = in_queue.get()
            if item is None:
                break

            task_id = item["task_id"]
            prompt = item["prompt"]
            max_tokens = item["max_tokens"]
            temperature = item["temperature"]

            try:
                output = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = output["choices"][0]["message"]["content"]
                ans = extract_yes_no(raw)

                out_queue.put({
                    "type": "result",
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "raw": raw,
                    "answer": ans,
                })
            except Exception as e:
                out_queue.put({
                    "type": "error",
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    except Exception as e:
        out_queue.put({
            "type": "fatal",
            "agent_id": agent_id,
            "gpu_id": gpu_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


class AgentPool:
    def __init__(
        self,
        agent_ids: list[str],
        gpu_ids: list[int],
        repo_id: str,
        filename: str,
        n_ctx: int,
        n_gpu_layers: int,
    ):
        assert len(agent_ids) == len(gpu_ids), "agent_ids 和 gpu_ids 数量必须一致"

        self.agent_ids = agent_ids
        self.gpu_ids = gpu_ids
        self.in_queues = {}
        self.out_queue = mp.Queue()
        self.procs = {}

        for agent_id, gpu_id in zip(agent_ids, gpu_ids):
            iq = mp.Queue()
            p = mp.Process(
                target=worker_main,
                args=(
                    agent_id,
                    gpu_id,
                    repo_id,
                    filename,
                    n_ctx,
                    n_gpu_layers,
                    iq,
                    self.out_queue,
                ),
                daemon=True,
            )
            p.start()
            self.in_queues[agent_id] = iq
            self.procs[agent_id] = p

        ready = set()
        while len(ready) < len(agent_ids):
            msg = self.out_queue.get()
            if msg["type"] == "ready":
                ready.add(msg["agent_id"])
                print(f"[READY] {msg['agent_id']} on GPU {msg['gpu_id']}", flush=True)
            else:
                raise RuntimeError(f"Worker init failed: {msg}")

    def submit(self, agent_id: str, task_id: str, prompt: str, max_tokens: int, temperature: float):
        self.in_queues[agent_id].put({
            "task_id": task_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

    def gather(self, expected_task_ids: set[str]) -> dict[str, dict]:
        results = {}
        while len(results) < len(expected_task_ids):
            msg = self.out_queue.get()

            if msg["type"] == "result":
                task_id = msg["task_id"]
                if task_id in expected_task_ids:
                    results[task_id] = msg
            elif msg["type"] in ("error", "fatal"):
                raise RuntimeError(
                    f"Worker {msg.get('agent_id')} failed:\n{msg.get('error')}\n{msg.get('traceback')}"
                )
            else:
                raise RuntimeError(f"Unexpected message: {msg}")

        return results

    def close(self):
        for q in self.in_queues.values():
            q.put(None)
        for p in self.procs.values():
            p.join(timeout=5)


def build_header(agent_ids: list[str], num_rounds: int, n_samples: int) -> list[str]:
    header = ["file", "qid", "question"]
    for agent_id in agent_ids:
        for rnd in range(1, num_rounds + 1):
            for run in range(1, n_samples + 1):
                header.append(f"{agent_id}_r{rnd}_run{run}")
    return header


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def append_csv_row(csv_path: Path, header: list[str], row: dict):
    ensure_parent(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


def save_jsonl(log_path: Path, record: dict):
    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one_question(
    question_block: str,
    qid: int,
    file_key: str,
    pool: AgentPool,
    agent_ids: list[str],
    num_rounds: int,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    log_path: Path,
):
    row = {
        "file": file_key,
        "qid": qid,
        "question": question_block.replace("\n", " ").strip(),
    }

    for run_idx in range(1, n_samples + 1):
        prev_raw_by_agent = {}

        for rnd in range(1, num_rounds + 1):
            expected_task_ids = set()

            for i, agent_id in enumerate(agent_ids):
                task_id = f"{file_key}|q{qid}|run{run_idx}|r{rnd}|{agent_id}"

                if rnd == 1:
                    prompt = build_round1_prompt(question_block)
                else:
                    prev_agent = agent_ids[(i - 1) % len(agent_ids)]
                    prev_raw = prev_raw_by_agent[prev_agent]
                    prompt = build_review_prompt(question_block, prev_raw, prev_agent)

                pool.submit(
                    agent_id=agent_id,
                    task_id=task_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                expected_task_ids.add(task_id)

            results = pool.gather(expected_task_ids)

            current_raw_by_agent = {}
            for agent_id in agent_ids:
                task_id = f"{file_key}|q{qid}|run{run_idx}|r{rnd}|{agent_id}"
                msg = results[task_id]

                raw = msg["raw"]
                ans = msg["answer"]
                current_raw_by_agent[agent_id] = raw

                row[f"{agent_id}_r{rnd}_run{run_idx}"] = ans

                save_jsonl(log_path, {
                    "file": file_key,
                    "qid": qid,
                    "run": run_idx,
                    "round": rnd,
                    "agent": agent_id,
                    "raw": raw,
                    "answer": ans,
                })

            prev_raw_by_agent = current_raw_by_agent

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_root", type=str, default="questions")
    parser.add_argument("--output_root", type=str, default="results_llamacpp")
    parser.add_argument("--log_root", type=str, default="logs_llamacpp")
    parser.add_argument("--gpus", type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_ctx", type=int, default=16384)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--filename", type=str, default=FILENAME)
    args = parser.parse_args()

    questions_root = Path(args.questions_root)
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)

    question_files = sorted(questions_root.glob("Batch*/medical_questions_v*.txt"))
    if not question_files:
        raise FileNotFoundError(f"No files found under {questions_root}/Batch*/medical_questions_v*.txt")

    print("[INFO] Found question files:")
    for p in question_files:
        print(f"  - {p}", flush=True)

    mp.set_start_method("spawn", force=True)

    pool = AgentPool(
        agent_ids=DEFAULT_AGENT_IDS,
        gpu_ids=args.gpus,
        repo_id=REPO_ID,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    header = build_header(DEFAULT_AGENT_IDS, args.num_rounds, args.n_samples)

    try:
        for qfile in question_files:
            batch_name = qfile.parent.name.lower()
            file_key = f"{batch_name}_{qfile.stem}_rolling"

            csv_path = output_root / batch_name / f"{qfile.stem}_rolling.csv"
            log_path = log_root / batch_name / f"{qfile.stem}_rolling.jsonl"

            questions = read_questions(qfile)
            print(f"\n[FILE] {qfile} | total_questions={len(questions)}", flush=True)

            for qidx, question_block in enumerate(questions, start=1):
                t0 = time.time()
                print(f"[RUN] {qfile.name} | Q{qidx}/{len(questions)}", flush=True)

                row = run_one_question(
                    question_block=question_block,
                    qid=qidx,
                    file_key=file_key,
                    pool=pool,
                    agent_ids=DEFAULT_AGENT_IDS,
                    num_rounds=args.num_rounds,
                    n_samples=args.n_samples,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    log_path=log_path,
                )

                append_csv_row(csv_path, header, row)
                dt = time.time() - t0
                print(f"[DONE] {qfile.name} | Q{qidx} | {dt:.2f}s", flush=True)

            print(f"[SAVED] {csv_path}", flush=True)

    finally:
        pool.close()


if __name__ == "__main__":
    main()