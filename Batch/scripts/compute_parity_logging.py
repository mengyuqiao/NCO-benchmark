
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

VALID_EXTRACTED = {"yes", "no", "unparseable"}


class CallLogger:
    """Thread-safe JSONL append logger for one experiment run.

    One instance owns one run directory: runs/<timestamp>_<config>/calls.jsonl
    (or an existing directory passed via `run_dir`, to resume a run that was
    started earlier). Never overwrites -- always appends.
    """

    def __init__(self, config, base_dir="runs", run_dir=None):
        self._lock = threading.Lock()
        self.config = config
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            self.run_dir = Path(base_dir) / f"{ts}_{config}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "calls.jsonl"
        self._done_cache = self._load_done_cache_locked()  # loaded eagerly, not lazily

    @staticmethod
    def _key(task, batch, qid, run, sample):
        return (task, batch, qid, run, sample)

    def log_call(
        self, *, task, batch, qid, gold, config, model, prompt_version, run, sample,
        seed, temperature, top_p, max_new_tokens, prompt_tokens,
        completion_tokens, raw_output, extracted, timestamp=None,
    ):
        """Append one call record. Returns the record as written."""
        if extracted not in VALID_EXTRACTED:
            raise ValueError(
                f"extracted must be one of {sorted(VALID_EXTRACTED)}, got {extracted!r} "
                "-- do not pass a defaulted value here (the evaluation protocol)"
            )
        if gold not in ("yes", "no"):
            raise ValueError(f"gold must be 'yes' or 'no' (it's ground truth, never unparseable), got {gold!r}")
        record = {
            "task": task,
            "batch": batch,
            "qid": qid,
            "gold": gold,
            "config": config,
            "model": model,
            "prompt_version": prompt_version,
            "run": run,
            "sample": sample,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "raw_output": raw_output,  # verbatim, untruncated -- do not slice this
            "extracted": extracted,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._done_cache.add(self._key(task, batch, qid, run, sample))
        return record

    def _load_done_cache_locked(self):
        done = set()
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    done.add(self._key(rec["task"], rec["batch"], rec["qid"], rec["run"], rec["sample"]))
        return done

    def already_done(self, task, batch, qid, run, sample):
        """True if a record for this (task, batch, qid, run, sample) already
        exists in this run's JSONL on disk. Used for resumable scripts:
        check before making a model call, skip if already logged."""
        with self._lock:
            return self._key(task, batch, qid, run, sample) in self._done_cache

    def read_all(self):
        """Read back every record currently in this run's JSONL. For
        verification / debugging, not for the hot path."""
        records = []
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records
