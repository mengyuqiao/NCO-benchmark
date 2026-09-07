from pathlib import Path
from datetime import datetime
import csv


def get_log_path(base_dir: Path, file_key: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"peg_logs_{file_key}.txt"

def get_csv_path(base_dir: Path, file_key: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"peg_answers_{file_key}_new.csv"

def write_csv_header(csv_path: Path, header: list):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def append_csv_row(csv_path: Path, row: list):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def format_log_entry(file_name: str, question_num: int, round_num: int, model_name: str, stage: str, message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}][{file_name}][Question {question_num}][Round {round_num}][{model_name}][{stage}] {message}\n"

def append_log(log_path: Path, entry: str):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    # 只打印 entry 的头部信息（不包含正文内容）
    first_line = entry.strip().split("] ")[0] + "]"
    print(first_line)