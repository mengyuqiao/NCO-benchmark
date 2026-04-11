# question_loader.py

from pathlib import Path
import re
from typing import List, Tuple

def load_question_blocks(filepath: Path) -> List[Tuple[str, str]]:
    """
    加载并按块切分问题：每个问题以 "Question <num>" 开头，直到下一个问题或结尾。

    Returns:
        List of (question_id, question_text)
    """
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    pattern = re.compile(
        r"(?mi)^Question\s*#?\s*(\d+).*?(?=^Question\s*#?\s*\d+|\Z)",
        re.DOTALL
    )
    results = []
    for match in pattern.finditer(text):
        qid = match.group(1)
        block = match.group(0).strip()
        results.append((qid, block))
    return results

def preview_question_ids(questions: List[Tuple[str, str]], k: int = 5) -> List[str]:
    """提取前几个问题的 ID 用于预览"""
    return [qid for qid, _ in questions[:k]]

if __name__ == "__main__":
    path = Path("Questions/NCO/nco_v1_questions.txt")
    qs = load_question_blocks(path)
    print(f"Loaded {len(qs)} questions.")
    print("Preview:", preview_question_ids(qs))