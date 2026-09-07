# run_multi_models_qwen3.py
import os
import subprocess
from pathlib import Path

HF_MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507::q1",
    "Qwen/Qwen3-4B-Thinking-2507::q2",
    "Qwen/Qwen3-4B-Thinking-2507::q3",
]

# 物理 GPU
PHYSICAL_GPUS = ["3", "4", "5"]
# 进程内可见 GPU 索引（与 PHYSICAL_GPUS 一一对应）
LOCAL_GPUS = ["0", "1", "2"]

QUESTION_PATHS = []
for b in range(1, 6):
    for v in range(1, 6):
        p = Path(f"questions/Batch{b}/medical_questions_v{v}.txt")
        if not p.exists():
            raise FileNotFoundError(f"Missing question file: {p}")
        QUESTION_PATHS.append(str(p))

env = os.environ.copy()

# 只暴露物理 3,4,5 给当前进程
env["CUDA_VISIBLE_DEVICES"] = ",".join(PHYSICAL_GPUS)

# 三个 agent（唯一 id）
env["HF_MODELS"] = ",".join(HF_MODELS)

# 关键：这里必须用 0,1,2（进程内索引），不要用 3,4,5
env["HF_DEVICES"] = ",".join(LOCAL_GPUS)

env["NUM_ROUNDS"] = "3"
env["N_SAMPLES"] = env.get("N_SAMPLES", "10")
env["QUESTION_PATHS"] = ",".join(QUESTION_PATHS)
env["RUN_TAG"] = "qwen3x3"

subprocess.check_call(["python", "main_multi_model.py"], env=env)