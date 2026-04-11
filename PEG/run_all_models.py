# run_multi_models.py
import os
import subprocess
from pathlib import Path

# ===== 模型配置 =====
# HF_MODELS = [
#     "meta-llama/Llama-3.2-3B-Instruct",
#     "Qwen/Qwen3-4B-Thinking-2507",
#     "google/gemma-3n-E2B-it",
#     "tiiuae/Falcon-E-3B-Instruct",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
# ]
# DEVICES = ["0", "1", "2", "3", "4"]

# ===== 模型配置：三个 gemma3（注意用 :: 后缀让 agent_id 唯一）=====
HF_MODELS = [
    "google/gemma-3-4b-it::g1",
    "google/gemma-3-4b-it::g2",
    "google/gemma-3-4b-it::g3",
]

# 用哪三张卡就写哪三张（例：0,1,2）
DEVICES = ["0", "1", "2"]

# ===== 题目文件：Batch1..5 下的 medical_questions_v1..v5（共 25 个）=====
QUESTION_PATHS = []
for b in range(1, 6):
    for v in range(1, 6):
        p = Path(f"questions/Batch{b}/medical_questions_v{v}.txt")
        if not p.exists():
            raise FileNotFoundError(f"Missing question file: {p}")
        QUESTION_PATHS.append(str(p))

# ===== 环境变量 =====
env = os.environ.copy()

# 只暴露你要用的卡（推荐保留）
env["CUDA_VISIBLE_DEVICES"] = ",".join(DEVICES)

# main_multi_model.py 里 HF_MODELS/HF_DEVICES 数量必须一致
env["HF_MODELS"] = ",".join(HF_MODELS)
env["HF_DEVICES"] = ",".join(DEVICES)

env["NUM_ROUNDS"] = "3"
env["N_SAMPLES"] = env.get("N_SAMPLES", "10")  # 可选：沿用你默认 multi10

# 一次性把 25 个文件路径传给 main_multi_model.py
env["QUESTION_PATHS"] = ",".join(QUESTION_PATHS)
env["RUN_TAG"] = "gemma3x3"

# ===== 启动 =====
subprocess.check_call(["python", "main_multi_model.py"], env=env)