# run_multi_models.py
import os
import subprocess

# ===== 模型配置 =====
HF_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "tiiuae/Falcon3-7B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
]
DEVICES = ["0", "1", "2", "3"]

# ===== 题目文件（10个）=====
QUESTION_PATHS = [
    "Questions/NCO/nco_v1_questions.txt",
    "Questions/NCO/nco_v2_questions.txt",
    "Questions/NCO/nco_v3_questions.txt",
    "Questions/NCO/nco_v4_questions.txt",
    "Questions/NCO/nco_v5_questions.txt",
    "Questions/Positive/medical_questions_v1.txt",
    "Questions/Positive/medical_questions_v2.txt",
    "Questions/Positive/medical_questions_v3.txt",
    "Questions/Positive/medical_questions_v4.txt",
    "Questions/Positive/medical_questions_v5.txt",
]

# ===== 环境变量 =====
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ",".join(DEVICES)
env["HF_MODELS"] = ",".join(HF_MODELS)
env["HF_DEVICES"] = ",".join(DEVICES)
env["NUM_ROUNDS"] = "3"

# ✅ 一次性把10个文件路径传给 main_multi_model.py
env["QUESTION_PATHS"] = ",".join(QUESTION_PATHS)

# ===== 启动 =====
subprocess.check_call(["python", "main_multi_model.py"], env=env)
