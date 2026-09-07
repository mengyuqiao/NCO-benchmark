import json
import re
import csv
import os
from collections import defaultdict

# ================= 配置区 =================
# 1. 预测结果文件路径
PREDICTIONS_FILE = "runs/20260228_125154/predictions.jsonl" 
# 2. 标准答案文件所在的目录
ANSWERS_DIR = "./answers" 
# 3. 输出统计报告的文件名
OUTPUT_REPORT = "model_performance_report.csv"
# ==========================================

def extract_yes_no(text):
    """
    针对各种模型乱象（复读、截断、推理）的鲁棒提取逻辑
    """
    if not isinstance(text, str) or not text: return "unknown"
    text_lower = text.lower().strip()
    
    # 优先匹配特定的 JSON 片段或结论标记
    json_match = re.search(r'"final_answer"\s*:\s*"(yes|no)"', text_lower)
    if json_match: return json_match.group(1)
    
    match = re.findall(r'final answer:?\s*(yes|no)', text_lower)
    if match: return match[-1]
    
    # 兜底：提取最后出现的独立 yes/no，并排除 common 干扰词
    cleaned = text_lower.replace("no causal", "").replace("no clinical", "")
    words = re.findall(r'\b(yes|no)\b', cleaned)
    return words[-1] if words else "unknown"

def load_simple_ground_truth(directory):
    """
    加载纯文本答案：行号即为 ID
    """
    truth = {}
    for i in range(1, 6):
        file_path = os.path.join(directory, f"medical_answers_b{i}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用 enumerate，从 1 开始计数作为 question_id
                for idx, line in enumerate(f, 1):
                    ans = line.strip().lower()
                    if ans in ['yes', 'no']:
                        truth[(f"Batch{i}", idx)] = ans
    return truth

def main():
    # 1. 加载标准答案
    ground_truth = load_simple_ground_truth(ANSWERS_DIR)
    if not ground_truth:
        print(f"❌ 错误：在 {ANSWERS_DIR} 下未找到任何有效的答案文件。")
        return

    # 2. 统计字典: stats[Batch][Version][Model] = {"correct": 0, "total": 0}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})))

    print(f"🔍 正在读取预测结果并比对答案...")
    
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 解析文件路径中的 Batch 和 Version
            # 预期格式如: ".../Batch1/medical_questions_v2.txt"
            file_path = data.get("file", "")
            batch_match = re.search(r'Batch(\d+)', file_path)
            ver_match = re.search(r'_v(\d+)', file_path)
            
            if not batch_match or not ver_match: continue
            
            batch_key = f"Batch{batch_match.group(1)}"
            version_key = f"v{ver_match.group(1)}"
            qid = data.get("question_id")
            
            # 获取正确答案
            correct_ans = ground_truth.get((batch_key, qid))
            if not correct_ans: continue

            # 提取各模型预测值
            agent_preds = {
                "Qwen": extract_yes_no(data["agents"].get("agent1", "")),
                "Gemma": extract_yes_no(data["agents"].get("agent2", "")),
                "Llama": extract_yes_no(data["agents"].get("agent3", "")),
                "judge": data["judge"].get("final_answer", "unknown")
            }
            
            # 裁判补救提取
            if agent_preds["judge"] not in ["yes", "no"]:
                agent_preds["judge"] = extract_yes_no(data["judge"].get("raw_output", ""))

            # 填充统计数据
            for model_name, pred_ans in agent_preds.items():
                stats[batch_key][version_key][model_name]["total"] += 1
                if pred_ans == correct_ans:
                    stats[batch_key][version_key][model_name]["correct"] += 1

    # 3. 生成数据汇总表格
    report = []
    for batch, versions in sorted(stats.items()):
        for ver, models in sorted(versions.items()):
            for model_name, counts in sorted(models.items()):
                correct = counts["correct"]
                total = counts["total"]
                accuracy = (correct / total) if total > 0 else 0
                
                report.append({
                    "Batch": batch,
                    "Version": ver,
                    "Model": model_name,
                    "Total_Questions": total,
                    "Correct_Answers": correct,
                    "Accuracy": f"{accuracy:.2%}"
                })

    # 4. 写入 CSV
    if report:
        with open(OUTPUT_REPORT, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)
        print(f"✅ 评估报告已生成: {OUTPUT_REPORT}")
    else:
        print("⚠️ 未能生成报告，请检查预测文件与答案文件是否匹配。")

if __name__ == "__main__":
    main()