import json
import re
import csv
import os

# ================= 配置区 =================
# 1. 输入文件夹地址 (如果是当前目录，留空 "" 即可)
INPUT_DIR = "runs/20260228_125154" 
# 2. 输入文件名
INPUT_FILENAME = "predictions.jsonl"
# 3. 输出文件夹地址
OUTPUT_DIR = "results"
# 4. 输出文件名 (CSV格式)
OUTPUT_FILENAME = "extracted_data.csv"
# ==========================================

def extract_yes_no(text):
    """
    鲁棒性提取：从包含推理、复读或截断的文本中提取最终结论
    """
    if not isinstance(text, str):
        return "unknown"
    
    text_lower = text.lower().strip()
    
    # 策略1：匹配标准的 Final Answer 格式
    match = re.findall(r'final answer:\s*(yes|no)', text_lower)
    if match:
        return match[-1]
    
    # 策略2：匹配推理结尾的 "the answer is yes/no"
    match = re.findall(r'answer is[:\s]*(yes|no)', text_lower)
    if match:
        return match[-1]
    
    # 策略3：极端情况，提取最后出现的独立单词
    words = re.findall(r'\b(yes|no)\b', text_lower)
    if words:
        return words[-1]

    return "unknown"

def main():
    # 自动拼接完整路径
    input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件 '{input_path}'，请检查配置。")
        return

    # 自动创建输出目录
    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已创建输出目录: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    results = []
    print(f"🔍 正在读取: {input_path} ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            # 尝试解析标准JSON列表
            data = json.loads(content)
            if not isinstance(data, list): data = [data]
        except json.JSONDecodeError:
            # 尝试解析JSONL格式
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    for entry in data:
        # 获取基础信息
        qid = entry.get('question_id', 'N/A')
        source_file = os.path.basename(entry.get('file', 'unknown'))
        
        # 提取 Agents 的答案
        agents = entry.get('agents', {})
        a1 = extract_yes_no(agents.get('agent1', ''))
        a2 = extract_yes_no(agents.get('agent2', ''))
        a3 = extract_yes_no(agents.get('agent3', ''))
        
        # 提取 Judge 的答案 (包含对 raw_output 的二次提取)
        judge_data = entry.get('judge', {})
        j_ans = judge_data.get('final_answer', 'unknown')
        if j_ans not in ['yes', 'no'] and 'raw_output' in judge_data:
            j_ans = extract_yes_no(judge_data['raw_output'])
            
        results.append({
            'Question_ID': qid,
            'Source_File': source_file,
            'Agent1': a1,
            'Agent2': a2,
            'Agent3': a3,
            'Judge': j_ans
        })

    # 写入 CSV
    if results:
        keys = results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ 提取成功！共计 {len(results)} 条数据。")
        print(f"📊 结果保存位置: {output_path}")
    else:
        print("⚠️ 未能提取到有效数据。")

if __name__ == "__main__":
    main()