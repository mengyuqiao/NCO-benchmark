import pandas as pd
from pathlib import Path

# 设置路径
RESULTS_ROOT = Path(r"./results/acc")
OUTPUT_FILE = RESULTS_ROOT / "accuracy.csv"

def merge_accuracy_files():
    # 1. 找到所有以 accuracy_ 开头的 csv 文件
    files = list(RESULTS_ROOT.glob("accuracy_*.csv"))
    
    if not files:
        print(f"在 {RESULTS_ROOT} 下没找到任何 accuracy_*.csv 文件")
        return

    dfs = []
    for f in files:
        print(f"正在读取: {f.name}")
        df = pd.read_csv(f)
        
        # 建议：如果原表里没有 model 列，可以在合并时根据文件名自动加上
        # 这样合并后你还能区分哪行数据属于哪个模型
        if "model" not in df.columns:
            model_name = f.stem.replace("accuracy_", "")
            df["model"] = model_name
            
        dfs.append(df)

    # 2. 合并所有 DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # 3. 保存为新的 csv
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 合并完成！总行数: {len(combined_df)}")
    print(f"文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_accuracy_files()