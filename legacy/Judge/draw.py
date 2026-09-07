import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区 =================
REPORT_FILE = "model_performance_report_updated.csv"  # 上一步生成的CSV
OUTPUT_FOLDER = "charts"                     # 存放图片的文件夹
# ==========================================

def plot_batch_performance():
    if not os.path.exists(REPORT_FILE):
        print(f"❌ 错误：找不到文件 {REPORT_FILE}")
        return

    # 读取数据
    df = pd.read_csv(REPORT_FILE)
    
    # 将百分数转换为浮点数 (例如 "85.00%" -> 0.85)
    df['Accuracy_Val'] = df['Accuracy'].str.rstrip('%').astype('float') / 100.0

    # 创建输出文件夹
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    batches = df['Batch'].unique()

    for batch in sorted(batches):
        plt.figure(figsize=(10, 6))
        batch_df = df[df['Batch'] == batch]
        
        # 绘制折线图
        sns.lineplot(data=batch_df, x='Version', y='Accuracy_Val', hue='Model', 
                     marker='o', linewidth=2.5)

        plt.title(f"Model Performance Trend - {batch}", fontsize=15)
        plt.xlabel("Question Version", fontsize=12)
        plt.ylabel("Accuracy (0.0 - 1.0)", fontsize=12)
        plt.ylim(-0.05, 1.05) # 固定纵坐标范围
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存图片
        file_name = f"{OUTPUT_FOLDER}/{batch}_trend.png"
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
        print(f"📊 已生成图表: {file_name}")

if __name__ == "__main__":
    plot_batch_performance()