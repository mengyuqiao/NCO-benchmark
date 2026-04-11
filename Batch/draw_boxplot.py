# from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================
# # 路径配置
# # ======================
# RESULTS_ROOT = Path(r"./results")
# OUT_DIR = Path(r"./figs")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ======================
# # 映射与顺序
# # ======================
# VERSION_TO_PROMPT = {
#     "v1": "plain",
#     "v2": "sys",
#     "v3": "2-shot",
#     "v4": "6-shot",
#     "v5": "10-shot",
# }
# PROMPT_ORDER = ["plain", "sys", "2-shot", "6-shot", "10-shot"]
# BATCH_ORDER = [f"Batch{i}" for i in range(1, 6)]

# METRICS = [
#     ("accuracy", "Accuracy"),
#     ("precision", "Precision"),
#     ("recall", "Recall"),
#     ("f1", "F1"),
# ]

# # ======================
# # 读取所有模型 accuracy.csv
# # ======================
# def load_all_models_metrics():
#     dfs = []
#     for model_dir in RESULTS_ROOT.iterdir():
#         if not model_dir.is_dir():
#             continue
#         acc_path = model_dir / "accuracy_adjusted.csv"
#         if not acc_path.exists():
#             continue

#         df = pd.read_csv(acc_path)
#         if "model" not in df.columns:
#             df["model"] = model_dir.name
#         dfs.append(df)

#     if not dfs:
#         raise RuntimeError("未找到任何 accuracy.csv")

#     df = pd.concat(dfs, ignore_index=True)
#     df["version"] = df["version"].str.lower()
#     df["prompt_type"] = df["version"].map(VERSION_TO_PROMPT)
#     return df


# # ======================
# # 单个 subplot（box + mean line）
# # ======================
# # ======================
# # 单个 subplot（box + mean line）
# # ======================
# def draw_subplot(ax, df, metric, title):
#     models = sorted(df["model"].unique())
#     n_models = len(models)

#     # 1. 为模型生成统一的颜色映射 (使用 matplotlib 默认颜色循环)
#     prop_cycle = plt.rcParams['axes.prop_cycle']
#     colors = prop_cycle.by_key()['color']
#     model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

#     x_centers = np.arange(len(PROMPT_ORDER))
#     total_width = 0.75
#     box_width = total_width / max(n_models, 1)
#     offsets = (np.arange(n_models) - (n_models - 1) / 2) * box_width

#     # 2. 绘制 boxplot（run 分布）
#     for mi, model in enumerate(models):
#         color = model_colors[model]  # 获取当前模型的颜色
#         for pi, prompt in enumerate(PROMPT_ORDER):
#             vals = df[
#                 (df["model"] == model) &
#                 (df["prompt_type"] == prompt)
#             ][metric].dropna().values

#             if len(vals) == 0:
#                 continue

#             pos = x_centers[pi] + offsets[mi]
#             bp = ax.boxplot(
#                 [vals],
#                 positions=[pos],
#                 widths=box_width * 0.85,
#                 patch_artist=True,
#                 manage_ticks=False,
#                 showfliers=True,
#                 # 设置箱线图边框和线条颜色
#                 boxprops=dict(color=color),
#                 capprops=dict(color=color),
#                 whiskerprops=dict(color=color),
#                 flierprops=dict(markeredgecolor=color, marker='.'),
#                 medianprops=dict(color="black"), # 中位数线设为黑色以便区分
#             )
            
#             # 设置填充颜色并调整透明度
#             for b in bp["boxes"]:
#                 b.set_facecolor(color)
#                 b.set_alpha(0.25)

#     # 3. 绘制均值折线
#     for model in models:
#         color = model_colors[model]  # 获取相同的颜色
#         xs, means = [], []
#         for pi, prompt in enumerate(PROMPT_ORDER):
#             sub = df[
#                 (df["model"] == model) &
#                 (df["prompt_type"] == prompt)
#             ][metric].dropna()
#             if len(sub) > 0:
#                 xs.append(x_centers[pi])
#                 means.append(sub.mean())

#         if xs:
#             # 显式传入 color 参数
#             ax.plot(xs, means, marker="o", linewidth=2, label=model, color=color)

#     ax.set_title(title, fontsize=12)
#     ax.set_xticks(x_centers)
#     ax.set_xticklabels(PROMPT_ORDER, fontsize=10)
#     ax.set_ylim(0.0, 1.02)
#     ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# # ======================
# # 每个 Batch 一张 2×2 图
# # ======================
# def plot_batch_metrics(df_all, batch):
#     df = df_all[df_all["batch"] == batch]
#     if df.empty:
#         print(f"[WARN] {batch} 无数据，跳过")
#         return

#     fig, axes = plt.subplots(2, 2, figsize=(15, 9))
#     axes = axes.flatten()

#     for ax, (metric, title) in zip(axes, METRICS):
#         draw_subplot(ax, df, metric, title)

#     # legend 放在右上 subplot
#     axes[1].legend(loc="lower right", framealpha=0.9)

#     fig.suptitle(
#         f"NCO Performance Comparison — {batch}",
#         fontsize=16,
#         y=0.98
#     )

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     out_path = OUT_DIR / f"{batch}_metrics_boxplot_adjusted.png"
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)

#     print(f"✅ Saved: {out_path}")


# # ======================
# # 主流程
# # ======================
# if __name__ == "__main__":
#     df_all = load_all_models_metrics()

#     df_all["batch"] = pd.Categorical(df_all["batch"], BATCH_ORDER, ordered=True)
#     df_all["prompt_type"] = pd.Categorical(df_all["prompt_type"], PROMPT_ORDER, ordered=True)

#     for batch in BATCH_ORDER:
#         plot_batch_metrics(df_all, batch)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 路径配置
# ======================
RESULTS_ROOT = Path(r"./results")
OUT_DIR = Path(r"./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# multi-agent 汇总 csv
MULTI_AGENT_CSV = Path(r"./model_performance_report_updated.csv")

# ======================
# 映射与顺序
# ======================
VERSION_TO_PROMPT = {
    "v1": "plain",
    "v2": "sys",
    "v3": "2-shot",
    "v4": "6-shot",
    "v5": "10-shot",
}
PROMPT_ORDER = ["plain", "sys", "2-shot", "6-shot", "10-shot"]
BATCH_ORDER = [f"Batch{i}" for i in range(1, 6)]

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]

# ======================
# single-agent: 读取所有模型 accuracy_adjusted.csv
# ======================
def load_single_agent_metrics():
    dfs = []
    for model_dir in RESULTS_ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        acc_path = model_dir / "accuracy_adjusted.csv"
        if not acc_path.exists():
            continue

        df = pd.read_csv(acc_path)

        if "model" not in df.columns:
            df["model"] = model_dir.name

        df["model"] = df["model"].astype(str).str.strip().str.lower()
        df["version"] = df["version"].astype(str).str.lower()
        df["prompt_type"] = df["version"].map(VERSION_TO_PROMPT)
        df["source"] = "single"

        dfs.append(df)

    if not dfs:
        raise RuntimeError("未找到任何 single-agent 的 accuracy_adjusted.csv")

    df = pd.concat(dfs, ignore_index=True)
    return df


# ======================
# multi-agent: 读取汇总 csv
# ======================
def load_multi_agent_metrics():
    if not MULTI_AGENT_CSV.exists():
        print(f"[WARN] 未找到 multi-agent CSV: {MULTI_AGENT_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(MULTI_AGENT_CSV)

    # 统一列名
    df = df.rename(columns={
        "Batch": "batch",
        "Version": "version",
        "Model": "model",
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F-1": "f1",
    })

    # 统一字符串格式
    df["batch"] = df["batch"].astype(str)
    df["version"] = df["version"].astype(str).str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    # 百分号转小数
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float) / 100.0
            )

    df["prompt_type"] = df["version"].map(VERSION_TO_PROMPT)

    # 为了和 single-agent 图例区分
    df["model"] = df["model"] + "_multi"
    df["source"] = "multi"

    return df


# ======================
# 同底模同色：gemma 和 gemma_multi 用同一种颜色
# ======================
def build_model_color_map(single_df, multi_df):
    single_models = sorted(single_df["model"].dropna().unique().tolist())
    multi_models = sorted(multi_df["model"].dropna().unique().tolist()) if not multi_df.empty else []

    def base_name(model_name):
        return model_name.replace("_multi", "")

    base_models = sorted(set(base_name(m) for m in single_models + multi_models))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    base_color_map = {m: colors[i % len(colors)] for i, m in enumerate(base_models)}

    full_color_map = {}
    for m in single_models + multi_models:
        full_color_map[m] = base_color_map[base_name(m)]

    return full_color_map


# ======================
# 单个 subplot
# single-agent: box + mean line
# multi-agent: dashed line + star marker
# ======================
def draw_subplot(ax, single_df, multi_df, metric, title, model_colors):
    x_centers = np.arange(len(PROMPT_ORDER))

    # ---------- 画 single-agent boxplot ----------
    single_models = sorted(single_df["model"].dropna().unique())
    n_single = len(single_models)

    total_width = 0.75
    box_width = total_width / max(n_single, 1)
    offsets = (np.arange(n_single) - (n_single - 1) / 2) * box_width

    for mi, model in enumerate(single_models):
        color = model_colors[model]

        for pi, prompt in enumerate(PROMPT_ORDER):
            vals = single_df[
                (single_df["model"] == model) &
                (single_df["prompt_type"] == prompt)
            ][metric].dropna().values

            if len(vals) == 0:
                continue

            pos = x_centers[pi] + offsets[mi]
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
                boxprops=dict(color=color),
                capprops=dict(color=color),
                whiskerprops=dict(color=color),
                flierprops=dict(markeredgecolor=color, marker='.'),
                medianprops=dict(color="black"),
            )

            for b in bp["boxes"]:
                b.set_facecolor(color)
                b.set_alpha(0.25)

    # ---------- 画 single-agent mean line ----------
    for model in single_models:
        color = model_colors[model]
        xs, means = [], []

        for pi, prompt in enumerate(PROMPT_ORDER):
            sub = single_df[
                (single_df["model"] == model) &
                (single_df["prompt_type"] == prompt)
            ][metric].dropna()

            if len(sub) > 0:
                xs.append(x_centers[pi])
                means.append(sub.mean())

        if xs:
            ax.plot(
                xs, means,
                marker="o",
                linewidth=2,
                linestyle="-",
                color=color,
                label=f"{model} (single)"
            )

    # ---------- 叠加 multi-agent ----------
    if not multi_df.empty:
        multi_models = sorted(multi_df["model"].dropna().unique())

        for model in multi_models:
            color = model_colors[model]
            xs, ys = [], []

            for pi, prompt in enumerate(PROMPT_ORDER):
                sub = multi_df[
                    (multi_df["model"] == model) &
                    (multi_df["prompt_type"] == prompt)
                ][metric].dropna()

                if len(sub) > 0:
                    xs.append(x_centers[pi])
                    ys.append(sub.iloc[0])  # 每个 batch/version/model 只有一个汇总值

            if xs:
                ax.plot(
                    xs, ys,
                    marker="*",
                    markersize=10,
                    linewidth=2.2,
                    linestyle="--",
                    color=color,
                    label=f"{model} (multi)"
                )

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(PROMPT_ORDER, fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# ======================
# 每个 Batch 一张 2×2 图
# ======================
def plot_batch_metrics(single_all, multi_all, batch, model_colors):
    single_df = single_all[single_all["batch"] == batch].copy()
    multi_df = multi_all[multi_all["batch"] == batch].copy() if not multi_all.empty else pd.DataFrame()

    if single_df.empty and multi_df.empty:
        print(f"[WARN] {batch} 无数据，跳过")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRICS):
        draw_subplot(ax, single_df, multi_df, metric, title, model_colors)

    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(
        unique.values(),
        unique.keys(),
        loc="lower right",
        framealpha=0.9,
        fontsize=9
    )

    fig.suptitle(
        f"NCO Performance Comparison — {batch}",
        fontsize=16,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / f"{batch}_metrics_boxplot_with_multi_agent.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    # single-agent
    single_all = load_single_agent_metrics()
    single_all["batch"] = pd.Categorical(single_all["batch"], BATCH_ORDER, ordered=True)
    single_all["prompt_type"] = pd.Categorical(single_all["prompt_type"], PROMPT_ORDER, ordered=True)

    # multi-agent
    multi_all = load_multi_agent_metrics()
    if not multi_all.empty:
        multi_all["batch"] = pd.Categorical(multi_all["batch"], BATCH_ORDER, ordered=True)
        multi_all["prompt_type"] = pd.Categorical(multi_all["prompt_type"], PROMPT_ORDER, ordered=True)

    # 统一颜色
    model_colors = build_model_color_map(single_all, multi_all)

    # 逐 batch 画图
    for batch in BATCH_ORDER:
        plot_batch_metrics(single_all, multi_all, batch, model_colors)