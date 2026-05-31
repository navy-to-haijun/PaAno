from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


base_dir = Path(__file__).parent
score_dir = base_dir / "data" / "anomaly_scores"

csv_files = sorted(score_dir.glob("*_scores*.csv"))

if not csv_files:
    print("没有找到csv文件")
    exit()

for csv_path in csv_files:
    print(f"处理: {csv_path.name}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    index = df["index"].to_numpy()
    values = df["value"].to_numpy()
    scores = df["anomaly_score"].to_numpy()

    # 分数归一化到0-1
    scrores_min = 0
    scores_max = 0.05
    scores = (scores - scrores_min) / (scores_max - scrores_min)

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # value
    axes[0].plot(index, values, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)

    # anomaly score(限制y轴范围)
    axes[1].plot(index, scores, color="tab:red", linewidth=1.2)
    axes[1].set_ylabel("anomaly_score")
    axes[1].set_xlabel("index")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(csv_path.stem)

    plt.tight_layout()

    # 保存到同目录
    out_path = csv_path.with_suffix(".png")

    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"保存: {out_path}")