import matplotlib.pyplot as plt
import numpy as np


def log_signal_with_gt_anomaly(writer, test_data, dist_scores, true_labels, step=0):
    t = np.arange(len(test_data))

    test_data = np.asarray(test_data)
    dist_scores = np.asarray(dist_scores)
    true_labels = np.asarray(true_labels)

    # score 归一化
    score = (dist_scores - dist_scores.min()) / (dist_scores.max() - dist_scores.min() + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # =========================
    # 1. 上图：原始信号 + GT
    # =========================
    axes[0].plot(t, test_data, label="signal", color="blue")

    anomaly_idx = true_labels == 1

    axes[0].scatter(
        t[anomaly_idx],
        test_data[anomaly_idx],
        color="red",
        s=10,
        label="GT anomaly"
    )

    # 红色区间高亮
    for i in range(len(t)):
        if anomaly_idx[i]:
            axes[0].axvspan(i, i + 1, color="red", alpha=0.08)

    axes[0].set_title("Signal with Ground Truth Anomaly")
    axes[0].legend()

    # =========================
    # 2. 下图：Anomaly Score
    # =========================
    axes[1].plot(t, score, label="anomaly score", color="orange")

    axes[1].set_title("Anomaly Score")
    axes[1].legend()

    plt.tight_layout()

    writer.add_figure("Visualization/gt_anomaly_dual", fig, step)
    plt.close()