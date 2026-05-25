import numpy as np
import pandas as pd
from pathlib import Path


def txt_to_csv_with_label(txt_file):
    txt_file = Path(txt_file)

    if not txt_file.exists():
        raise FileNotFoundError(f"文件不存在: {txt_file}")

    data = np.loadtxt(txt_file)

   # 解析数据
    parts = txt_file.stem.split("_")

    try:
        anomaly_start = int(parts[-2])
        anomaly_end = int(parts[-1])
    except Exception:
        raise ValueError(f"文件名无法解析异常区间: {txt_file.name}")
    # 生成标签
    labels = np.zeros(len(data), dtype=int)

    # 防止越界
    anomaly_start = max(0, anomaly_start)
    anomaly_end = min(len(data) - 1, anomaly_end)

    print(f"[INFO] 异常区间: {anomaly_start} 到 {anomaly_end}")

    labels[anomaly_start:anomaly_end + 1] = 1

    df = pd.DataFrame({
        "Data": data,
        "Label": labels
    })

    # 保存CSV文件
    csv_file = txt_file.with_suffix(".csv")
    df.to_csv(csv_file, index=False)

    print(f"[OK] {txt_file.name} -> {csv_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TXT to CSV converter with labels")
    parser.add_argument("file", type=str, help="input txt file path")

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    print(f"[INFO] 基本文件路径: {base_dir}")
    txt_file = base_dir.joinpath("data", "example", args.file)
    print(f"[INFO] 输入文件: {txt_file}")

    txt_to_csv_with_label(txt_file)