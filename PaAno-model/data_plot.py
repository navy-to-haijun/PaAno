from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "waveform_data"


txt_files = sorted(DATA_DIR.glob("*.txt"))

if not txt_files:
    print(f"No txt files found in: {DATA_DIR}")
    raise SystemExit(1)

for txt_path in txt_files:
    print(f"Processing: {txt_path.name}")

    data = np.loadtxt(txt_path, dtype=np.float32).squeeze()
    x = np.arange(len(data))

    plt.figure(figsize=(14, 4))
    plt.plot(x, data, color="tab:blue", linewidth=1.0)
    plt.title(txt_path.stem)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = txt_path.with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")
