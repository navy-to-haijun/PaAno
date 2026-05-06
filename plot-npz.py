import numpy as np
import matplotlib.pyplot as plt

# 加载 .npz 文件
file_path = 'data/data/waveform3.npz'
data = np.load(file_path)

with np.load(file_path) as data:
    print(f"可用的键: {list(data.keys())}")
    print(f"{data['points']}")

    # 可视化，time和voltage
    plt.plot(data['time'], data['voltage'])
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Waveform')
    plt.show()

