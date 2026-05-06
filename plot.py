import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取npz文件
# filename = "data/example/405_UCR_id_103_Sensor_tr_2827_1st_5988_scores.csv"  
filename = "data/data/waveform5_scores.csv"  
df = pd.read_csv(filename)

data = df["Data"]
labels = df["True Labels"]
scores = df["Anomaly scores"]

filename = "data/data/waveform2_scores.csv"  
df = pd.read_csv(filename)

data1 = df["Data"]
labels1 = df["True Labels"]
scores1 = df["Anomaly scores"]

data = np.concatenate([data, data1])
labels = np.concatenate([labels, labels1])
scores = np.concatenate([scores, scores1])

# 2. 创建时间轴
t = np.arange(len(data))

# 3. 画图

fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# 上：原始信号
ax[0].plot(t, data, label="Data")
anomaly_idx = labels == 1
ax[0].scatter(t[anomaly_idx], data[anomaly_idx],
              color='red', label="True Anomaly")
ax[0].legend()

# 下：score
ax[1].plot(t, scores, label="Anomaly Score", color='orange')
ax[1].legend()

plt.show()