import numpy as np
import pandas as pd

from utils.LoggerConfig import init_logger
import torch
from torch.utils.data import Dataset, DataLoader

# import torch.nn.functional as F
# 
# import numpy as np
# import os

# import warnings
# from sklearn.exceptions import ConvergenceWarning


logger = init_logger()

def load_data(file_path) :
    """
    读取csv数据, 返回数据集和标签
    """
    try:
        df = pd.read_csv(file_path)  
        if df.empty: 
            logger.error(f"空文件: {file_path}")
            return [], []
        # 读取除lablel列以外的其他列数据（尝试将列size=1压缩掉)）
        data = df.iloc[:, :-1].squeeze(axis=1) 
        # 读取标签列（最后一列）
        labels = df.iloc[:, -1]
        logger.info(f"data shape: {data.shape}") 
        logger.info(f"labels shape: {labels.shape}")

        return data, labels
    except Exception as e:
        logger.error(f"读取文件时出错 {file_path}: {e}")
        return [], []

def load_npz_data(file_path):
    """
    加载 .npz 文件
    """
    try:
        data = np.load(file_path)
        if 'points' in data.keys():
            points = data['points']
            time = data['time']
            voltage = data['voltage']

            logger.info(f"points: {points}")

            train_data = voltage[:8000000]
            train_labels = np.zeros(len(train_data))
            test_data = voltage[8000000:10000000]
            test_labels = np.zeros(len(test_data))

            return train_data, train_labels, test_data, test_labels
        else:
            logger.error(f"文件 {file_path} 中没有points字段")
            return None, None, None, None
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {e}")
        return None, None, None, None

def load_and_split_data(file_path):
    """
    加载数据集，将数据划分为训练集和测试集

    Args:
        file_path (str): 
        文件路径

    Returns:
        train_data (numpy.ndarray): 训练数据
        train_labels (numpy.ndarray): 训练标签
        test_data (numpy.ndarray): 测试数据
        test_labels (numpy.ndarray): 测试标签
    """
    try:
        data, labels = load_data(file_path)
        if len(data) == 0 or len(labels) == 0:  
            logger.error(f"Empty data or labels for file: {file_path}")
            return [], [], [], []

    
        file_name_parts = file_path.name.split('/')[-1].split('_')       # 获取文件名的各个字段
        train_end_end = int(file_name_parts[-3])                        # 获取训练集的结束索引

        if train_end_end:  
            train_data = data[:train_end_end]
            train_labels = labels[:train_end_end]

            test_data = data[train_end_end:]
            test_labels = labels[train_end_end:]

            return train_data, train_labels, test_data, test_labels
        else:
            logger.error(f"解析文件名称失败: {file_name_parts}")
            return [], [], [], []
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return None, None, None, None


def preprocess_to_patches(data, patch_size, stride):
    patches = []
    for i in range(0, len(data) - patch_size + 1, stride):
        patch = data[i:i + patch_size]
        patches.append(patch)
    
    patches_array = np.array(patches)                      # (N, L) or (N, L, C)
    t = torch.tensor(patches_array, dtype=torch.float32)
    if t.ndim == 2:                   # (N, L) -> (N, 1, L)
        t = t.unsqueeze(1).contiguous()
    elif t.ndim == 3:                 # (N, L, C) -> (N, C, L)
        t = t.permute(0, 2, 1).contiguous()

    return t    


class _tsdataset(Dataset):
    """
    实现pytorch Dataset接口
    """
    def __init__(self, data, indices=None):
        # indice means relative order among patches 
        self.data = torch.from_numpy(np.array(data)).float()
        if indices is not None:
            self.indices = torch.from_numpy(np.array(indices)).long().unsqueeze(1)
        else:
            self.indices = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]      
            # x: (L,) or (L, C) or (C, L)
        if x.ndim == 1:          # (L,) -> (1, L)
            x = x.unsqueeze(0).contiguous()
       
        if self.indices is not None:
            return x, self.indices[idx]
        return x, torch.tensor([idx])




class PatchCreator:
    '''

    基于 patch 创建用于训练的和测试的数据集 
    Args:
        L (int): 窗口长度
        s (int): 窗口步长
    '''
    def __init__(self, L, s):
        self.L = L 
        self.s = s  
    def create_patches(self, data):
        """
        将数据集划分为patch
        """

        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data must be a list or numpy array.")
        if len(data) < self.L:
            raise ValueError(f"Data length {len(data)} is less than patch size {self.L}.")
    
        num_patches = (len(data) - self.L) // self.s + 1
        # 按照窗口长度和步长进行切片
        patches = []
        indices = []
        for i in range(0, num_patches, self.s):
            patch = data[i:i + self.L]
            patches.append(patch)
            indices.append(i)

        return patches, indices

    def create_dataloaders(self, train_data, test_data, test_labels, batch_size=512):
        
        train_patches, train_indices = self.create_patches(train_data)
        test_patches, test_indices = self.create_patches(test_data)

        logger.info(f"训练集 patches 数量: {len(train_patches)}")
        logger.info(f"测试集 patches 数量: {len(test_patches)}")
    
        train_loader = DataLoader(_tsdataset(train_patches, indices=train_indices), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(_tsdataset(test_patches, indices=test_indices), batch_size=batch_size, shuffle=False)

        true_test_labels = test_labels

        return train_loader, test_loader, true_test_labels
    
    def create_all_dataloader(self, all_data, batch_size=512):
        all_patches, all_indices = self.create_patches(all_data)

        logger.info(f"所有数据 patches 数量: {len(all_patches)}")
        all_loader = DataLoader(_tsdataset(all_patches, indices=all_indices), batch_size=batch_size, shuffle=True)

        return all_loader



# # cited from https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py
# from statsmodels.tsa.stattools import acf
# from scipy.signal import argrelextrema
# import numpy as np
# from statsmodels.graphics.tsaplots import plot_acf

# # determine sliding window (period) based on ACF
# def find_length_rank(data, rank=1):
#     """"
#     自动寻找最优窗口长度
#     """
#     data = data.squeeze()
#     if len(data.shape)>1: return 100 #0->100
#     if rank==0: return 1
#     data = data[:min(20000, len(data))]
    
#     base = 3
#     auto_corr = acf(data, nlags=400, fft=True)[base:]
    
#     # plot_acf(data, lags=400, fft=True)
#     # plt.xlabel('Lags')
#     # plt.ylabel('Autocorrelation')
#     # plt.title('Autocorrelation Function (ACF)')
#     # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

#     local_max = argrelextrema(auto_corr, np.greater)[0]

#     # print('auto_corr: ', auto_corr)
#     # print('local_max: ', local_max)

#     try:
#         # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
#         sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
#         max_local_max = sorted_local_max[0]     # Default
#         if rank == 1: max_local_max = sorted_local_max[0]
#         if rank == 2: 
#             for i in sorted_local_max[1:]: 
#                 if i > sorted_local_max[0]: 
#                     max_local_max = i 
#                     break
#         if rank == 3:
#             for i in sorted_local_max[1:]: 
#                 if i > sorted_local_max[0]: 
#                     id_tmp = i
#                     break
#             for i in sorted_local_max[id_tmp:]:
#                 if i > sorted_local_max[id_tmp]: 
#                     max_local_max = i           
#                     break
#         # print('sorted_local_max: ', sorted_local_max)
#         # print('max_local_max: ', max_local_max)
#         if local_max[max_local_max]<3 or local_max[max_local_max]>300:
#             return 125
#         return local_max[max_local_max]+base
#     except:
#         return 125
    

# # determine sliding window (period) based on ACF, Original version
# def find_length(data):
#     if len(data.shape)>1:
#         return 0
#     data = data[:min(20000, len(data))]
    
#     base = 3
#     auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
#     local_max = argrelextrema(auto_corr, np.greater)[0]
#     try:
#         max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
#         if local_max[max_local_max]<3 or local_max[max_local_max]>300:
#             return 125
#         return local_max[max_local_max]+base
#     except:
#         return 125

