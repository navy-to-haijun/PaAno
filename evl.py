import os
import time
from argparse import ArgumentParser
from pathlib import Path
from utils.LoggerConfig import init_logger
import yaml
import numpy as np
import torch
from preprocessing.data_preprocess import *
from models.model import PatchEncoder
from models.train_model import train_model
from torch.utils.tensorboard import SummaryWriter
from utils.testplot import log_signal_with_gt_anomaly


import pandas as pd
from tqdm import tqdm


from preprocessing.data_preprocess import *
from utils.utils import *
from utils.evaluation import *


class EvlAnomalyDetection:
    """
    评估异常检测模型

    Args:
        config: 配置字典
    """
    def __init__(self, config:dict, device=None):
        # 获取参数
        self.config = config
        self.date_file = self.config['dataset']["date_file"]
        self.patch_size = self.config['dataset']["patch_size"]
        self.batch_size = self.config['model']["batch_size"]
        self.in_channels = self.config['model']["in_channels"]
        self.log_dir = self.config['model']['log_dir']
        self.num_iters = self.config['model']["num_iters"]
        self.lr = float(self.config['model']["lr"])
        self.base_dir = Path(__file__).parent
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def loaddateset(self):
        # 拼接路径
        file_path = self.base_dir.joinpath(self.date_file)
        # 加载数据集
        # train_data, train_labels, test_data, test_labels = load_and_split_data(file_path)
        train_data, train_labels, test_data, test_labels = load_npz_data(file_path)
        # 转换数据类型
        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32) 
        test_data = np.array(test_data, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)
       
        # 拼接数据
        # all_data = np.concatenate([train_data, test_data], axis=0)
        # all_labels = np.concatenate([train_labels, test_labels], axis=0)
        all_data = test_data
        all_labels = test_labels
        # dataloader
        patch_creator = PatchCreator(L=self.patch_size, s=1)
   
        all_loader = patch_creator.create_all_dataloader(all_data, batch_size=self.batch_size)
        # 打印train_loader和test_loader的信息
        logger.info(f"批数量: {len(all_loader)}")

        return all_loader, all_data, all_labels

    def evaluate(self):
        # 数据
        all_loader, all_data, all_labels = self.loaddateset()
        x, y = next(iter(all_loader))
        logger.info(f"data shape: {x.shape}")
        # 模型
        model = PatchEncoder(in_channels=self.in_channels, use_revin=True).to(self.device)
        
        logger.info(f"模型初始化完成，加载模型参数...")
        model.load_state_dict(torch.load("best_trained_encoder.pth", map_location=self.device))

        logger.info(f"模型参数加载完成，继续加载记忆力库...")

        logger.info(f"加载memory_bank..")
        memory_bank = torch.load("memory_bank.pth", map_location=self.device)
        logger.info(f"加载memory_bank完成, shape: {memory_bank.shape}")

        # 计算异常分数
        all_scores = calculate_anomaly_scores(model, all_loader, memory_bank, top_k=3, device=self.device)
         
        # 获取点级别的异常分数
        dist_scores = distribute_patch_scores_to_points(all_scores, patch_size=self.patch_size, num_points=len(all_labels))


        #将异常分数保存到csv中
        df = pd.DataFrame({
            'Data': all_data,          
            'True Labels': all_labels,
            'Anomaly scores': dist_scores,
        })

        file_name = os.path.splitext(self.date_file)[0] # 去掉扩展名
        output_file_path = os.path.join(self.base_dir, f"{file_name}_scores.csv")
        logger.info(f"保存异常分数到csv文件: {output_file_path}")
        df.to_csv(output_file_path, index=False)
        logger.info(f"保存异常分数完成...")





if __name__ == "__main__":
    # log
    logger = init_logger()
    # 运行参数：配置文件
    parser = ArgumentParser(description="Run PaAno Anomaly Detection")
    parser.add_argument('--config', type=str, help="配置文件.",
                        default=Path(__file__).parent.joinpath('configs', 'evlconfig.yaml'))
    args = parser.parse_args()

    logger.info(f"配置文件路径: {args.config}")

    # 从yaml获取配置参数
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"配置文件加载错误: {e}")
        exit(1)
    
    logger.info(f"配置参数: {config}")

    experiment = EvlAnomalyDetection(config)
    experiment.evaluate()
    

    


    
    

    