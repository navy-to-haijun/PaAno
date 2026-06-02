import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rknnlite.api import RKNNLite
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from utils.LoggerConfig import init_logger
import yaml
import csv

logger = init_logger()

# # 初始化
# rknn_lite = RKNNLite()
# # 加载RKNN模型
# print(f'加载RKNN模型')
# ret = rknn_lite.load_rknn('../output/rk3588-PaAno.rknn')
# if ret != 0:
#     print('加载RKNN模型失败')
#     exit(ret)
# # 初始化Runtime环境
# print(f'初始化Runtime环境')
# ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
# if ret != 0:
#     print('初始化Runtime环境失败')
#     exit(ret)
# # 推理
# print(f'开始推理')
# data = np.random.randn(10, 1, 256).astype(np.float32)
# outputs = rknn_lite.inference([data])
# print(f"{outputs}")

class RKNNPaAnoInference:
    def __init__(self, config: dict):
        # 加载参数
        self.config = config
        self.date_file = self.config["dataset"]["date_file"]
        self.patch_size = self.config["params"]["patch_size"]
        self.stride = self.config["params"]["stride"]
        self.top_k = self.config["params"]["top_k"]

        self.memory_bank_path = self.config["model"]["memory_bank_path"]
        self.model_path = self.config["model"]["rknn_model_path"]
        self.score_dir = self.config["output"]["score_dir"]

        self.base_dir = Path(__file__).parent.parent
        self.data_path = self.base_dir.joinpath(self.date_file)
        self.model_path = self.base_dir.joinpath(self.model_path)
        self.memory_bank_path = self.base_dir.joinpath(self.memory_bank_path)
        self.score_dir = self.base_dir.joinpath(self.score_dir)

        # 打印下路径
        logger.info(f"date_file: {self.data_path}")
        logger.info(f"rknn_model_path: {self.model_path}")
        logger.info(f"memory_bank_path: {self.memory_bank_path}")
        logger.info(f"score_dir: {self.score_dir}")

        # 创建输出目录
        self.score_dir.mkdir(parents=True, exist_ok=True)

        # 加载memory bank
        logger.info(f'加载memory bank')
        # 归一化memory_bank
        memory_bank = np.load(self.memory_bank_path)
        mb_norm = np.linalg.norm(memory_bank, axis=1, keepdims=True)
        mb_norm = np.where(mb_norm == 0, 1e-12, mb_norm)
        mb_normalized = memory_bank / mb_norm
        logger.info(f'memory bank shape: {mb_normalized.shape}')
        # 清除NaN和Inf值
        self.mb_normalized = np.nan_to_num(mb_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        # 加载RKNN模型
        self.rknn_lite = self.load_rknn_model()

        
    
    def load_rknn_model(self, use_verbose=False, core_mask=RKNNLite.NPU_CORE_0_1_2):
        """
        加载RKNN模型
        """
        rknn_lite = RKNNLite()
        logger.info(f'加载RKNN模型')
        ret = rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            logger.error('加载RKNN模型失败')
            exit(ret)
            
        logger.info(f'初始化Runtime环境')
        ret = rknn_lite.init_runtime(core_mask=core_mask)
        if ret != 0:
            logger.error('初始化Runtime环境失败')
            exit(ret)
        return rknn_lite
    def load_txt_data(self, txt_path):
        """
        加载txt离线数据
        """
        data = np.loadtxt(txt_path, dtype=np.float32)
        return data
    def preprocess_to_patches(self, data, patch_size, stride):
        """
        将数据预处理成patches
        Args:
            data (np.ndarray): 输入数据
            patch_size (int): patch大小
            stride (int): 步长
        Returns:
            patches (np.ndarray): 预处理后的patches
        """
        patches = []
        for i in range(0, len(data) - patch_size + 1, stride):
            patch = data[i:i + patch_size]
            patches.append(patch)
        # 数组
        patches_array = np.array(patches)                      # (N, L)
        # 转换为模型输入格式 (N, 1, L)
        patches_array = np.expand_dims(patches_array, axis=1)
        return patches_array

    def preprocess_to_batchs(self, batch_size=1, patches=None):
        """
        将patches预处理成batchs
        Args:
            patches (np.ndarray): 输入patches
            batch_size (int): batch大小
        Returns:
            batchs (np.ndarray): 预处理后的batchs
        """
        batchs = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batchs.append(batch)
        return batchs
    def rknn_inference(self, rknn_lite, data):
        """
        使用RKNN模型进行推理
        Args:
            rknn_lite (RKNNLite): RKNNLite对象
            data (np.ndarray): 输入数据
        Returns:
            outputs (list): 推理结果
        """
        outputs = []
         # 切片
        patches = self.preprocess_to_patches(data, patch_size=self.patch_size, stride=self.stride)
        # 批处理
        batchs = self.preprocess_to_batchs(batch_size=1, patches=patches)
        # 推理
        for batch in batchs:
            # logger.info(f"推理输入形状: {batch.shape}")
            output = rknn_lite.inference([batch])
            outputs.append(output)
        # 转数组
        outputs = np.array(outputs)
        return outputs
    def calculate_patch_scores(self, outputs, memory_bank, top_k):
        """
        计算patch的异常分数
        Args:
            outputs (np.ndarray): 模型输出
            memory_bank (np.ndarray): memory bank(已经归一化)
        Returns:
            patch_scores (np.ndarray): patch的异常分数
        """
        # 将结果由[B,1, 1,F]转换为[B, F]
        outputs = outputs.squeeze(1).squeeze(1)
        # 归一化推理结果
        output_norm = np.linalg.norm(outputs, axis=1, keepdims=True)
        output_norm = np.where(output_norm == 0, 1e-12, output_norm)
        outputs_normalized = outputs / output_norm
        # 清除NaN和Inf值
        outputs_normalized = np.nan_to_num(outputs_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        # 计算余弦相似度
        # [B, F] * [M, F].T -> [B, M]
        similarity = np.dot(outputs_normalized, memory_bank.T)
        # 防止异常值
        similarity = np.nan_to_num(similarity, nan=-1.0, posinf=1.0, neginf=-1.0)
        # 取top-k相似度(最大的k个相似度)
        top_k_similarities = np.partition(similarity, -top_k, axis=1)[:, -top_k:]
        # 计算异常分数（1 - 平均相似度）
        dists = 1.0 - top_k_similarities
        patch_scores = np.mean(dists, axis=1)
        patch_scores = np.nan_to_num(patch_scores, nan=1.0, posinf=1.0, neginf=0.0)
        return patch_scores
    def distribute_patch_scores_to_points(self, patch_scores: np.ndarray, patch_size: int): 
        """
        转化为点分数
        Args:
            patch_scores (np.ndarray): patch的异常分数
            patch_size (int): patch大小
        """
        num_points = len(patch_scores) + patch_size - 1

        kernel = np.ones(patch_size, dtype=np.float32)
        sums   = np.convolve(patch_scores, kernel, mode='full')[:num_points]
        counts = np.convolve(np.ones_like(patch_scores), kernel, mode='full')[:num_points]

        point_scores = np.divide(
            sums, counts,
            out=np.zeros(num_points, dtype=np.float32),
            where=counts != 0
        )
        return np.nan_to_num(point_scores, nan=0.0, posinf=0.0, neginf=0.0)



    def evaluate(self, use_livestream = False,data=None):
        """
        推理逻辑
        Args:
            use_livestream (bool): 是否使用实时数据进行推理, 默认为False, 使用txt离线数据进行推理
            data (np.ndarray): 实时数据, 仅在use_livestream为True时有效
        returns:
            point_scores (np.ndarray): 点分数, 仅在use_livestream为True时返回, 否则返回None
        """
        if use_livestream:
            logger.info(f'使用流式数据进行推理')
            if data is None:
                logger.error("实时数据未提供")
                exit(1)
            # 推理
            outputs = self.rknn_inference(self.rknn_lite, data)
            # 计算patch分数
            patch_scores = self.calculate_patch_scores(outputs, self.mb_normalized, self.top_k)
            # 计算点分数
            point_scores = self.distribute_patch_scores_to_points(patch_scores, self.patch_size)
            # 分数 归一化到0-1
            scrores_min = 0
            scrores_max = 0.05
            point_scores = (point_scores - scrores_min) / (scrores_max - scrores_min)
            return point_scores
            
        else:
            logger.info(f'使用txt离线数据进行推理')
            for txt_path in sorted(self.data_path.glob("*.txt")):
                # 加载数据
                logger.info(f"推理txt文件: {txt_path}")
                data = self.load_txt_data(txt_path)
                # 推理
                outputs = self.rknn_inference(self.rknn_lite, data)
                logger.info(f"推理结果: {outputs.shape}")
                # 计算patch分数
                patch_scores = self.calculate_patch_scores(outputs, self.mb_normalized, self.top_k)
                # 计算点分数
                point_scores = self.distribute_patch_scores_to_points(patch_scores, self.patch_size)
                # 保存分数
                output_path = self.score_dir / f"{txt_path.stem}_scores_rk3588.csv"
                with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "value", "anomaly_score"])
                    for idx, (value, score) in enumerate(zip(data, point_scores)):
                        writer.writerow([idx, float(value), float(score)])

                logger.info(f"Saved anomaly scores: {output_path}")
            
            return None



if __name__ == "__main__":
    parser = ArgumentParser(description="Run PaAno RKNN inference.")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default=Path(__file__).parent.joinpath("evlconfig.yaml"),
    )
    args = parser.parse_args()
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise SystemExit(1)

    logger.info(f"Config: {config}")
    experiment = RKNNPaAnoInference(config)
    experiment.evaluate(use_livestream=False)


