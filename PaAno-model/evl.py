from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from preprocessing.data_preprocess import preprocess_to_patches
from utils.LoggerConfig import init_logger
from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points


logger = init_logger()


class EvlAnomalyDetection:
    """
    PaAno 离线推理。

    按 evlconfig.yaml 指定的 txt 目录逐个文件推理，计算点级异常分数，
    并为每个 txt 保存一个对应的 csv 结果文件。
    """

    def __init__(self, config: dict, device=None):
        # 读取推理流程需要的配置项，保持和 train.py 一致的写法。
        self.config = config
        self.date_file = self.config["dataset"]["date_file"]
        self.patch_size = self.config["params"]["patch_size"]
        self.stride = self.config["params"]["stride"]
        self.in_channels = self.config["params"]["in_channels"]
        self.top_k = self.config["params"]["top_k"]
        self.model_path = self.config["model"]["model_path"]
        self.memory_bank_path = self.config["model"]["memory_bank_path"]
        self.score_dir = self.config["output"]["score_dir"]
        self.base_dir = Path(__file__).parent
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = self.base_dir.joinpath(self.date_file)
        self.model_path = self.base_dir.joinpath(self.model_path)
        self.memory_bank_path = self.base_dir.joinpath(self.memory_bank_path)
        self.score_dir = self.base_dir.joinpath(self.score_dir)

        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Memory bank path: {self.memory_bank_path}")
        logger.info(f"Score dir: {self.score_dir}")

    def evaluate(self):
        """
        执行推理并保存异常分数。
        """
        # 加载模型
        logger.info(f"Loading TorchScript model: {self.model_path}")
        with open(self.model_path, "rb") as f:
            model = torch.jit.load(f, map_location=self.device)
        model.eval()

        # 加载memory_bank
        logger.info(f"Loading memory bank: {self.memory_bank_path}")
        memory_bank = np.load(self.memory_bank_path)
        memory_bank = torch.as_tensor(memory_bank, dtype=torch.float32, device=self.device)
        
        # 创建结果目录
        self.score_dir.mkdir(parents=True, exist_ok=True)
        # 循环推理所有txt文件
        for txt_path in sorted(self.data_path.glob("*.txt")):
            logger.info(f"txt file: {txt_path}")
            # 加载txt文件
            data = np.loadtxt(txt_path, dtype=np.float32).squeeze()
            # 切片
            patches = preprocess_to_patches(data, patch_size=self.patch_size, stride=self.stride)
            logger.info(f"Generated {len(patches)} patches from {len(data)} data points.")

            # 计算异常分数
            patch_scores = calculate_anomaly_scores(
                model,
                memory_bank,
                patches,
                device=self.device,
                top_k=self.top_k,
            )
            point_scores = distribute_patch_scores_to_points(
                patch_scores,
                patch_size=self.patch_size,
                num_points=len(data),
            )

            output_path = self.score_dir / f"{txt_path.stem}_scores.csv"
            with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "value", "anomaly_score"])
                for idx, (value, score) in enumerate(zip(data, point_scores)):
                    writer.writerow([idx, float(value), float(score)])

            logger.info(f"Saved anomaly scores: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run PaAno txt inference and save anomaly scores.")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default=Path(__file__).parent.joinpath("configs", "evlconfig.yaml"),
    )
    args = parser.parse_args()

    logger.info(f"Config path: {args.config}")

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise SystemExit(1)

    logger.info(f"Config: {config}")
    experiment = EvlAnomalyDetection(config)
    experiment.evaluate()
