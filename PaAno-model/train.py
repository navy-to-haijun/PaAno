import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.model import PatchEncoder
from models.train_model import train_model
from preprocessing.data_preprocess import create_txt_patch_dataloader
from utils.LoggerConfig import init_logger
from utils.utils import create_memory_bank
import numpy as np


logger = init_logger()


class TrainAnomalyDetection:
    """Train PaAno encoder with file-wise patches from all configured data."""

    def __init__(self, config: dict, device=None):
        # 读取训练流程需要的配置项。
        self.config = config
        self.date_dir = self.config["dataset"]["date_dir"]
        self.patch_size = self.config["dataset"]["patch_size"]
        self.batch_size = self.config["model"]["batch_size"]
        self.in_channels = self.config["model"]["in_channels"]
        self.log_dir = self.config["model"]["log_dir"]
        self.num_iters = self.config["model"]["num_iters"]
        self.lr = float(self.config["model"]["lr"])
        self.base_dir = Path(__file__).parent
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loaddateset(self):
        """
        创建数据集
        """
        data_path = self.base_dir.joinpath(self.date_dir)
        train_loader, train_patches = create_txt_patch_dataloader(
            data_path,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            stride=1,
            shuffle=True,
        )
        x, y = next(iter(train_loader))
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")

        return train_loader, train_patches

    def train(self):
        train_loader, train_patches = self.loaddateset()
        # 创建模型
        model = PatchEncoder(in_channels=self.in_channels, use_revin=True).to(self.device)
        logger.info("Model initialized.")
        logger.info(model)
        # 统计模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        # 创建 TensorBoard writer
        writer = SummaryWriter(self.log_dir)

        # 训练模型
        train_model(
            model,
            train_loader,
            train_patches,
            self.device,
            num_iter=self.num_iters,
            pretext_step=self.patch_size,
            lr=self.lr,
            writer=writer,
        )

        # 导出 TorchScript 编码器，供部署或推理使用。
        model.eval()
        torch.jit.trace(
            model,
            torch.randn(1, self.in_channels, self.patch_size).to(self.device),
        ).save("./output/trained_model.pt")

        # 保存正常模式的 embedding，作为后续异常评分的参考库。
        memory_bank, indices_tensor = create_memory_bank(model, train_loader, self.device, num_cores=500)
        writer.add_embedding(memory_bank, metadata=indices_tensor, tag="memory_bank")

        t0 = time.time()
        memory_bank =  memory_bank.detach().cpu().numpy()
        np.save("./output/memory_bank.npy", memory_bank)
        logger.info(f"Memory bank shape: {memory_bank.shape}, dtype: {memory_bank.dtype}")
        logger.info("Saved memory_bank in %.3f seconds", time.time() - t0)
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Run PaAno Anomaly Detection")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default=Path(__file__).parent.joinpath("configs", "exampleconfig.yaml"),
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

    experiment = TrainAnomalyDetection(config)
    experiment.train()
