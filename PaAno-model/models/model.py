# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F 

class RevIN1d(nn.Module):
    """
    可逆实例归一化
    Args:
        num_channels (int): 通道数.
        eps (float, optional): 防止除零. Default: 1e-5.
        min_sigma (float, optional): 标准差下限. Default: 1e-5.
        affine (bool, optional): 是否学习学习缩放和平移参数. Default: False.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, min_sigma: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x):
        """
        每个样本每个通道的均值和标准差
        """
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return mu, sigma
    def norm(self, x):
        """
        归一化
        """
        self._mu, self._sigma = self._stats(x)
        x_hat = (x - self._mu) / self._sigma
        
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        
        return x_hat

    def denorm(self, x_hat):
        """
        反归一化
        """
        mu, sigma = self._mu, self._sigma
        if mu is None or sigma is None:
            raise RuntimeError("Call norm() before denorm().")
        if self.affine:
            w = self.weight if self.weight is not None else 1.0
            b = self.bias if self.bias is not None else 0.0
            x_hat = (x_hat - b) / (w + self.eps)
        return x_hat * sigma + mu
    

class PatchEncoder(nn.Module): #Simple 1D CNN with RevIN
    """
    模型
    """
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,       
                 revin_affine: bool = False,   
                 revin_eps: float = 1e-5,      
                 revin_min_sigma: float = 1e-5 
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # RevIN 
        self.revin = None
        if use_revin:
            # 可逆归一化
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        #  一维卷积块
        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, self.layers[i],
                          kernel_size=self.kss[i], stride=1, padding=self.kss[i] // 2, bias=False),
                nn.BatchNorm1d(self.layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(self.layers))
        ])

        # 池化 
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        # 投影头：MLP
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        # 分类头 MLP
        self.classification_head = nn.Linear(self.layers[-1]*2, 1)

    def forward(self, x, return_embedding=True, return_projection=False):
       
       # 可逆归一化
        if self.revin is not None:
            x = self.revin.norm(x)  
        # 一维卷积块
        for block in self.convblocks:
            x = block(x)
        # 池化
        h = self.fc_embedding(x).flatten(start_dim=1)  # (N, D)
        # 返回投影头的输出
        if return_projection:
            return self.projection_head(h)
        # 返回卷积学习到的特征
        if return_embedding:
            return h
        

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        """
        卷积学习到的向量
        """
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        """
        投影头的输出
        """
        return self.projection_head(h)

