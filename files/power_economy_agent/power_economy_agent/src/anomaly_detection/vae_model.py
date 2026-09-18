"""
时序变分自编码器 (VAE) 用于经济异常检测
========================================
思路：VAE 学习"正常经济波动"的分布；对每个滑窗重构，
      重构误差(+KL) 越大 → 越偏离正常模式 → 越可能是经济异常段。
输入：解耦后的多通道波动窗口 (window × n_features)。
"""
import torch
import torch.nn as nn


class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        # 编码器
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # 解码器
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 0.5):
    """重构损失(MSE) + β·KL 散度。"""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="none").sum(dim=1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (recon_loss + beta * kld).mean(), recon_loss
