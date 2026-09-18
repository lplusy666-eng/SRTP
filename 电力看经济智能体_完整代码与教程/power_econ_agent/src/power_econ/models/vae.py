from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SequenceVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.encoder(x)
        h = h[-1]
        return self.mu(h), self.logvar(h).clamp(-12, 12)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        decoder_input = z.unsqueeze(1).expand(-1, sequence_length, -1)
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, h0)
        return self.output(decoded)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return {"reconstruction": recon, "mu": mu, "logvar": logvar}


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kld, reconstruction_loss, kld
