from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GroupEncoder(nn.Module):
    """Fast multi-scale statistical encoder for one explicitly defined feature group.

    The upstream feature builder already performs causal trend/daily/weekly
    decomposition. This encoder summarizes the raw lookback at two scales (recent
    24 hours and full window), avoiding a recurrent-network dependency while
    preserving temporal direction, volatility and extrema.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        summary_dim = input_dim * 8
        layers: list[nn.Module] = [
            nn.Linear(summary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(num_layers - 1, 0)):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recent = x[:, -min(24, x.size(1)) :, :]
        full_mean = x.mean(dim=1)
        full_std = x.std(dim=1, unbiased=False)
        recent_mean = recent.mean(dim=1)
        recent_std = recent.std(dim=1, unbiased=False)
        last = x[:, -1, :]
        delta = last - x[:, 0, :]
        maximum = x.amax(dim=1)
        minimum = x.amin(dim=1)
        summary = torch.cat(
            [last, recent_mean, recent_std, full_mean, full_std, delta, maximum, minimum],
            dim=-1,
        )
        return self.network(summary)


class DisentangledForecaster(nn.Module):
    """Multi-branch temporal model with explicit feature-group gates.

    Each source group is encoded independently, then fused through dynamic gates.
    The decoder produces ordered P10/P50/P90 predictions for every forecast step.
    """

    def __init__(
        self,
        group_indices: dict[str, list[int]],
        horizon: int,
        hidden_dim: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_indices = {k: list(v) for k, v in group_indices.items()}
        self.group_names = list(group_indices)
        self.horizon = horizon
        self.encoders = nn.ModuleDict(
            {
                group: GroupEncoder(len(indices), hidden_dim, num_layers, dropout)
                for group, indices in group_indices.items()
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 4), 1),
        )
        self.horizon_embedding = nn.Embedding(horizon, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        contexts = []
        for group in self.group_names:
            indices = self.group_indices[group]
            contexts.append(self.encoders[group](x[:, :, indices]))
        context_stack = torch.stack(contexts, dim=1)  # [B, G, H]
        gate_logits = self.gate(context_stack).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(context_stack * gate_weights.unsqueeze(-1), dim=1)

        horizon_ids = torch.arange(self.horizon, device=x.device)
        h_emb = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(x.size(0), -1, -1)
        fused_h = fused.unsqueeze(1).expand(-1, self.horizon, -1)
        raw = self.decoder(torch.cat([fused_h, h_emb], dim=-1))
        median = raw[..., 0]
        lower_delta = F.softplus(raw[..., 1]) + 1e-4
        upper_delta = F.softplus(raw[..., 2]) + 1e-4
        quantiles = torch.stack([median - lower_delta, median, median + upper_delta], dim=-1)
        return {
            "quantiles": quantiles,
            "gate_weights": gate_weights,
            "contexts": context_stack,
        }


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    target = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    error = target - pred
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def orthogonality_loss(contexts: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(contexts, p=2, dim=-1)
    gram = torch.bmm(normalized, normalized.transpose(1, 2))
    identity = torch.eye(gram.size(1), device=gram.device).unsqueeze(0)
    off_diag = gram - identity
    return off_diag.pow(2).mean()
