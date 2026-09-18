from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ..config import Settings
from ..features.builder import FeatureSpec
from .baseline import seasonal_baseline_matrix


class WindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        lookback: int,
        horizon: int,
        target_start: int,
        target_end: int,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.timestamps = timestamps
        self.lookback = lookback
        self.horizon = horizon
        self.indices = np.arange(max(target_start, lookback), min(target_end, len(x) - horizon + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        t = int(self.indices[item])
        return {
            "x": torch.from_numpy(self.x[t - self.lookback : t]),
            "y": torch.from_numpy(self.y[t : t + self.horizon]),
            "target_index": t,
        }


class VAEWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        selected_indices: list[int],
        sequence_length: int,
        end_indices: np.ndarray,
    ):
        self.x = np.asarray(x[:, selected_indices], dtype=np.float32)
        self.sequence_length = sequence_length
        self.end_indices = np.asarray(end_indices, dtype=int)
        self.end_indices = self.end_indices[
            (self.end_indices >= sequence_length) & (self.end_indices <= len(self.x))
        ]

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        end = int(self.end_indices[item])
        return {
            "x": torch.from_numpy(self.x[end - self.sequence_length : end]),
            "target_index": end,
        }


@dataclass
class PreparedData:
    frame: pd.DataFrame
    spec: FeatureSpec
    x_scaled: np.ndarray
    y_scaled: np.ndarray
    seasonal_baseline: np.ndarray
    seasonal_period: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_end: int
    val_end: int
    train_dataset: WindowDataset
    val_dataset: WindowDataset
    test_dataset: WindowDataset


def prepare_data(frame: pd.DataFrame, spec: FeatureSpec, settings: Settings) -> PreparedData:
    cfg = settings.forecaster
    n = len(frame)
    min_required = cfg.lookback + cfg.horizon + 48
    if n < min_required:
        raise ValueError(f"数据量不足: {n}，至少需要 {min_required} 行")
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    if train_end <= cfg.lookback + cfg.horizon:
        raise ValueError("训练集太短，请扩大日期范围或降低 lookback/horizon")

    x = frame[spec.feature_columns].to_numpy(dtype=float)
    load = frame["load_mw"].to_numpy(dtype=float)
    all_indices = np.arange(n, dtype=int)
    seasonal_baseline = seasonal_baseline_matrix(
        load, all_indices, horizon=1, seasonal_period=cfg.seasonal_period
    ).reshape(-1)
    residual_target = (load - seasonal_baseline).reshape(-1, 1)
    feature_scaler = StandardScaler().fit(x[:train_end])
    target_scaler = StandardScaler().fit(residual_target[:train_end])
    x_scaled = np.clip(feature_scaler.transform(x), -8.0, 8.0).astype(np.float32)
    y_scaled = target_scaler.transform(residual_target).reshape(-1).astype(np.float32)
    timestamps = frame["timestamp"].astype(str).to_numpy()

    train_ds = WindowDataset(
        x_scaled,
        y_scaled,
        timestamps,
        cfg.lookback,
        cfg.horizon,
        cfg.lookback,
        train_end - cfg.horizon + 1,
    )
    val_ds = WindowDataset(
        x_scaled,
        y_scaled,
        timestamps,
        cfg.lookback,
        cfg.horizon,
        train_end,
        val_end - cfg.horizon + 1,
    )
    test_ds = WindowDataset(
        x_scaled,
        y_scaled,
        timestamps,
        cfg.lookback,
        cfg.horizon,
        val_end,
        n - cfg.horizon + 1,
    )
    if min(len(train_ds), len(val_ds), len(test_ds)) <= 0:
        raise ValueError("时间划分后至少有一个集合为空，请扩大数据范围")
    return PreparedData(
        frame=frame,
        spec=spec,
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        seasonal_baseline=seasonal_baseline,
        seasonal_period=cfg.seasonal_period,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_end=train_end,
        val_end=val_end,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
