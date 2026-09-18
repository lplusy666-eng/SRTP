from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..config import Settings
from ..features.builder import FeatureSpec
from ..schemas import AnomalyEvent, FeatureContribution
from ..utils import configure_torch_threads, get_device, read_json
from .baseline import seasonal_baseline_matrix
from .forecaster import DisentangledForecaster
from .vae import SequenceVAE

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionBatch:
    indices: np.ndarray
    quantiles_mw: np.ndarray  # [N, horizon, 3]
    gate_weights: np.ndarray  # [N, groups]


def _robust_scale(values: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    return median, max(1.4826 * mad, eps)


def _load_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        return torch.load(path, map_location=device)


class InferenceEngine:
    """Loads trained artifacts and provides forecasting, anomaly scoring and ablation explanations."""

    def __init__(
        self,
        settings: Settings,
        frame: pd.DataFrame | None = None,
        spec: FeatureSpec | None = None,
    ):
        self.settings = settings
        configure_torch_threads()
        self.device = get_device(settings.forecaster.device)
        model_dir = settings.resolve(settings.paths.model_dir)
        assert model_dir is not None
        self.model_dir = model_dir
        self.metadata = read_json(model_dir / "model_metadata.json")
        if not self.metadata:
            raise FileNotFoundError(
                f"缺少模型元数据 {model_dir / 'model_metadata.json'}，请先执行 power-econ train"
            )

        self.feature_columns: list[str] = list(self.metadata["feature_columns"])
        self.groups: dict[str, list[str]] = {
            k: list(v) for k, v in self.metadata["groups"].items()
        }
        self.group_indices: dict[str, list[int]] = {
            k: list(v) for k, v in self.metadata["group_indices"].items()
        }
        self.group_names = list(self.group_indices)
        self.spec = spec or FeatureSpec(self.feature_columns, self.groups, ["anomaly_label", "anomaly_cause"])

        scalers_path = model_dir / "scalers.joblib"
        if not scalers_path.exists():
            raise FileNotFoundError(f"缺少缩放器: {scalers_path}")
        scalers = joblib.load(scalers_path)
        self.feature_scaler = scalers["feature_scaler"]
        self.target_scaler = scalers["target_scaler"]

        cfg = self.metadata["forecaster"]
        self.lookback = int(cfg["lookback"])
        self.horizon = int(cfg["horizon"])
        self.seasonal_period = int(cfg.get("seasonal_period", 168))
        self.model = DisentangledForecaster(
            group_indices=self.group_indices,
            horizon=self.horizon,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(_load_state(model_dir / "forecaster.pt", self.device))
        self.model.eval()

        vae_meta = self.metadata.get("vae", {})
        self.vae_enabled = bool(vae_meta.get("enabled", False)) and (model_dir / "vae.pt").exists()
        self.vae_columns: list[str] = list(vae_meta.get("feature_columns", []))
        self.vae_indices = [self.feature_columns.index(c) for c in self.vae_columns]
        self.vae_sequence_length = int(vae_meta.get("sequence_length", settings.vae.sequence_length))
        self.vae: SequenceVAE | None = None
        if self.vae_enabled:
            self.vae = SequenceVAE(
                input_dim=len(self.vae_indices),
                hidden_dim=int(vae_meta["hidden_dim"]),
                latent_dim=int(vae_meta["latent_dim"]),
            ).to(self.device)
            self.vae.load_state_dict(_load_state(model_dir / "vae.pt", self.device))
            self.vae.eval()

        self.calibration = read_json(model_dir / "anomaly_calibration.json", default={}) or {}
        self.frame = self._normalize_frame(frame) if frame is not None else None

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"推理数据缺少特征列: {missing}")
        out = frame.copy().sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(out["timestamp"], errors="raise")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(self.settings.project.timezone)
        else:
            ts = ts.dt.tz_convert(self.settings.project.timezone)
        out["timestamp"] = ts
        return out

    def _get_frame(self, frame: pd.DataFrame | None) -> pd.DataFrame:
        selected = frame if frame is not None else self.frame
        if selected is None:
            path = self.settings.resolve(self.settings.paths.processed_dir / "features.csv")
            assert path is not None
            if not path.exists():
                raise FileNotFoundError(f"缺少特征数据: {path}")
            selected = pd.read_csv(path)
        return self._normalize_frame(selected)

    def _scaled(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.feature_scaler.transform(frame[self.feature_columns].to_numpy(dtype=float)), -8.0, 8.0).astype(np.float32)

    def predict_indices(
        self,
        frame: pd.DataFrame,
        indices: Iterable[int],
        batch_size: int | None = None,
    ) -> PredictionBatch:
        frame = self._normalize_frame(frame)
        indices_array = np.asarray(list(indices), dtype=int)
        if indices_array.size == 0:
            return PredictionBatch(indices_array, np.empty((0, self.horizon, 3)), np.empty((0, len(self.group_names))))
        if indices_array.min() < self.lookback or indices_array.max() > len(frame):
            raise IndexError(
                f"目标索引必须位于 [{self.lookback}, {len(frame)}]，实际范围 "
                f"[{indices_array.min()}, {indices_array.max()}]"
            )
        x_scaled = self._scaled(frame)
        batch_size = batch_size or self.settings.forecaster.batch_size
        quantile_chunks: list[np.ndarray] = []
        gate_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(indices_array), batch_size):
                chunk = indices_array[start : start + batch_size]
                windows = np.stack([x_scaled[t - self.lookback : t] for t in chunk])
                x = torch.from_numpy(windows).to(self.device)
                output = self.model(x)
                quantile_chunks.append(output["quantiles"].cpu().numpy())
                gate_chunks.append(output["gate_weights"].cpu().numpy())
        q_scaled = np.concatenate(quantile_chunks, axis=0)
        scale = float(self.target_scaler.scale_[0])
        mean = float(self.target_scaler.mean_[0])
        q_residual = q_scaled * scale + mean
        baseline = seasonal_baseline_matrix(
            frame["load_mw"].to_numpy(dtype=float),
            indices_array,
            self.horizon,
            self.seasonal_period,
        )
        q_mw = q_residual + baseline[..., None]
        return PredictionBatch(
            indices=indices_array,
            quantiles_mw=q_mw,
            gate_weights=np.concatenate(gate_chunks, axis=0),
        )

    def vae_errors(
        self,
        frame: pd.DataFrame,
        indices: Iterable[int],
        batch_size: int | None = None,
    ) -> np.ndarray:
        indices_array = np.asarray(list(indices), dtype=int)
        if not self.vae_enabled or self.vae is None:
            return np.zeros(len(indices_array), dtype=float)
        frame = self._normalize_frame(frame)
        x_scaled = self._scaled(frame)[:, self.vae_indices]
        seq = self.vae_sequence_length
        batch_size = batch_size or self.settings.vae.batch_size
        errors = np.zeros(len(indices_array), dtype=float)
        valid_positions = [i for i, t in enumerate(indices_array) if t + 1 >= seq and t < len(frame)]
        with torch.inference_mode():
            for start in range(0, len(valid_positions), batch_size):
                pos_chunk = valid_positions[start : start + batch_size]
                windows = np.stack(
                    [x_scaled[indices_array[pos] - seq + 1 : indices_array[pos] + 1] for pos in pos_chunk]
                )
                x = torch.from_numpy(windows.astype(np.float32)).to(self.device)
                mu, _ = self.vae.encode(x)
                reconstruction = self.vae.decode(mu, x.size(1))
                batch_error = torch.mean((reconstruction - x) ** 2, dim=(1, 2)).cpu().numpy()
                errors[np.asarray(pos_chunk, dtype=int)] = batch_error
        return errors

    def _normalize_scores(self, forecast_raw: np.ndarray, vae_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f_med = float(self.calibration.get("forecast_median", 0.0))
        f_scale = max(float(self.calibration.get("forecast_scale", 1.0)), 1e-6)
        v_med = float(self.calibration.get("vae_median", 0.0))
        v_scale = max(float(self.calibration.get("vae_scale", 1.0)), 1e-6)
        forecast_score = np.maximum((forecast_raw - f_med) / f_scale, 0.0)
        vae_score = np.maximum((vae_raw - v_med) / v_scale, 0.0)
        cfg = self.settings.anomaly
        combined = cfg.forecast_weight * forecast_score + cfg.vae_weight * vae_score
        return forecast_score, vae_score, combined

    @staticmethod
    def _severity(score: float, threshold: float) -> str:
        ratio = score / max(threshold, 1e-6)
        if ratio >= 2.5:
            return "critical"
        if ratio >= 1.75:
            return "high"
        if ratio >= 1.25:
            return "medium"
        return "low"

    def scan_dataframe(
        self,
        frame: pd.DataFrame | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> pd.DataFrame:
        frame = self._get_frame(frame)
        start = max(self.lookback, start_index if start_index is not None else self.lookback)
        end = min(len(frame), end_index if end_index is not None else len(frame))
        if end <= start:
            return pd.DataFrame()
        indices = np.arange(start, end, dtype=int)
        predictions = self.predict_indices(frame, indices)
        first_q = predictions.quantiles_mw[:, 0, :]
        observed = frame.iloc[indices]["load_mw"].to_numpy(dtype=float)
        lower, median, upper = first_q[:, 0], first_q[:, 1], first_q[:, 2]
        half_width = np.maximum((upper - lower) / 2.0, max(float(frame["load_mw"].std(ddof=0)) * 0.03, 5.0))
        forecast_raw = np.abs(observed - median) / half_width
        vae_raw = self.vae_errors(frame, indices)
        forecast_score, vae_score, combined = self._normalize_scores(forecast_raw, vae_raw)
        threshold = float(self.calibration.get("threshold", 3.0))

        result = pd.DataFrame(
            {
                "target_index": indices,
                "timestamp": frame.iloc[indices]["timestamp"].to_numpy(),
                "observed_load_mw": observed,
                "expected_load_mw": median,
                "lower_mw": lower,
                "upper_mw": upper,
                "residual_mw": observed - median,
                "forecast_raw": forecast_raw,
                "vae_raw": vae_raw,
                "forecast_score": forecast_score,
                "vae_score": vae_score,
                "anomaly_score": combined,
                "threshold": threshold,
                "is_anomaly": combined >= threshold,
            }
        )
        for i, group in enumerate(self.group_names):
            result[f"gate_{group}"] = predictions.gate_weights[:, i]
        if "anomaly_label" in frame.columns:
            result["true_label"] = frame.iloc[indices]["anomaly_label"].to_numpy(dtype=int)
        if "anomaly_cause" in frame.columns:
            result["true_cause"] = frame.iloc[indices]["anomaly_cause"].astype(str).to_numpy()
        return result

    def _row_to_event(self, row: pd.Series) -> AnomalyEvent:
        timestamp = pd.Timestamp(row["timestamp"])
        event_id = hashlib.sha1(
            f"{self.settings.project.region}|{timestamp.isoformat()}".encode()
        ).hexdigest()[:16]
        gates = {g: float(row[f"gate_{g}"]) for g in self.group_names}
        true_label = int(row["true_label"]) if "true_label" in row and pd.notna(row["true_label"]) else None
        true_cause = str(row["true_cause"]) if "true_cause" in row and pd.notna(row["true_cause"]) else None
        return AnomalyEvent(
            event_id=event_id,
            timestamp=timestamp.to_pydatetime(),
            region=self.settings.project.region,
            observed_load_mw=float(row["observed_load_mw"]),
            expected_load_mw=float(row["expected_load_mw"]),
            lower_mw=float(row["lower_mw"]),
            upper_mw=float(row["upper_mw"]),
            residual_mw=float(row["residual_mw"]),
            forecast_score=float(row["forecast_score"]),
            vae_score=float(row["vae_score"]),
            anomaly_score=float(row["anomaly_score"]),
            threshold=float(row["threshold"]),
            severity=self._severity(float(row["anomaly_score"]), float(row["threshold"])),
            is_anomaly=bool(row["is_anomaly"]),
            gate_weights=gates,
            true_label=true_label,
            true_cause=true_cause,
        )

    def scan(
        self,
        frame: pd.DataFrame | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        events_only: bool = True,
    ) -> list[AnomalyEvent]:
        scored = self.scan_dataframe(frame, start_index, end_index)
        if scored.empty:
            return []
        if not events_only:
            return [self._row_to_event(row) for _, row in scored.iterrows()]
        candidates = scored.loc[scored["is_anomaly"]].copy()
        if candidates.empty:
            return []

        # Keep the strongest representative in each temporal neighbourhood.
        selected_rows: list[pd.Series] = []
        blocked: list[pd.Timestamp] = []
        separation = pd.Timedelta(hours=self.settings.anomaly.min_separation_hours)
        for _, row in candidates.sort_values("anomaly_score", ascending=False).iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if any(abs(ts - kept) < separation for kept in blocked):
                continue
            selected_rows.append(row)
            blocked.append(ts)
            if len(selected_rows) >= self.settings.anomaly.max_events:
                break
        selected_rows.sort(key=lambda r: pd.Timestamp(r["timestamp"]))
        return [self._row_to_event(row) for row in selected_rows]

    def explain_index(
        self,
        index: int,
        frame: pd.DataFrame | None = None,
    ) -> list[FeatureContribution]:
        frame = self._get_frame(frame)
        if not self.lookback <= index < len(frame):
            raise IndexError(f"解释索引必须在 [{self.lookback}, {len(frame) - 1}]")
        x_scaled = self._scaled(frame)
        window = x_scaled[index - self.lookback : index]
        x = torch.from_numpy(window[None, ...]).to(self.device)
        with torch.inference_mode():
            baseline_scaled = float(self.model(x)["quantiles"][0, 0, 1].cpu())
            effects: list[tuple[str, float]] = []
            for group, columns in self.group_indices.items():
                ablated = x.clone()
                ablated[:, :, columns] = 0.0  # zero is the training mean after StandardScaler
                ablated_scaled = float(self.model(ablated)["quantiles"][0, 0, 1].cpu())
                effect_mw = (baseline_scaled - ablated_scaled) * float(self.target_scaler.scale_[0])
                effects.append((group, effect_mw))
        total = sum(abs(v) for _, v in effects) or 1.0
        return [
            FeatureContribution(
                group=group,
                contribution_mw=float(value),
                normalized_importance=float(abs(value) / total),
            )
            for group, value in sorted(effects, key=lambda item: abs(item[1]), reverse=True)
        ]

    def forecast_latest(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = self._get_frame(frame)
        if len(frame) < self.lookback:
            raise ValueError("历史数据长度小于 lookback")
        prediction = self.predict_indices(frame, [len(frame)])
        q = prediction.quantiles_mw[0]
        last = pd.Timestamp(frame["timestamp"].iloc[-1])
        try:
            offset = pd.tseries.frequencies.to_offset(self.settings.data.frequency)
        except ValueError:
            offset = pd.Timedelta(hours=1)
        timestamps = [last + offset * (i + 1) for i in range(self.horizon)]
        result = pd.DataFrame(
            {
                "timestamp": timestamps,
                "lower_mw": q[:, 0],
                "median_mw": q[:, 1],
                "upper_mw": q[:, 2],
            }
        )
        for i, group in enumerate(self.group_names):
            result[f"gate_{group}"] = float(prediction.gate_weights[0, i])
        return result


__all__ = ["InferenceEngine", "PredictionBatch", "_robust_scale"]
