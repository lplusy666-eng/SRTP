from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim import AdamW

from ..config import Settings
from ..features.builder import FeatureSpec
from ..schemas import TrainSummary
from ..utils import configure_torch_threads, get_device, safe_mape, seed_everything, write_json
from .baseline import seasonal_baseline_matrix
from .dataset import PreparedData, VAEWindowDataset, make_loader, prepare_data
from .forecaster import DisentangledForecaster, orthogonality_loss, quantile_loss
from .inference import InferenceEngine, _robust_scale
from .nowcast import EconomicNowcaster
from .vae import SequenceVAE, vae_loss

LOGGER = logging.getLogger(__name__)

VAE_FEATURE_CANDIDATES = [
    "load_residual",
    "load_diff_1",
    "load_rolling_std_24",
    "load_residual_mean_24",
    "temperature_2m",
    "temp_anomaly",
    "relative_humidity_2m",
    "policy_event",
    "sensor_quality",
    "missing_fraction",
]


def _inverse_target(values: np.ndarray, prepared: PreparedData) -> np.ndarray:
    return values * float(prepared.target_scaler.scale_[0]) + float(prepared.target_scaler.mean_[0])


def _run_forecaster_epoch(
    model: DisentangledForecaster,
    loader,
    device: torch.device,
    quantiles: list[float],
    disentangle_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            q_loss = quantile_loss(out["quantiles"], y, quantiles)
            d_loss = orthogonality_loss(out["contexts"])
            loss = q_loss + disentangle_weight * d_loss
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            batch_size = x.size(0)
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
    return total / max(count, 1)


def _predict_dataset(
    model: DisentangledForecaster,
    loader,
    device: torch.device,
    prepared: PreparedData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    q_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    gate_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            q_chunks.append(out["quantiles"].cpu().numpy())
            y_chunks.append(batch["y"].numpy())
            index_chunks.append(np.asarray(batch["target_index"], dtype=int))
            gate_chunks.append(out["gate_weights"].cpu().numpy())
    indices = np.concatenate(index_chunks)
    q_residual = _inverse_target(np.concatenate(q_chunks, axis=0), prepared)
    y_residual = _inverse_target(np.concatenate(y_chunks, axis=0), prepared)
    baseline = seasonal_baseline_matrix(
        prepared.frame["load_mw"].to_numpy(dtype=float),
        indices,
        q_residual.shape[1],
        prepared.seasonal_period,
    )
    q = q_residual + baseline[..., None]
    y = y_residual + baseline
    return q, y, indices, np.concatenate(gate_chunks)


def _forecast_metrics(
    q: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    frame: pd.DataFrame,
    lookback: int,
    seasonal_period: int,
) -> dict[str, float]:
    median = q[..., 1]
    metrics = {
        "mae": float(mean_absolute_error(y.reshape(-1), median.reshape(-1))),
        "rmse": float(mean_squared_error(y.reshape(-1), median.reshape(-1)) ** 0.5),
        "mape": safe_mape(y.reshape(-1), median.reshape(-1)),
        "interval_80_coverage": float(np.mean((y >= q[..., 0]) & (y <= q[..., 2]))),
        "interval_80_mean_width_mw": float(np.mean(q[..., 2] - q[..., 0])),
        "one_step_mae": float(mean_absolute_error(y[:, 0], median[:, 0])),
        "one_step_rmse": float(mean_squared_error(y[:, 0], median[:, 0]) ** 0.5),
    }
    # A transparent weekly seasonal baseline is included for research comparison.
    load = frame["load_mw"].to_numpy(dtype=float)
    baseline = np.empty_like(y)
    for i, t in enumerate(indices):
        for h in range(y.shape[1]):
            source = t + h - seasonal_period
            if source < 0 or source >= len(load):
                source = t + h - 24
            if source < 0 or source >= len(load):
                source = max(t - 1, 0)
            baseline[i, h] = load[source]
    baseline_mae = float(mean_absolute_error(y.reshape(-1), baseline.reshape(-1)))
    metrics["seasonal_naive_mae"] = baseline_mae
    metrics["mae_improvement_vs_seasonal_pct"] = (
        float((baseline_mae - metrics["mae"]) / max(baseline_mae, 1e-6) * 100.0)
    )
    return metrics


def _run_vae_epoch(
    model: SequenceVAE,
    loader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, _, _ = vae_loss(out["reconstruction"], x, out["mu"], out["logvar"], beta)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total += float(loss.detach().cpu()) * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def _vae_errors_for_indices(
    model: SequenceVAE | None,
    x_scaled: np.ndarray,
    selected_indices: list[int],
    sequence_length: int,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if model is None:
        return np.zeros(len(indices), dtype=float)
    selected = x_scaled[:, selected_indices]
    errors = np.zeros(len(indices), dtype=float)
    positions = [i for i, t in enumerate(indices) if t + 1 >= sequence_length]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            windows = np.stack(
                [selected[indices[pos] - sequence_length + 1 : indices[pos] + 1] for pos in chunk]
            ).astype(np.float32)
            x = torch.from_numpy(windows).to(device)
            mu, _ = model.encode(x)
            recon = model.decode(mu, x.size(1))
            err = torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()
            errors[np.asarray(chunk)] = err
    return errors


def _anomaly_metrics(scored: pd.DataFrame) -> dict[str, float]:
    if scored.empty or "true_label" not in scored:
        return {"evaluated_points": 0.0}
    y_true = scored["true_label"].astype(int).to_numpy()
    y_pred = scored["is_anomaly"].astype(int).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_anomaly_rate": float(y_pred.mean()),
        "true_anomaly_rate": float(y_true.mean()),
        "evaluated_points": float(len(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        scores = scored["anomaly_score"].to_numpy(dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    return metrics


class TrainingPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = get_device(settings.forecaster.device)

    def run(self, frame: pd.DataFrame, spec: FeatureSpec) -> TrainSummary:
        started = time.time()
        seed_everything(self.settings.project.seed)
        threads = configure_torch_threads()
        LOGGER.info("PyTorch CPU threads=%s", threads)
        self.settings.ensure_directories()
        model_dir = self.settings.resolve(self.settings.paths.model_dir)
        output_dir = self.settings.resolve(self.settings.paths.output_dir)
        assert model_dir is not None and output_dir is not None
        prepared = prepare_data(frame, spec, self.settings)
        cfg = self.settings.forecaster
        LOGGER.info(
            "训练特征解耦预测模型: device=%s, train=%s, val=%s, test=%s",
            self.device,
            len(prepared.train_dataset),
            len(prepared.val_dataset),
            len(prepared.test_dataset),
        )

        train_loader = make_loader(
            prepared.train_dataset, cfg.batch_size, True, cfg.num_workers
        )
        val_loader = make_loader(prepared.val_dataset, cfg.batch_size, False, cfg.num_workers)
        test_loader = make_loader(prepared.test_dataset, cfg.batch_size, False, cfg.num_workers)

        model = DisentangledForecaster(
            spec.group_indices,
            cfg.horizon,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.dropout,
        ).to(self.device)
        optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = _run_forecaster_epoch(
                model,
                train_loader,
                self.device,
                cfg.quantiles,
                cfg.disentangle_weight,
                optimizer,
                cfg.gradient_clip_norm,
            )
            val_loss = _run_forecaster_epoch(
                model,
                val_loader,
                self.device,
                cfg.quantiles,
                cfg.disentangle_weight,
                None,
                cfg.gradient_clip_norm,
            )
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            LOGGER.info("Forecaster epoch %s/%s train=%.5f val=%.5f", epoch, cfg.epochs, train_loss, val_loss)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    LOGGER.info("Forecaster early stopping at epoch %s", epoch)
                    break
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_dir / "forecaster.pt")
        joblib.dump(
            {
                "feature_scaler": prepared.feature_scaler,
                "target_scaler": prepared.target_scaler,
            },
            model_dir / "scalers.joblib",
        )
        write_json(output_dir / "forecaster_history.json", history)

        val_q, val_y, val_indices, val_gates = _predict_dataset(
            model, val_loader, self.device, prepared
        )
        test_q, test_y, test_indices, test_gates = _predict_dataset(
            model, test_loader, self.device, prepared
        )
        forecaster_metrics = _forecast_metrics(
            test_q, test_y, test_indices, prepared.frame, cfg.lookback, cfg.seasonal_period
        )
        forecaster_metrics["best_validation_loss"] = float(best_val)

        vae_model: SequenceVAE | None = None
        vae_columns = [c for c in VAE_FEATURE_CANDIDATES if c in spec.feature_columns]
        vae_indices = [spec.feature_columns.index(c) for c in vae_columns]
        vae_history: list[dict[str, float]] = []
        if self.settings.vae.enabled and vae_columns:
            vcfg = self.settings.vae
            train_ends = np.arange(vcfg.sequence_length, prepared.train_end, dtype=int)
            val_ends = np.arange(max(prepared.train_end, vcfg.sequence_length), prepared.val_end, dtype=int)
            vae_train_ds = VAEWindowDataset(
                prepared.x_scaled, vae_indices, vcfg.sequence_length, train_ends
            )
            vae_val_ds = VAEWindowDataset(
                prepared.x_scaled, vae_indices, vcfg.sequence_length, val_ends
            )
            vae_train_loader = make_loader(vae_train_ds, vcfg.batch_size, True, cfg.num_workers)
            vae_val_loader = make_loader(vae_val_ds, vcfg.batch_size, False, cfg.num_workers)
            vae_model = SequenceVAE(len(vae_indices), vcfg.hidden_dim, vcfg.latent_dim).to(self.device)
            vae_optimizer = AdamW(vae_model.parameters(), lr=vcfg.learning_rate)
            best_vae_state = copy.deepcopy(vae_model.state_dict())
            best_vae_val = float("inf")
            stale = 0
            for epoch in range(1, vcfg.epochs + 1):
                train_loss = _run_vae_epoch(
                    vae_model,
                    vae_train_loader,
                    self.device,
                    vcfg.beta,
                    vae_optimizer,
                    cfg.gradient_clip_norm,
                )
                val_loss = _run_vae_epoch(
                    vae_model,
                    vae_val_loader,
                    self.device,
                    vcfg.beta,
                    None,
                    cfg.gradient_clip_norm,
                )
                vae_history.append(
                    {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
                )
                LOGGER.info("VAE epoch %s/%s train=%.5f val=%.5f", epoch, vcfg.epochs, train_loss, val_loss)
                if val_loss < best_vae_val - 1e-6:
                    best_vae_val = val_loss
                    best_vae_state = copy.deepcopy(vae_model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= vcfg.patience:
                        LOGGER.info("VAE early stopping at epoch %s", epoch)
                        break
            vae_model.load_state_dict(best_vae_state)
            torch.save(vae_model.state_dict(), model_dir / "vae.pt")
            write_json(output_dir / "vae_history.json", vae_history)

        # Calibrate unsupervised anomaly scores on the validation period only.
        val_first = val_q[:, 0, :]
        val_observed = val_y[:, 0]
        val_half_width = np.maximum(
            (val_first[:, 2] - val_first[:, 0]) / 2.0,
            max(float(prepared.frame["load_mw"].std(ddof=0)) * 0.03, 5.0),
        )
        forecast_raw = np.abs(val_observed - val_first[:, 1]) / val_half_width
        vae_raw = _vae_errors_for_indices(
            vae_model,
            prepared.x_scaled,
            vae_indices,
            self.settings.vae.sequence_length,
            val_indices,
            self.device,
            self.settings.vae.batch_size,
        )
        f_med, f_scale = _robust_scale(forecast_raw)
        v_med, v_scale = _robust_scale(vae_raw)
        f_norm = np.maximum((forecast_raw - f_med) / f_scale, 0.0)
        v_norm = np.maximum((vae_raw - v_med) / v_scale, 0.0)
        combined = (
            self.settings.anomaly.forecast_weight * f_norm
            + self.settings.anomaly.vae_weight * v_norm
        )
        threshold = float(np.quantile(combined, self.settings.anomaly.threshold_quantile))
        threshold = max(threshold, 0.25)
        calibration = {
            "forecast_median": f_med,
            "forecast_scale": f_scale,
            "vae_median": v_med,
            "vae_scale": v_scale,
            "threshold": threshold,
            "threshold_quantile": self.settings.anomaly.threshold_quantile,
            "forecast_weight": self.settings.anomaly.forecast_weight,
            "vae_weight": self.settings.anomaly.vae_weight,
            "validation_points": int(len(val_indices)),
        }
        write_json(model_dir / "anomaly_calibration.json", calibration)

        metadata = {
            "version": "0.1.0",
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "region": self.settings.project.region,
            "timezone": self.settings.project.timezone,
            "feature_columns": spec.feature_columns,
            "groups": spec.groups,
            "group_indices": spec.group_indices,
            "forecaster": {
                "lookback": cfg.lookback,
                "horizon": cfg.horizon,
                "seasonal_period": cfg.seasonal_period,
                "target_mode": "seasonal_residual",
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "quantiles": cfg.quantiles,
            },
            "vae": {
                "enabled": bool(vae_model is not None),
                "feature_columns": vae_columns,
                "sequence_length": self.settings.vae.sequence_length,
                "hidden_dim": self.settings.vae.hidden_dim,
                "latent_dim": self.settings.vae.latent_dim,
            },
            "split": {
                "train_end": prepared.train_end,
                "val_end": prepared.val_end,
                "rows": len(prepared.frame),
            },
            "average_validation_gates": {
                name: float(val_gates[:, i].mean())
                for i, name in enumerate(spec.group_indices)
            },
        }
        write_json(model_dir / "model_metadata.json", metadata)

        # Reload from disk: this is an important acceptance test for the persisted artifacts.
        engine = InferenceEngine(self.settings, prepared.frame, spec)
        test_scored = engine.scan_dataframe(
            prepared.frame, start_index=prepared.val_end, end_index=len(prepared.frame)
        )
        test_scored.to_csv(output_dir / "test_anomaly_scores.csv", index=False)
        anomaly_metrics = _anomaly_metrics(test_scored)
        events = engine.scan(
            prepared.frame,
            start_index=prepared.val_end,
            end_index=len(prepared.frame),
            events_only=True,
        )
        write_json(output_dir / "detected_events.json", [e.model_dump(mode="json") for e in events])

        latest_forecast = engine.forecast_latest(prepared.frame)
        latest_forecast.to_csv(output_dir / "latest_forecast.csv", index=False)

        nowcast_metrics: dict[str, float] = {}
        nowcast_path: Path | None = None
        if self.settings.nowcast.enabled:
            nowcast_result = EconomicNowcaster(self.settings).train(prepared.frame)
            nowcast_metrics = nowcast_result.metrics
            nowcast_path = nowcast_result.model_path

        artifacts = {
            "forecaster": str(model_dir / "forecaster.pt"),
            "scalers": str(model_dir / "scalers.joblib"),
            "metadata": str(model_dir / "model_metadata.json"),
            "calibration": str(model_dir / "anomaly_calibration.json"),
            "test_scores": str(output_dir / "test_anomaly_scores.csv"),
            "events": str(output_dir / "detected_events.json"),
            "latest_forecast": str(output_dir / "latest_forecast.csv"),
        }
        if vae_model is not None:
            artifacts["vae"] = str(model_dir / "vae.pt")
        if nowcast_path is not None:
            artifacts["nowcaster"] = str(nowcast_path)

        summary = TrainSummary(
            forecaster_metrics=forecaster_metrics,
            anomaly_metrics=anomaly_metrics,
            nowcast_metrics=nowcast_metrics,
            artifacts=artifacts,
            metadata={
                "device": str(self.device),
                "duration_seconds": float(time.time() - started),
                "train_rows": prepared.train_end,
                "validation_rows": prepared.val_end - prepared.train_end,
                "test_rows": len(prepared.frame) - prepared.val_end,
                "detected_test_events": len(events),
            },
        )
        write_json(output_dir / "training_summary.json", summary.model_dump(mode="json"))
        self._write_model_card(summary, model_dir / "MODEL_CARD.md")
        return summary

    def _write_model_card(self, summary: TrainSummary, path: Path) -> None:
        f = summary.forecaster_metrics
        a = summary.anomaly_metrics
        lines = [
            "# 模型卡：电力看经济智能体",
            "",
            f"- 区域：{self.settings.project.region}",
            f"- 数据模式：{self.settings.data.mode}",
            f"- 预测窗口：回看 {self.settings.forecaster.lookback} 小时，预测 {self.settings.forecaster.horizon} 小时",
            f"- 测试 MAE：{f.get('mae', float('nan')):.3f} MW",
            f"- 测试 MAPE：{f.get('mape', float('nan')):.3f}%",
            f"- 80% 区间覆盖率：{f.get('interval_80_coverage', float('nan')):.3f}",
            f"- 异常 F1：{a.get('f1', float('nan')):.3f}",
            "",
            "## 适用范围",
            "用于负荷异常监测、干扰剥离、原因候选排序、报告与问答原型。",
            "",
            "## 不适用范围",
            "不能把相关性解释自动等同于因果结论；真实决策需结合调度日志、行业分项用电与人工复核。",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["TrainingPipeline"]
