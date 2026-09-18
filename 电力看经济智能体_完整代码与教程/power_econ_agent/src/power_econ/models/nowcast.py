from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings
from ..utils import safe_mape

NOWCAST_FEATURES = [
    "energy_mwh",
    "load_mean",
    "load_peak",
    "load_valley",
    "load_std",
    "load_factor",
    "peak_valley_ratio",
    "residual_mean",
    "residual_abs_mean",
    "residual_std",
    "positive_residual_share",
    "business_load_mean",
    "night_load_mean",
    "business_night_ratio",
    "temperature_mean",
    "apparent_stress_mean",
    "holiday_share",
    "sensor_quality_mean",
]


def aggregate_monthly(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    df = frame.copy()
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    df["month"] = ts.to_period("M").to_timestamp()
    df["business_mask"] = pd.to_numeric(df.get("is_business_hour", 0), errors="coerce").fillna(0)
    df["night_mask"] = ((ts.hour <= 6) | (ts.hour >= 23)).astype(int)

    rows: list[dict] = []
    for month, g in df.groupby("month", sort=True):
        load = g["load_mw"].astype(float)
        peak = float(load.max())
        mean = float(load.mean())
        business = g.loc[g["business_mask"].eq(1), "load_mw"]
        night = g.loc[g["night_mask"].eq(1), "load_mw"]
        residual = g["load_residual"].astype(float)
        row = {
            "month": month,
            "energy_mwh": float(load.sum()),
            "load_mean": mean,
            "load_peak": peak,
            "load_valley": float(load.min()),
            "load_std": float(load.std(ddof=0)),
            "load_factor": mean / max(peak, 1e-6),
            "peak_valley_ratio": peak / max(float(load.min()), 1e-6),
            "residual_mean": float(residual.mean()),
            "residual_abs_mean": float(residual.abs().mean()),
            "residual_std": float(residual.std(ddof=0)),
            "positive_residual_share": float((residual > 0).mean()),
            "business_load_mean": float(business.mean()) if len(business) else mean,
            "night_load_mean": float(night.mean()) if len(night) else mean,
            "business_night_ratio": (float(business.mean()) if len(business) else mean)
            / max(float(night.mean()) if len(night) else mean, 1e-6),
            "temperature_mean": float(g["temperature_2m"].mean()),
            "apparent_stress_mean": float(g["apparent_stress"].mean()),
            "holiday_share": float(g["is_holiday"].mean()),
            "sensor_quality_mean": float(g["sensor_quality"].mean()),
            target: float(g[target].dropna().iloc[-1]) if g[target].notna().any() else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


@dataclass
class NowcastResult:
    metrics: dict[str, float]
    monthly: pd.DataFrame
    feature_importance: dict[str, float]
    model_path: Path | None


class EconomicNowcaster:
    def __init__(self, settings: Settings):
        self.settings = settings

    def train(self, frame: pd.DataFrame) -> NowcastResult:
        cfg = self.settings.nowcast
        monthly = aggregate_monthly(frame, cfg.target).dropna(subset=[cfg.target]).copy()
        if len(monthly) < cfg.min_months:
            return NowcastResult(
                metrics={"status": 0.0, "months": float(len(monthly))},
                monthly=monthly,
                feature_importance={},
                model_path=None,
            )
        test_months = min(cfg.test_months, max(3, len(monthly) // 4))
        train = monthly.iloc[:-test_months]
        test = monthly.iloc[-test_months:]
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=cfg.alpha)),
            ]
        )
        model.fit(train[NOWCAST_FEATURES], train[cfg.target])
        pred = model.predict(test[NOWCAST_FEATURES])
        monthly["nowcast"] = model.predict(monthly[NOWCAST_FEATURES])
        metrics = {
            "mae": float(mean_absolute_error(test[cfg.target], pred)),
            "rmse": float(mean_squared_error(test[cfg.target], pred) ** 0.5),
            "mape": safe_mape(test[cfg.target].to_numpy(), pred),
            "r2": float(r2_score(test[cfg.target], pred)) if len(test) > 1 else 0.0,
            "correlation": float(np.corrcoef(test[cfg.target], pred)[0, 1]) if len(test) > 1 else 0.0,
            "months": float(len(monthly)),
        }
        actual_direction = np.sign(np.diff(test[cfg.target].to_numpy()))
        pred_direction = np.sign(np.diff(pred))
        metrics["directional_accuracy"] = (
            float(np.mean(actual_direction == pred_direction)) if len(actual_direction) else 0.0
        )
        scaler = model.named_steps["scaler"]
        coef = model.named_steps["ridge"].coef_
        standardized_effect = coef / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        importance = {
            name: float(value)
            for name, value in sorted(
                zip(NOWCAST_FEATURES, standardized_effect, strict=True), key=lambda kv: abs(kv[1]), reverse=True
            )
        }
        model_path = self.settings.resolve(self.settings.paths.model_dir / "economic_nowcaster.joblib")
        assert model_path is not None
        joblib.dump(
            {
                "model": model,
                "features": NOWCAST_FEATURES,
                "target": cfg.target,
                "importance": importance,
            },
            model_path,
        )
        output = self.settings.resolve(self.settings.paths.output_dir / "monthly_economic_signals.csv")
        assert output is not None
        monthly.to_csv(output, index=False)
        return NowcastResult(metrics, monthly, importance, model_path)
