from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..utils import write_json
from .decomposition import CausalMultiScaleDecomposer, past_group_mean


@dataclass(frozen=True)
class FeatureSpec:
    feature_columns: list[str]
    groups: dict[str, list[str]]
    label_columns: list[str]

    @property
    def group_indices(self) -> dict[str, list[int]]:
        index = {name: i for i, name in enumerate(self.feature_columns)}
        return {group: [index[c] for c in cols] for group, cols in self.groups.items()}


class FeatureBuilder:
    LABEL_COLUMNS = ["anomaly_label", "anomaly_cause"]
    MACRO_COLUMNS = [
        "industrial_yoy",
        "pmi",
        "retail_yoy",
        "power_consumption_yoy",
        "electricity_price_index",
        "renewable_share",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.decomposer = CausalMultiScaleDecomposer(trend_span=168)

    @staticmethod
    def _safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
        return a / b.abs().clip(lower=eps)

    def _add_published_macro_features(self, df: pd.DataFrame, ts: pd.DatetimeIndex) -> None:
        """Create publication-safe macro features without deleting target columns.

        The first value observed in each month is treated as that month's final macro
        reading. Forecasting features receive the configured release lag so that a
        model never reads a same-month target before publication.
        """
        lag = max(0, int(self.settings.data.macro.release_lag_months))
        month = ts.tz_localize(None).to_period("M")
        for col in self.MACRO_COLUMNS:
            values = pd.to_numeric(df[col], errors="coerce")
            monthly = pd.Series(values.to_numpy(), index=month).groupby(level=0).last().sort_index()
            published = monthly.shift(lag) if lag else monthly
            mapped = pd.Series(month.map(published), index=df.index, dtype=float)
            # Neutral defaults are only used before the first available publication.
            default = 50.0 if col == "pmi" else (0.20 if col == "renewable_share" else 0.0)
            df[f"{col}_published"] = mapped.ffill().fillna(default)

    def transform(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
        df = raw.copy().sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        if ts.tz is None:
            ts = ts.tz_localize(self.settings.project.timezone)
        else:
            ts = ts.tz_convert(self.settings.project.timezone)
        df["timestamp"] = ts
        load = pd.to_numeric(df["load_mw"], errors="coerce").interpolate().ffill().bfill()
        df["load_mw"] = load

        components = self.decomposer.transform(load, df["timestamp"])
        df = pd.concat([df, components], axis=1)

        prior = load.shift(1)
        df["load_diff_1"] = load.diff().fillna(0.0)
        df["load_rolling_mean_24"] = prior.rolling(24, min_periods=1).mean().fillna(load.iloc[0])
        df["load_rolling_std_24"] = prior.rolling(24, min_periods=4).std().fillna(0.0)
        df["load_rolling_mean_168"] = prior.rolling(168, min_periods=24).mean().fillna(df["load_rolling_mean_24"])
        df["load_rolling_std_168"] = prior.rolling(168, min_periods=24).std().fillna(df["load_rolling_std_24"])
        df["load_residual_mean_24"] = (
            df["load_residual"].shift(1).rolling(24, min_periods=4).mean().fillna(0.0)
        )
        df["load_residual_mean_168"] = (
            df["load_residual"].shift(1).rolling(168, min_periods=24).mean().fillna(0.0)
        )
        df["load_factor_24"] = self._safe_ratio(
            df["load_rolling_mean_24"], prior.rolling(24, min_periods=1).max().fillna(load.iloc[0])
        ).clip(0, 2)

        hour = ts.hour.to_numpy()
        dow = ts.dayofweek.to_numpy()
        month_number = ts.month.to_numpy()
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * (month_number - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month_number - 1) / 12)
        df["is_business_hour"] = ((hour >= 8) & (hour <= 20)).astype(int)

        temp = pd.to_numeric(df["temperature_2m"], errors="coerce").interpolate().ffill().bfill()
        df["temperature_2m"] = temp
        df["cdd"] = np.maximum(temp - 24.0, 0.0)
        df["hdd"] = np.maximum(10.0 - temp, 0.0)
        climate_key = pd.Series(month_number * 24 + hour, index=df.index)
        climatology = past_group_mean(temp, climate_key)
        climatology = climatology.fillna(temp.shift(1).expanding(min_periods=1).mean()).fillna(temp.iloc[0])
        df["temp_anomaly"] = temp - climatology
        humidity = pd.to_numeric(df["relative_humidity_2m"], errors="coerce").fillna(60.0)
        df["apparent_stress"] = df["cdd"] * (1 + humidity / 200.0) + df["hdd"]

        # A data-quality channel lets the VAE separate physical changes from sensor faults.
        df["missing_fraction"] = df[
            ["load_mw", "temperature_2m", "relative_humidity_2m", "pmi", "industrial_yoy"]
        ].isna().mean(axis=1)
        df["sensor_quality"] = pd.to_numeric(df["sensor_quality"], errors="coerce").fillna(0.0).clip(0, 1)
        self._add_published_macro_features(df, ts)

        groups = {
            "load": [
                "load_mw",
                "load_trend",
                "load_daily",
                "load_weekly",
                "load_residual",
                "load_diff_1",
                "load_rolling_mean_24",
                "load_rolling_std_24",
                "load_rolling_mean_168",
                "load_rolling_std_168",
                "load_residual_mean_24",
                "load_residual_mean_168",
                "load_factor_24",
            ],
            "weather": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "cdd",
                "hdd",
                "temp_anomaly",
                "apparent_stress",
            ],
            "calendar": [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "is_weekend",
                "is_holiday",
                "is_business_hour",
            ],
            "macro": [f"{c}_published" for c in self.MACRO_COLUMNS],
            "event": ["policy_event", "sensor_quality", "missing_fraction"],
        }
        feature_columns = [c for cols in groups.values() for c in cols]
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
        df[feature_columns] = df[feature_columns].ffill().bfill().fillna(0.0)
        spec = FeatureSpec(feature_columns, groups, self.LABEL_COLUMNS)
        return df, spec

    def run(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
        features, spec = self.transform(raw)
        data_path = self.settings.resolve(self.settings.paths.processed_dir / "features.csv")
        spec_path = self.settings.resolve(self.settings.paths.processed_dir / "feature_spec.json")
        assert data_path is not None and spec_path is not None
        features.to_csv(data_path, index=False)
        write_json(
            spec_path,
            {
                "feature_columns": spec.feature_columns,
                "groups": spec.groups,
                "group_indices": spec.group_indices,
                "label_columns": spec.label_columns,
                "macro_release_lag_months": self.settings.data.macro.release_lag_months,
            },
        )
        return features, spec

    @staticmethod
    def load_feature_spec(path: Path) -> FeatureSpec:
        import json

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return FeatureSpec(raw["feature_columns"], raw["groups"], raw.get("label_columns", []))
