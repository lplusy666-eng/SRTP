from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..utils import write_json
from .akshare_macro import AKShareMacroClient
from .calendar import build_calendar
from .csv_load import CSVLoadProvider
from .open_meteo import OpenMeteoClient
from .synthetic import SyntheticPowerEconomyGenerator
from .validation import validate_raw_data

LOGGER = logging.getLogger(__name__)

DEFAULT_NUMERIC = {
    "temperature_2m": 18.0,
    "relative_humidity_2m": 60.0,
    "precipitation": 0.0,
    "wind_speed_10m": 2.0,
    "industrial_yoy": np.nan,
    "pmi": np.nan,
    "retail_yoy": np.nan,
    "power_consumption_yoy": np.nan,
    "electricity_price_index": np.nan,
    "renewable_share": np.nan,
    "is_weekend": 0,
    "is_holiday": 0,
    "policy_event": 0,
    "sensor_quality": 1.0,
    "anomaly_label": 0,
}


def _normalize_timestamp(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("数据必须包含 timestamp 列")
    ts = pd.to_datetime(df["timestamp"], errors="raise")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(timezone)
    out = df.copy()
    out["timestamp"] = ts
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _load_weather_csv(path: Path, timezone: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"天气 CSV 不存在: {path}")
    weather = pd.read_csv(path)
    return _normalize_timestamp(weather, timezone)


def _merge_real_sources(settings: Settings) -> pd.DataFrame:
    load_path = settings.resolve(settings.data.load_csv_path)
    if load_path is None:
        raise ValueError("real/csv 模式必须在 data.load_csv_path 指定高频负荷 CSV")
    load = CSVLoadProvider(load_path, settings.project.timezone, settings.data.frequency).fetch()
    load = _normalize_timestamp(load, settings.project.timezone)
    start = load["timestamp"].min()
    end = load["timestamp"].max()
    result = load.copy()

    if settings.data.weather.provider == "open_meteo":
        cache = settings.resolve(settings.paths.raw_dir / "weather_open_meteo.csv")
        assert cache is not None
        if cache.exists() and not settings.data.refresh:
            weather = pd.read_csv(cache)
        else:
            client = OpenMeteoClient(settings.data.weather.timeout_seconds)
            # Open-Meteo uses inclusive dates; the merge below limits the result to load timestamps.
            weather = client.historical(
                settings.project.latitude,
                settings.project.longitude,
                start.date().isoformat(),
                end.date().isoformat(),
                settings.project.timezone,
                settings.data.weather.variables,
            )
            weather.to_csv(cache, index=False)
        weather = _normalize_timestamp(weather, settings.project.timezone)
        result = result.merge(weather, on="timestamp", how="left", suffixes=("", "_weather"))
    elif settings.data.weather.provider == "csv":
        weather_path = settings.resolve(settings.data.weather.csv_path)
        if weather_path is None:
            raise ValueError("weather.provider=csv 时必须设置 weather.csv_path")
        weather = _load_weather_csv(weather_path, settings.project.timezone)
        result = result.merge(weather, on="timestamp", how="left", suffixes=("", "_weather"))

    if settings.data.macro.provider == "akshare":
        cache = settings.resolve(settings.paths.raw_dir / "macro_akshare.csv")
        assert cache is not None
        if cache.exists() and not settings.data.refresh:
            macro = pd.read_csv(cache, parse_dates=["month"])
        else:
            macro = AKShareMacroClient().fetch()
            macro.to_csv(cache, index=False)
    elif settings.data.macro.provider == "csv":
        macro_path = settings.resolve(settings.data.macro.csv_path)
        if macro_path is None or not macro_path.exists():
            raise FileNotFoundError("macro.provider=csv 时必须提供存在的 macro.csv_path")
        macro = pd.read_csv(macro_path)
        if "month" not in macro.columns:
            raise ValueError("宏观 CSV 需要 month 列")
        macro["month"] = pd.to_datetime(macro["month"])
    else:
        macro = pd.DataFrame()

    if not macro.empty:
        left = result.sort_values("timestamp").copy()
        left["month_key"] = left["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
        macro = macro.sort_values("month").rename(columns={"month": "month_key"})
        result = left.merge(macro, on="month_key", how="left").drop(columns="month_key")

    event_path = settings.resolve(settings.data.event_csv_path)
    cal = build_calendar(result["timestamp"], settings.project.timezone, event_path)
    # Keep user-provided columns; use generated calendar only for missing values.
    for col in ["is_weekend", "is_holiday", "policy_event", "event_name"]:
        if col not in result.columns:
            result[col] = cal[col].to_numpy()
        else:
            result[col] = result[col].where(result[col].notna(), cal[col].to_numpy())
    return result


def _fill_defaults(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df.copy()
    if "region" not in out:
        out["region"] = settings.project.region
    if "anomaly_cause" not in out:
        out["anomaly_cause"] = "unknown"
    for col, default in DEFAULT_NUMERIC.items():
        if col not in out:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce")

    macro_cols = [
        "industrial_yoy",
        "pmi",
        "retail_yoy",
        "power_consumption_yoy",
        "electricity_price_index",
        "renewable_share",
    ]
    # Forward-only filling avoids importing future releases into earlier timestamps.
    out[macro_cols] = out[macro_cols].ffill()
    neutral = {
        "industrial_yoy": 5.0,
        "pmi": 50.0,
        "retail_yoy": 5.0,
        "power_consumption_yoy": 5.0,
        "electricity_price_index": 0.0,
        "renewable_share": 0.20,
    }
    out = out.fillna(value=neutral)

    weather_cols = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
    out[weather_cols] = out[weather_cols].interpolate(limit=6).ffill().bfill()
    out["sensor_quality"] = out["sensor_quality"].fillna(0.0).clip(0, 1)
    out["anomaly_label"] = out["anomaly_label"].fillna(0).astype(int)
    out["anomaly_cause"] = out["anomaly_cause"].fillna("unknown").astype(str)
    return out


def collect_raw_data(settings: Settings) -> pd.DataFrame:
    settings.ensure_directories()
    output_path = settings.resolve(settings.paths.raw_dir / "power_economy_raw.csv")
    validation_path = settings.resolve(settings.paths.raw_dir / "data_validation.json")
    assert output_path is not None and validation_path is not None

    if output_path.exists() and not settings.data.refresh:
        LOGGER.info("使用缓存原始数据: %s", output_path)
        df = pd.read_csv(output_path)
        df = _normalize_timestamp(df, settings.project.timezone)
        return _fill_defaults(df, settings)

    if settings.data.mode == "demo":
        df = SyntheticPowerEconomyGenerator(settings).generate()
    else:
        df = _merge_real_sources(settings)
    df = _normalize_timestamp(df, settings.project.timezone)
    df = _fill_defaults(df, settings)
    report = validate_raw_data(df, settings.data.frequency)
    write_json(validation_path, report)
    if report["status"] == "fail":
        raise ValueError(f"数据质量检查失败: {report['warnings']}")
    df.to_csv(output_path, index=False)
    LOGGER.info("原始数据已写入 %s，共 %s 行", output_path, len(df))
    return df
