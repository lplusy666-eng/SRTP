from __future__ import annotations

import numpy as np
import pandas as pd


def past_group_mean(values: pd.Series, keys: pd.Series) -> pd.Series:
    """Mean of previous observations in the same group, excluding the current row."""
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "key": keys})
    group = frame.groupby("key", sort=False)["value"]
    previous_sum = group.cumsum() - frame["value"]
    previous_count = frame.groupby("key", sort=False).cumcount()
    result = previous_sum / previous_count.replace(0, np.nan)
    result.index = values.index
    return result


class CausalMultiScaleDecomposer:
    """Past-only decomposition into trend, daily, weekly, and irregular components."""

    def __init__(self, trend_span: int = 168):
        self.trend_span = trend_span

    def transform(self, load: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
        load = pd.to_numeric(load, errors="coerce").astype(float)
        ts = pd.DatetimeIndex(timestamps)
        prior = load.shift(1)
        trend = prior.ewm(span=self.trend_span, adjust=False, min_periods=24).mean()
        expanding_prior = prior.expanding(min_periods=1).mean()
        trend = trend.fillna(expanding_prior).fillna(load.iloc[0])

        detrended = load - trend
        hour = pd.Series(ts.hour, index=load.index)
        daily = past_group_mean(detrended, hour)
        daily = daily.fillna(detrended.shift(1).expanding(min_periods=1).mean()).fillna(0.0)

        hour_of_week = pd.Series(ts.dayofweek * 24 + ts.hour, index=load.index)
        weekly_remainder = detrended - daily
        weekly = past_group_mean(weekly_remainder, hour_of_week)
        weekly = weekly.fillna(0.0)

        residual = load - trend - daily - weekly
        return pd.DataFrame(
            {
                "load_trend": trend,
                "load_daily": daily,
                "load_weekly": weekly,
                "load_residual": residual,
            },
            index=load.index,
        )
