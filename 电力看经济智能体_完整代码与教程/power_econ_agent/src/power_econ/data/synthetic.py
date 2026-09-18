from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings


class SyntheticPowerEconomyGenerator:
    """Generate an hourly, causally structured demonstration dataset.

    The data include ordinary calendar/weather effects, a latent economic cycle,
    and labeled anomaly episodes. It is deliberately realistic enough to test the
    whole pipeline, but it must not be presented as measured grid data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.project.seed)

    def _monthly_macro(self, months: pd.PeriodIndex) -> pd.DataFrame:
        unique = pd.PeriodIndex(sorted(months.unique()), freq="M")
        n = len(unique)
        cycle = np.zeros(n)
        shocks = self.rng.normal(0, 0.55, n)
        for i in range(1, n):
            cycle[i] = 0.82 * cycle[i - 1] + shocks[i]
        trend = np.linspace(-0.25, 0.45, n)
        activity = cycle + trend
        industrial = 5.0 + 1.35 * activity + self.rng.normal(0, 0.35, n)
        pmi = 50.1 + 0.55 * activity + self.rng.normal(0, 0.18, n)
        retail = 5.8 + 1.05 * activity + self.rng.normal(0, 0.45, n)
        power_yoy = 4.9 + 1.2 * activity + self.rng.normal(0, 0.30, n)
        price_index = 1.3 + 0.7 * np.sin(np.arange(n) / 5.0) + self.rng.normal(0, 0.15, n)
        renewable_share = 0.19 + np.linspace(0, 0.07, n) + self.rng.normal(0, 0.006, n)
        return pd.DataFrame(
            {
                "month": unique,
                "economic_activity": activity,
                "industrial_yoy": industrial,
                "pmi": pmi,
                "retail_yoy": retail,
                "power_consumption_yoy": power_yoy,
                "electricity_price_index": price_index,
                "renewable_share": renewable_share.clip(0.05, 0.8),
            }
        )

    def generate(self) -> pd.DataFrame:
        cfg = self.settings.data
        tz = self.settings.project.timezone
        ts = pd.date_range(
            start=pd.Timestamp(cfg.start, tz=tz),
            end=pd.Timestamp(cfg.end, tz=tz),
            freq=cfg.frequency,
            inclusive="left",
        )
        if len(ts) < 24 * 30:
            raise ValueError("演示数据至少需要 30 天")
        n = len(ts)
        hour = ts.hour.to_numpy()
        dow = ts.dayofweek.to_numpy()
        doy = ts.dayofyear.to_numpy()
        month_period = ts.tz_localize(None).to_period("M")
        macro = self._monthly_macro(month_period)
        macro_map = macro.set_index("month")

        economic_activity = macro_map.loc[month_period, "economic_activity"].to_numpy()
        industrial_yoy = macro_map.loc[month_period, "industrial_yoy"].to_numpy()
        pmi = macro_map.loc[month_period, "pmi"].to_numpy()
        retail_yoy = macro_map.loc[month_period, "retail_yoy"].to_numpy()
        power_yoy = macro_map.loc[month_period, "power_consumption_yoy"].to_numpy()
        price_idx = macro_map.loc[month_period, "electricity_price_index"].to_numpy()
        renewable = macro_map.loc[month_period, "renewable_share"].to_numpy()

        annual_temp = 16.0 + 12.5 * np.sin(2 * np.pi * (doy - 172) / 365.25)
        daily_temp = 3.7 * np.sin(2 * np.pi * (hour - 14) / 24.0)
        temperature = annual_temp + daily_temp + self.rng.normal(0, 1.25, n)
        humidity = (67 - 0.75 * (temperature - 16) + self.rng.normal(0, 7, n)).clip(20, 100)
        rain_flag = self.rng.random(n) < (0.035 + 0.02 * (humidity > 82))
        precipitation = np.where(rain_flag, self.rng.gamma(1.5, 1.1, n), 0.0)
        wind = (2.2 + self.rng.gamma(1.4, 0.8, n)).clip(0, 16)

        is_weekend = (dow >= 5).astype(int)
        synthetic_festival = np.zeros(n, dtype=int)
        # One 3-day festival block per ~120 days, deterministic under the seed.
        block_step = max(24 * 120, 1)
        for start in range(24 * 45, n, block_step):
            synthetic_festival[start : min(start + 72, n)] = 1
        is_holiday = np.maximum(is_weekend, synthetic_festival)
        business_hour = ((hour >= 8) & (hour <= 20)).astype(float)

        daily_curve = (
            105 * np.sin(2 * np.pi * (hour - 8) / 24.0)
            + 42 * np.sin(4 * np.pi * (hour - 7) / 24.0)
            + 125 * business_hour
        )
        weekly_effect = -120 * is_weekend - 85 * synthetic_festival
        heat = np.maximum(temperature - 25.0, 0.0)
        cold = np.maximum(8.0 - temperature, 0.0)
        weather_effect = 11.5 * heat**1.25 + 8.0 * cold**1.18 + 1.3 * humidity
        economic_effect = 62 * economic_activity + 22 * business_hour * economic_activity
        renewable_effect = -155 * (renewable - np.nanmean(renewable))
        long_trend = np.linspace(0, 95, n)
        noise = self.rng.normal(0, 24, n)
        load = 1080 + long_trend + daily_curve + weekly_effect + weather_effect + economic_effect + renewable_effect + noise

        anomaly_label = np.zeros(n, dtype=int)
        anomaly_cause = np.full(n, "normal", dtype=object)
        policy_event = np.zeros(n, dtype=int)
        sensor_quality = np.ones(n, dtype=float)

        if cfg.inject_demo_anomalies:
            # Heat-wave episode: weather-related non-economic disturbance.
            summer_candidates = np.where((ts.month >= 6) & (ts.month <= 8))[0]
            if len(summer_candidates) > 96:
                s = int(summer_candidates[len(summer_candidates) // 2])
                e = min(s + 72, n)
                temperature[s:e] += 7.5
                load[s:e] += 135 + 5 * np.maximum(temperature[s:e] - 33, 0)
                anomaly_label[s:e] = 1
                anomaly_cause[s:e] = "extreme_heat"

            # Sustained industrial slowdown: economic signal concentrated in working hours.
            s = min(max(n // 3, 24 * 35), max(n - 24 * 16, 0))
            e = min(s + 24 * 10, n)
            mask = np.zeros(n, dtype=bool)
            mask[s:e] = True
            mask &= business_hour.astype(bool)
            load[mask] -= 145
            anomaly_label[mask] = 1
            anomaly_cause[mask] = "industrial_slowdown"

            # Policy/price response episode.
            s = min(max(2 * n // 3, 24 * 50), max(n - 24 * 12, 0))
            e = min(s + 24 * 5, n)
            policy_event[s:e] = 1
            load[s:e] -= 55 * business_hour[s:e]
            anomaly_label[s:e] = np.maximum(anomaly_label[s:e], 1)
            anomaly_cause[s:e] = np.where(
                anomaly_cause[s:e] == "normal", "policy_demand_response", anomaly_cause[s:e]
            )

            # A guaranteed hold-out-period slowdown makes the end-to-end demo
            # evaluable even when random sensor spikes miss the test split.
            s = min(max(int(n * 0.88), 24 * 14), max(n - 24 * 8, 0))
            e = min(s + 24 * 4, n)
            holdout_mask = np.zeros(n, dtype=bool)
            holdout_mask[s:e] = True
            holdout_mask &= business_hour.astype(bool)
            load[holdout_mask] -= 175
            anomaly_label[holdout_mask] = 1
            anomaly_cause[holdout_mask] = "industrial_slowdown"

            # Sparse sensor spikes.
            eligible = np.arange(24 * 7, max(n - 24 * 7, 24 * 7 + 1))
            if len(eligible) >= 4:
                spikes = self.rng.choice(eligible, size=min(6, len(eligible)), replace=False)
                for idx in spikes:
                    load[idx] += float(self.rng.choice([380, -330]))
                    sensor_quality[idx] = 0.0
                    anomaly_label[idx] = 1
                    anomaly_cause[idx] = "sensor_fault"

        df = pd.DataFrame(
            {
                "timestamp": ts,
                "region": self.settings.project.region,
                "load_mw": load.clip(50),
                "temperature_2m": temperature,
                "relative_humidity_2m": humidity,
                "precipitation": precipitation,
                "wind_speed_10m": wind,
                "industrial_yoy": industrial_yoy,
                "pmi": pmi,
                "retail_yoy": retail_yoy,
                "power_consumption_yoy": power_yoy,
                "electricity_price_index": price_idx,
                "renewable_share": renewable,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "policy_event": policy_event,
                "sensor_quality": sensor_quality,
                "anomaly_label": anomaly_label,
                "anomaly_cause": anomaly_cause,
            }
        )
        return df
