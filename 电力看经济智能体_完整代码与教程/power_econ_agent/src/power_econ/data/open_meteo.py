from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import httpx
import pandas as pd

LOGGER = logging.getLogger(__name__)


class OpenMeteoClient:
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout = timeout_seconds

    def _request(self, url: str, params: dict, retries: int = 3) -> dict:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                LOGGER.warning("Open-Meteo 请求失败，第 %s/%s 次: %s", attempt + 1, retries, exc)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Open-Meteo 请求失败: {last_error}")

    @staticmethod
    def _to_frame(payload: dict) -> pd.DataFrame:
        hourly = payload.get("hourly") or {}
        if "time" not in hourly:
            raise ValueError(f"Open-Meteo 返回中没有 hourly.time: {payload}")
        df = pd.DataFrame(hourly)
        df["timestamp"] = pd.to_datetime(df.pop("time"), errors="raise")
        return df

    def historical(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: str,
        variables: Iterable[str],
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": timezone,
        }
        return self._to_frame(self._request(self.HISTORICAL_URL, params))

    def forecast(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        variables: Iterable[str],
        forecast_days: int = 7,
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(variables),
            "timezone": timezone,
            "forecast_days": forecast_days,
        }
        return self._to_frame(self._request(self.FORECAST_URL, params))
