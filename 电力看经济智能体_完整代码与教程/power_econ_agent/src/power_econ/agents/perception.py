from __future__ import annotations

import pandas as pd

from ..models.inference import InferenceEngine
from ..schemas import AnomalyEvent
from ..storage import EventStore


class PerceptionAgent:
    """Perception agent: forecasting, anomaly fusion and persistent event registration."""

    def __init__(self, engine: InferenceEngine, frame: pd.DataFrame, store: EventStore):
        self.engine = engine
        self.frame = frame.sort_values("timestamp").reset_index(drop=True)
        self.store = store

    def _period_indices(self, start: str | None, end: str | None) -> tuple[int | None, int | None]:
        timestamps = pd.DatetimeIndex(pd.to_datetime(self.frame["timestamp"]))
        start_index = None
        end_index = None
        if start:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None and timestamps.tz is not None:
                start_ts = start_ts.tz_localize(timestamps.tz)
            elif timestamps.tz is not None:
                start_ts = start_ts.tz_convert(timestamps.tz)
            start_index = int(timestamps.searchsorted(start_ts, side="left"))
        if end:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None and timestamps.tz is not None:
                end_ts = end_ts.tz_localize(timestamps.tz)
            elif timestamps.tz is not None:
                end_ts = end_ts.tz_convert(timestamps.tz)
            end_index = int(timestamps.searchsorted(end_ts, side="right"))
        return start_index, end_index

    def monitor(self, start: str | None = None, end: str | None = None) -> list[AnomalyEvent]:
        start_index, end_index = self._period_indices(start, end)
        events = self.engine.scan(
            self.frame,
            start_index=start_index,
            end_index=end_index,
            events_only=True,
        )
        for event in events:
            payload = event.model_dump(mode="json")
            self.store.upsert_anomaly(event.event_id, str(event.timestamp), payload)
        return events

    def scores(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start_index, end_index = self._period_indices(start, end)
        return self.engine.scan_dataframe(self.frame, start_index=start_index, end_index=end_index)

    def latest_forecast(self) -> pd.DataFrame:
        return self.engine.forecast_latest(self.frame)
