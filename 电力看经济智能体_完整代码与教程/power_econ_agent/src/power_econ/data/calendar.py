from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def build_calendar(
    timestamps: pd.Series | pd.DatetimeIndex,
    timezone: str,
    event_csv_path: Path | None = None,
) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.tz_convert(timezone)
    local_dates = pd.Series(ts.date, index=range(len(ts)))
    is_weekend = (ts.dayofweek >= 5).astype(int)

    holiday_dates: set = set()
    try:  # optional dependency
        import holidays

        years = sorted(set(ts.year))
        cn = holidays.country_holidays("CN", years=years)
        holiday_dates = set(cn.keys())
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("未启用 holidays 中国法定节假日库，仅保留周末与自定义事件: %s", exc)

    cal = pd.DataFrame(
        {
            "timestamp": ts,
            "is_weekend": is_weekend,
            "is_holiday": local_dates.isin(holiday_dates).astype(int).to_numpy(),
            "policy_event": 0,
            "event_name": "",
        }
    )

    if event_csv_path:
        event_csv_path = Path(event_csv_path)
        if not event_csv_path.exists():
            raise FileNotFoundError(f"事件日历不存在: {event_csv_path}")
        events = pd.read_csv(event_csv_path)
        if "date" not in events.columns:
            raise ValueError("事件日历至少需要 date 列")
        events["date"] = pd.to_datetime(events["date"]).dt.date
        events = events.drop_duplicates("date", keep="last")
        day_frame = pd.DataFrame({"date": local_dates})
        joined = day_frame.merge(events, on="date", how="left")
        if "policy_event" in joined:
            cal["policy_event"] = pd.to_numeric(joined["policy_event"], errors="coerce").fillna(0).astype(int)
        if "event_name" in joined:
            cal["event_name"] = joined["event_name"].fillna("").astype(str)
        if "is_holiday" in joined:
            override = pd.to_numeric(joined["is_holiday"], errors="coerce")
            mask = override.notna()
            cal.loc[mask, "is_holiday"] = override.loc[mask].astype(int)
        if "is_workday_override" in joined:
            workday = pd.to_numeric(joined["is_workday_override"], errors="coerce").fillna(0).astype(int)
            cal.loc[workday.eq(1), ["is_weekend", "is_holiday"]] = 0

    return cal
