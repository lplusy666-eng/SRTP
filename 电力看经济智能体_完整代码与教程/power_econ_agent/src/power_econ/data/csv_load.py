from __future__ import annotations

from pathlib import Path

import pandas as pd


class CSVLoadProvider:
    REQUIRED = {"timestamp", "load_mw"}

    def __init__(self, path: Path, timezone: str, frequency: str = "1h"):
        self.path = Path(path)
        self.timezone = timezone
        self.frequency = frequency

    def fetch(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"负荷 CSV 不存在: {self.path}")
        df = pd.read_csv(self.path)
        missing = self.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"负荷 CSV 缺少列: {sorted(missing)}")
        timestamps = pd.to_datetime(df["timestamp"], errors="raise")
        if getattr(timestamps.dt, "tz", None) is None:
            timestamps = timestamps.dt.tz_localize(
                self.timezone, ambiguous="infer", nonexistent="shift_forward"
            )
        else:
            timestamps = timestamps.dt.tz_convert(self.timezone)
        df["timestamp"] = timestamps
        df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.drop_duplicates("timestamp", keep="last")

        numeric = df.select_dtypes(include="number").columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric and c != "timestamp"]
        agg: dict[str, str] = {c: "mean" for c in numeric}
        agg.update({c: "last" for c in non_numeric})
        df = (
            df.set_index("timestamp")
            .resample(self.frequency)
            .agg(agg)
            .reset_index()
        )
        df["load_mw"] = df["load_mw"].interpolate(limit=3).ffill().bfill()
        return df
