from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from power_econ.config import load_settings
from power_econ.data import collect_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="将演示宽表拆成四类本地 CSV 输入")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output-dir", default="data/input/demo_split")
    args = parser.parse_args()

    settings = load_settings(args.config)
    frame = collect_raw_data(settings).copy()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_columns = [
        c
        for c in [
            "timestamp",
            "load_mw",
            "region",
            "sensor_quality",
            "anomaly_label",
            "anomaly_cause",
        ]
        if c in frame.columns
    ]
    frame[load_columns].to_csv(output_dir / "load.csv", index=False)

    weather_columns = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ]
    frame[weather_columns].to_csv(output_dir / "weather.csv", index=False)

    ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    month = ts.tz_localize(None).to_period("M").to_timestamp() if ts.tz is not None else ts.to_period("M").to_timestamp()
    macro_columns = [
        "industrial_yoy",
        "pmi",
        "retail_yoy",
        "power_consumption_yoy",
        "electricity_price_index",
        "renewable_share",
    ]
    macro = frame[macro_columns].copy()
    macro.insert(0, "month", month)
    macro = macro.groupby("month", as_index=False).last()
    macro.to_csv(output_dir / "macro.csv", index=False)

    event = pd.DataFrame({"date": ts.date})
    for col in ["policy_event", "event_name", "is_holiday"]:
        event[col] = frame[col].to_numpy() if col in frame.columns else ("" if col == "event_name" else 0)
    event = event.groupby("date", as_index=False).agg(
        policy_event=("policy_event", "max"),
        event_name=("event_name", lambda x: next((str(v) for v in x if str(v)), "")),
        is_holiday=("is_holiday", "max"),
    )
    event["is_workday_override"] = 0
    event = event.loc[(event["policy_event"] > 0) | (event["is_holiday"] > 0) | event["event_name"].ne("")]
    event.to_csv(output_dir / "events.csv", index=False)

    print(f"已写入：{output_dir}")
    for path in sorted(output_dir.glob("*.csv")):
        print(f"- {path.name}: {sum(1 for _ in path.open(encoding='utf-8')) - 1:,} 行")


if __name__ == "__main__":
    main()
