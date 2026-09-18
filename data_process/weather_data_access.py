import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 30.2741,
    "longitude": 120.1551,
    "start_date": "2022-01-01",
    "end_date": "2026-07-10",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "shortwave_radiation",
    ],
    "timezone": "Asia/Shanghai",
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()["hourly"]
weather = pd.DataFrame(data)
weather["time"] = pd.to_datetime(weather["time"])

weather.to_csv("weather_hourly.csv", index=False, encoding="utf-8-sig")