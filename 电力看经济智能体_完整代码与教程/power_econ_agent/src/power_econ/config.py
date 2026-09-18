from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str = "power-economy-agent"
    timezone: str = "Asia/Shanghai"
    region: str = "杭州"
    latitude: float = 30.2741
    longitude: float = 120.1551
    seed: int = 42


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("artifacts/models")
    output_dir: Path = Path("artifacts/outputs")
    report_dir: Path = Path("artifacts/reports")
    db_path: Path = Path("artifacts/power_econ.db")
    knowledge_dir: Path = Path("knowledge")


class WeatherConfig(BaseModel):
    provider: Literal["synthetic", "open_meteo", "csv"] = "synthetic"
    csv_path: Path | None = None
    timeout_seconds: float = 60.0
    variables: list[str] = Field(
        default_factory=lambda: [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ]
    )


class MacroConfig(BaseModel):
    provider: Literal["synthetic", "akshare", "csv"] = "synthetic"
    csv_path: Path | None = None
    # Macro publications are generally delayed. Forecast features use values shifted
    # by this many months; the unshifted columns remain available as nowcast targets.
    release_lag_months: int = 1


class DataConfig(BaseModel):
    mode: Literal["demo", "real", "csv"] = "demo"
    start: str = "2023-01-01"
    end: str = "2025-01-01"
    frequency: str = "1h"
    load_csv_path: Path | None = None
    event_csv_path: Path | None = None
    refresh: bool = False
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    macro: MacroConfig = Field(default_factory=MacroConfig)
    inject_demo_anomalies: bool = True


class ForecasterConfig(BaseModel):
    lookback: int = 168
    horizon: int = 24
    seasonal_period: int = 168
    hidden_dim: int = 48
    num_layers: int = 1
    dropout: float = 0.10
    batch_size: int = 128
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    disentangle_weight: float = 0.02
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0

    @model_validator(mode="after")
    def validate_splits(self) -> ForecasterConfig:
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio 必须小于 1")
        if self.lookback < 24 or self.horizon < 1 or self.seasonal_period < 1:
            raise ValueError("lookback 至少为 24，horizon 至少为 1")
        if self.quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("当前实现固定使用 [0.1, 0.5, 0.9] 三个分位数")
        return self


class VAEConfig(BaseModel):
    enabled: bool = True
    sequence_length: int = 48
    hidden_dim: int = 32
    latent_dim: int = 8
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 1e-3
    beta: float = 1e-3
    patience: int = 3


class AnomalyConfig(BaseModel):
    forecast_weight: float = 0.65
    vae_weight: float = 0.35
    threshold_quantile: float = 0.975
    min_separation_hours: int = 6
    max_events: int = 200

    @model_validator(mode="after")
    def validate_weights(self) -> AnomalyConfig:
        total = self.forecast_weight + self.vae_weight
        if total <= 0:
            raise ValueError("异常分数权重之和必须大于 0")
        if not 0.90 <= self.threshold_quantile < 1:
            raise ValueError("threshold_quantile 建议位于 [0.90, 1.0)")
        self.forecast_weight /= total
        self.vae_weight /= total
        return self


class NowcastConfig(BaseModel):
    enabled: bool = True
    target: Literal["industrial_yoy", "pmi", "retail_yoy"] = "industrial_yoy"
    min_months: int = 18
    test_months: int = 6
    alpha: float = 3.0


class LLMConfig(BaseModel):
    provider: Literal["template", "openai"] = "template"
    model: str = "gpt-5.6-luna"
    reasoning_effort: Literal["none", "low", "medium", "high"] = "low"
    max_output_tokens: int = 3000
    temperature: float | None = None


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class Settings(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    forecaster: ForecasterConfig = Field(default_factory=ForecasterConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    nowcast: NowcastConfig = Field(default_factory=NowcastConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    config_path: Path | None = Field(default=None, exclude=True)
    root_dir: Path = Field(default=Path("."), exclude=True)

    def resolve(self, path: Path | str | None) -> Path | None:
        if path is None:
            return None
        p = Path(path)
        return p if p.is_absolute() else (self.root_dir / p).resolve()

    def ensure_directories(self) -> None:
        for p in [
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.model_dir,
            self.paths.output_dir,
            self.paths.report_dir,
            self.paths.db_path.parent,
        ]:
            resolved = self.resolve(p)
            assert resolved is not None
            resolved.mkdir(parents=True, exist_ok=True)


def _apply_env_overrides(raw: dict) -> dict:
    provider = os.getenv("POWER_ECON_LLM_PROVIDER")
    model = os.getenv("POWER_ECON_OPENAI_MODEL")
    data_mode = os.getenv("POWER_ECON_DATA_MODE")
    if provider:
        raw.setdefault("llm", {})["provider"] = provider
    if model:
        raw.setdefault("llm", {})["model"] = model
    if data_mode:
        raw.setdefault("data", {})["mode"] = data_mode
    return raw


def load_settings(config_path: str | Path = "configs/demo.yaml") -> Settings:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    root_dir = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    settings = Settings.model_validate(raw)
    settings.config_path = config_path
    settings.root_dir = root_dir.resolve()
    settings.ensure_directories()
    return settings
