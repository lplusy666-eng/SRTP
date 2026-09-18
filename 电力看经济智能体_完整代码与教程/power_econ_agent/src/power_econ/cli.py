from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from .agents import PowerEconomyOrchestrator
from .config import Settings, load_settings
from .data import collect_raw_data
from .features.builder import FeatureBuilder
from .models import TrainingPipeline
from .utils import configure_logging, read_json

app = typer.Typer(
    name="power-econ",
    help="基于特征解耦与知识增强的电力看经济智能体",
    no_args_is_help=True,
)


def _settings(config: str, log_level: str = "INFO") -> Settings:
    configure_logging(log_level)
    return load_settings(config)


def _load_features(settings: Settings) -> tuple[pd.DataFrame, object]:
    data_path = settings.resolve(settings.paths.processed_dir / "features.csv")
    spec_path = settings.resolve(settings.paths.processed_dir / "feature_spec.json")
    assert data_path is not None and spec_path is not None
    if not data_path.exists() or not spec_path.exists():
        raise typer.BadParameter("缺少特征文件，请先执行 power-econ features")
    frame = pd.read_csv(data_path)
    spec = FeatureBuilder.load_feature_spec(spec_path)
    return frame, spec


@app.command()
def collect(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    refresh: bool = typer.Option(False, help="忽略缓存并重新获取/生成数据"),
) -> None:
    """获取负荷、天气、宏观与事件数据并进行质量检查。"""
    settings = _settings(config)
    if refresh:
        settings.data.refresh = True
    frame = collect_raw_data(settings)
    typer.echo(f"完成：{len(frame):,} 行，{frame['timestamp'].min()} — {frame['timestamp'].max()}")


@app.command("features")
def build_features(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """执行因果多尺度分解与多源特征构建。"""
    settings = _settings(config)
    raw = collect_raw_data(settings)
    features, spec = FeatureBuilder(settings).run(raw)
    typer.echo(f"完成：{len(features):,} 行，{len(spec.feature_columns)} 个模型特征，{len(spec.groups)} 个特征组")


@app.command()
def train(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """训练分位数预测模型、VAE 异常模型和经济指标 nowcast 模型。"""
    settings = _settings(config)
    frame, spec = _load_features(settings)
    summary = TrainingPipeline(settings).run(frame, spec)  # type: ignore[arg-type]
    typer.echo(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command()
def scan(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    start: str | None = typer.Option(None),
    end: str | None = typer.Option(None),
    limit: int = typer.Option(30, min=1, max=1000),
) -> None:
    """运行感知智能体并登记异常事件。"""
    configure_logging()
    orch = PowerEconomyOrchestrator.from_config(config)
    events = sorted(orch.monitor(start, end), key=lambda e: e.anomaly_score, reverse=True)[:limit]
    typer.echo(json.dumps([e.model_dump(mode="json") for e in events], ensure_ascii=False, indent=2))


@app.command()
def diagnose(
    event_id: str = typer.Argument(..., help="先由 scan 命令返回的事件 ID"),
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """对单个异常执行特征消融、知识图谱、RAG 与解释生成。"""
    configure_logging()
    result = PowerEconomyOrchestrator.from_config(config).diagnose_event(event_id)
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command()
def report(
    start: str = typer.Argument(..., help="开始日期/时间"),
    end: str = typer.Argument(..., help="结束日期/时间"),
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    max_diagnoses: int = typer.Option(8, min=0, max=30),
) -> None:
    """生成完整电力经济态势报告。"""
    configure_logging()
    orch = PowerEconomyOrchestrator.from_config(config)
    result = orch.generate_report(start, end, max_diagnoses)
    report_path = orch.settings.resolve(orch.settings.paths.report_dir / f"{result.report_id}.md")
    typer.echo(f"报告已生成：{report_path}")
    typer.echo(orch.report.to_markdown(result))


@app.command()
def chat(
    question: str = typer.Argument(...),
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    session_id: str = typer.Option("cli"),
) -> None:
    """与知识增强问答智能体交互。"""
    configure_logging()
    result = PowerEconomyOrchestrator.from_config(config).ask(question, session_id)
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command()
def forecast(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """输出最新多步分位数负荷预测。"""
    configure_logging()
    result = PowerEconomyOrchestrator.from_config(config).perception.latest_forecast()
    typer.echo(result.to_string(index=False))


@app.command()
def demo(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    skip_train: bool = typer.Option(False, help="已有模型时跳过训练"),
) -> None:
    """一键运行：数据→特征→训练→异常→诊断→报告→问答。"""
    settings = _settings(config)
    raw = collect_raw_data(settings)
    features, spec = FeatureBuilder(settings).run(raw)
    meta = settings.resolve(settings.paths.model_dir / "model_metadata.json")
    assert meta is not None
    if not skip_train or not meta.exists():
        summary = TrainingPipeline(settings).run(features, spec)
        typer.echo("训练完成：")
        typer.echo(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    orch = PowerEconomyOrchestrator.from_settings(settings)
    split = orch.engine.metadata.get("split", {})
    test_start_index = int(split.get("val_end", max(orch.engine.lookback, len(orch.frame) - 24 * 30)))
    test_start = str(pd.Timestamp(orch.frame["timestamp"].iloc[test_start_index]))
    test_end = str(pd.Timestamp(orch.frame["timestamp"].iloc[-1]))
    events = sorted(orch.monitor(test_start, test_end), key=lambda e: e.anomaly_score, reverse=True)
    typer.echo(f"测试期识别到 {len(events)} 个代表性异常。")
    if events:
        diagnosis_result = orch.diagnosis.diagnose(events[0])
        typer.echo("最高分事件诊断：")
        typer.echo(json.dumps(diagnosis_result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    report_start = str(pd.Timestamp(orch.frame["timestamp"].iloc[max(len(orch.frame) - 24 * 30, 0)]))
    report_result = orch.generate_report(report_start, test_end, max_diagnoses=5)
    typer.echo(f"报告编号：{report_result.report_id}")
    qa = orch.ask("最近一个月的异常主要是什么原因，能否直接说明经济走弱？", "demo")
    typer.echo(f"问答：{qa.answer}")


@app.command()
def serve(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    reload: bool | None = typer.Option(None),
) -> None:
    """启动 FastAPI 服务。"""
    settings = _settings(config)
    os.environ["POWER_ECON_CONFIG"] = str(Path(config).resolve())
    import uvicorn

    uvicorn.run(
        "power_econ.api.app:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=settings.api.reload if reload is None else reload,
    )


@app.command()
def dashboard(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """启动 Streamlit 软件原型界面。"""
    env = os.environ.copy()
    env["POWER_ECON_CONFIG"] = str(Path(config).resolve())
    module_path = Path(__file__).with_name("dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(module_path)], check=True, env=env)


@app.command("mcp")
def run_mcp(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
    transport: str = typer.Option("stdio", help="stdio 或 streamable-http"),
) -> None:
    """启动 MCP 工具服务器。"""
    os.environ["POWER_ECON_CONFIG"] = str(Path(config).resolve())
    os.environ["POWER_ECON_MCP_TRANSPORT"] = transport
    from .mcp_server import main

    main()


@app.command()
def doctor(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c"),
) -> None:
    """检查环境、配置、数据和模型是否就绪。"""
    settings = _settings(config)
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "config": str(settings.config_path),
    }
    for name, rel in {
        "raw_data": settings.paths.raw_dir / "power_economy_raw.csv",
        "features": settings.paths.processed_dir / "features.csv",
        "feature_spec": settings.paths.processed_dir / "feature_spec.json",
        "model_metadata": settings.paths.model_dir / "model_metadata.json",
        "training_summary": settings.paths.output_dir / "training_summary.json",
        "knowledge_graph": settings.paths.knowledge_dir / "domain_knowledge.yaml",
    }.items():
        path = settings.resolve(rel)
        checks[name] = {"path": str(path), "exists": bool(path and path.exists())}
    summary_path = settings.resolve(settings.paths.output_dir / "training_summary.json")
    if summary_path and summary_path.exists():
        checks["metrics"] = read_json(summary_path)
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
