"""
感知智能体 (Perception Agent) —— 对应研究目标1
==============================================
职责：多源数据理解 + 特征解耦 + 异常识别。
产出写入共享上下文：merged 数据、decoupled 信号、anomaly 事件列表。
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger
from agents.base_agent import BaseAgent
from data_acquisition.build_dataset import build_dataset
from feature_decoupling.decouple import run_decoupling
from anomaly_detection.detect import detect

log = get_logger("agent.perception")


class PerceptionAgent(BaseAgent):
    def __init__(self, ctx, cfg):
        super().__init__("感知智能体", "多源数据理解与异常识别", ctx)
        self.cfg = cfg

    def run(self, rebuild=True):
        self.ctx.log_step(self.name, "开始感知", "数据获取→解耦→异常检测")
        if rebuild or not Path(abspath(self.cfg["data"]["merged_csv"])).exists():
            build_dataset(self.cfg)
        run_decoupling(self.cfg)
        anom_df, events = detect(self.cfg)

        decoupled = pd.read_csv(abspath(self.cfg["decoupling"]["output_csv"]), parse_dates=["date"])
        merged = pd.read_csv(abspath(self.cfg["data"]["merged_csv"]), parse_dates=["date"])

        # 为每个事件抽取诊断所需特征
        enriched = []
        for e in events:
            seg = decoupled[(decoupled["date"] >= e["start"]) & (decoupled["date"] <= e["end"])]
            mseg = merged[(merged["date"] >= e["start"]) & (merged["date"] <= e["end"])]
            mean_dev = float(seg["economic_deviation"].mean())
            in_holiday = bool(mseg["is_holiday"].mean() > 0.4) if "is_holiday" in mseg else False
            near_cny = bool(mseg["days_to_cny"].abs().mean() < 20) if "days_to_cny" in mseg else False
            enriched.append({
                **e,
                "direction": "上升" if mean_dev > 0 else "下降",
                "mean_deviation": mean_dev,
                "magnitude_pct": round(mean_dev * 100, 2),  # log偏离≈百分比
                "in_holiday": in_holiday,
                "near_spring_festival": near_cny,
                "avg_pmi": round(float(mseg["pmi"].mean()), 1) if "pmi" in mseg else None,
                "avg_gdp_yoy": round(float(mseg["gdp_yoy"].mean()), 2) if "gdp_yoy" in mseg else None,
            })

        self.ctx.set("anomaly_events", enriched)
        self.ctx.set("decoupled_path", abspath(self.cfg["decoupling"]["output_csv"]))
        self.ctx.log_step(self.name, "感知完成", f"检出 {len(enriched)} 个异常事件")
        return enriched
