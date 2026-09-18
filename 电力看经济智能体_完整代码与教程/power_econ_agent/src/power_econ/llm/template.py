from __future__ import annotations

from typing import Any

from .base import AnswerNarrative, DiagnosisNarrative, LLMProvider, ReportNarrative


class TemplateLLMProvider(LLMProvider):
    """Deterministic fallback that keeps every workflow available without an API key."""

    def diagnosis(self, payload: dict[str, Any]) -> DiagnosisNarrative:
        event = payload.get("event", {})
        causes = payload.get("causes", [])
        top = causes[0] if causes else None
        residual = float(event.get("residual_mw", 0.0))
        direction = "高于" if residual >= 0 else "低于"
        if top:
            evidence = "；".join(top.get("evidence", [])[:3]) or "规则证据有限"
            narrative = (
                f"该时点实测负荷{direction}期望值 {abs(residual):.1f} MW。"
                f"首要候选原因为“{top.get('label', '未确定')}”，证据包括：{evidence}。"
            )
            category = top.get("category")
            confidence = float(top.get("confidence", 0.0))
            if category == "economic":
                signal = (
                    "剔除显著天气、日历、政策和数据质量因素后，波动与经济活动方向一致；"
                    "应结合行业分项用电和后续宏观数据复核。"
                )
            elif category == "data":
                signal = "当前更像数据质量问题，不应据此推断经济趋势。"
            else:
                signal = "当前主要由非经济扰动解释，原始负荷不宜直接作为经济变化结论。"
            review = category in {"data", "economic"} or confidence < 0.72
        else:
            narrative = (
                f"该时点实测负荷{direction}期望值 {abs(residual):.1f} MW，但现有规则未形成高置信原因。"
            )
            signal = "证据不足，暂不形成经济趋势判断。"
            confidence = 0.0
            review = True
        uncertainty = (
            "本诊断来自预测残差、VAE重构、特征消融和规则知识的联合排序，"
            "不是严格因果识别；置信度受负荷口径、事件日历和行业分项数据完整性影响。"
        )
        return DiagnosisNarrative(
            economic_signal=signal,
            narrative=narrative,
            uncertainty=uncertainty,
            needs_human_review=review,
        )

    def report(self, payload: dict[str, Any]) -> ReportNarrative:
        stats = payload.get("stats", {})
        diagnoses = payload.get("diagnoses", [])
        count = int(stats.get("anomaly_count", 0))
        avg = float(stats.get("average_load_mw", 0.0))
        peak = float(stats.get("peak_load_mw", 0.0))
        change = float(stats.get("comparison_change_pct", 0.0))
        economic = [
            d for d in diagnoses if (d.get("top_causes") or [{}])[0].get("category") == "economic"
        ]
        non_economic = max(len(diagnoses) - len(economic), 0)
        executive = (
            f"报告期平均负荷 {avg:.1f} MW、峰值 {peak:.1f} MW，"
            f"与可比前期相比变化 {change:+.2f}%。系统识别 {count} 个代表性异常事件。"
        )
        if economic:
            executive += f"其中 {len(economic)} 个事件包含经济活动候选信号，均需行业数据复核。"
        else:
            executive += "主要异常可由天气、日历、政策或数据质量因素解释，未形成强经济信号。"

        observations = [
            f"平均负荷为 {avg:.1f} MW，峰值为 {peak:.1f} MW，谷值为 {float(stats.get('valley_load_mw', 0.0)):.1f} MW。",
            f"报告期电量约为 {float(stats.get('energy_mwh', 0.0)):.1f} MWh。",
            f"对比窗口负荷变化为 {change:+.2f}%。",
        ]
        assessments = []
        for d in diagnoses[:8]:
            ev = d.get("event", {})
            causes = d.get("top_causes", [])
            label = causes[0].get("label", "原因未定") if causes else "原因未定"
            assessments.append(
                f"事件 {ev.get('event_id', '')}（{ev.get('timestamp', '')}）："
                f"残差 {float(ev.get('residual_mw', 0.0)):+.1f} MW，首因候选为{label}。"
            )
        if not assessments:
            assessments = ["报告期未发现超过校准阈值的代表性异常事件。"]

        interpretations = []
        if economic:
            interpretations.append(
                "部分持续性工作时段残差与经济活动方向一致，但尚不能排除行业结构、检修和未记录事件影响。"
            )
        if non_economic:
            interpretations.append(
                f"至少 {non_economic} 个已诊断事件更适合解释为非经济扰动，应从经济趋势序列中剔除或降权。"
            )
        if not interpretations:
            interpretations.append("当前证据不足以给出明显的经济扩张或收缩判断。")

        evidence = payload.get("key_evidence", [])[:12]
        if not evidence:
            evidence = ["预测区间、VAE重构分数、特征组消融贡献和知识规则联合给出结果。"]
        return ReportNarrative(
            executive_summary=executive,
            load_observations=observations,
            anomaly_assessment=assessments,
            economic_interpretation=interpretations,
            key_evidence=evidence,
            risks_and_limitations=[
                "总负荷与经济活动之间存在相关性，但不能自动解释为因果关系。",
                "宏观数据有发布时滞；模型仅使用已发布的滞后宏观特征。",
                "若缺少行业分项用电、调度日志或设备检修记录，原因排序存在遗漏风险。",
                "预测区间与规则置信度需要在本地真实历史数据上重新校准。",
            ],
            recommendations=[
                "对高严重度事件核对调度、计量和事件日志。",
                "按行业、区域和电压等级补充细分负荷以提高经济解释力。",
                "每月滚动重训并监控误差、区间覆盖率和异常率漂移。",
            ],
            data_quality=str(payload.get("data_quality", "数据质量检查结果未提供。")),
        )

    def answer(self, payload: dict[str, Any]) -> AnswerNarrative:
        snippets = payload.get("snippets", [])
        events = payload.get("events", [])
        diagnoses = payload.get("diagnoses", [])
        parts: list[str] = []
        if events:
            latest = events[0]
            parts.append(
                f"最近记录的代表性异常发生在 {latest.get('timestamp')}，"
                f"残差为 {float(latest.get('residual_mw', 0.0)):+.1f} MW，"
                f"严重度为 {latest.get('severity', '未知')}。"
            )
        if diagnoses:
            top = diagnoses[0]
            causes = top.get("top_causes", [])
            if causes:
                parts.append(f"当前首要原因候选是“{causes[0].get('label')}”。")
            if top.get("economic_signal"):
                parts.append(str(top["economic_signal"]))
        if snippets:
            parts.append("知识库提示：" + str(snippets[0].get("text", ""))[:220])
        if not parts:
            parts.append("现有数据库中没有足够的事件或知识证据回答该问题。")
        parts.append("回答基于当前模型与已入库信息，关键决策仍需人工复核。")
        evidence = [
            f"{x.get('title', x.get('source', '知识库'))}: {str(x.get('text', ''))[:160]}"
            for x in snippets[:4]
        ]
        evidence.extend(
            f"事件 {e.get('event_id')} @ {e.get('timestamp')}" for e in events[:3]
        )
        confidence = 0.75 if (events or diagnoses) and snippets else (0.55 if events or snippets else 0.25)
        return AnswerNarrative(answer="".join(parts), evidence=evidence, confidence=confidence)
