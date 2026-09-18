"""测试套件。

覆盖四层：
1. 契约层 —— 五段式输出必须齐全，因果强度枚举不能越界；
2. 摄入层 —— 自动发现与格式嗅探在真实两省数据上都要选对列；
3. 工具层 —— 注册表、schema 导出、调度、失败降级；
4. 循环层 —— 端到端跑通、留痕、终止条件。

这些测试刻意使用真实的 region pack 数据，而不是 mock ——
因为这一层最容易出问题的恰恰是"列名不一样"这种真实世界的脏活。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerecon.analysis import stl_decompose, yoy_at
from powerecon.agent import AgentLoop, RulePlanner, ScriptedPlanner
from powerecon.agent.planner import extract_period_from_question, synthesize_conclusion
from powerecon.contract import CausalStatus, Conclusion
from powerecon.ingest import available_packs, load_region
from powerecon.runtime import default_packs_root, make_context
from powerecon.series_utils import parse_period
from powerecon.tools import REGISTRY, anthropic_tools, dispatch, openai_tools

PACKS_ROOT = default_packs_root()


# ---------------------------------------------------------------- 契约层


def test_conclusion_requires_five_parts():
    """五段式少任何一段都应该被 pydantic 拒绝。"""
    base = dict(
        region="浙江省",
        question="测试问题",
        conclusion="测试结论",
        causal_status=CausalStatus.OBSERVED,
        confidence=0.5,
    )
    with pytest.raises(Exception):
        Conclusion(**base)  # 缺 evidence / counter_evidence / uncertainty
    ok = Conclusion(
        **base,
        evidence=[],
        counter_evidence=["反证"],
        uncertainty=["不确定性"],
    )
    assert ok.causal_status is CausalStatus.OBSERVED
    assert "反证" in ok.to_markdown()


def test_causal_status_rejects_free_text():
    """因果强度必须是枚举，堵住"相关性写成因果"的表述。"""
    with pytest.raises(Exception):
        Conclusion(
            region="浙江省",
            question="q",
            conclusion="c",
            causal_status="用电下降导致了经济走弱",
            confidence=0.5,
            evidence=[],
            counter_evidence=["x"],
            uncertainty=["y"],
        )


# ---------------------------------------------------------------- 摄入层


def test_both_packs_discoverable():
    names = available_packs(PACKS_ROOT)
    assert "zhejiang" in names
    assert "jiangsu" in names


@pytest.mark.parametrize("pack", ["zhejiang", "jiangsu"])
def test_region_loads_with_expected_datasets(pack):
    ctx = load_region(PACKS_ROOT / pack)
    assert ctx.info.code.startswith("CN-")
    assert ctx.info.caveats, "每个地区都必须声明数据边界，否则模型无法知道限制"
    assert any("electricity" in k for k in ctx.datasets)
    assert ctx.kg_relationships is not None
    assert ctx.knowledge_docs, "知识库为空会导致诊断没有依据"


def test_jiangsu_auto_discovery_picks_monthly_not_ytd():
    """江苏表里同时有当月值和年内累计值，必须选中当月值。

    这是最容易出错、也最致命的一处：选成累计值会让后续所有同比和季节分析全部失效。
    """
    ctx = load_region(PACKS_ROOT / "jiangsu")
    info = ctx.datasets["electricity_monthly"]
    assert "ytd" not in info.metric.lower()
    assert "累计" not in info.metric
    df = ctx.series("electricity_monthly")
    # 当月值不应单调递增；累计值会
    diffs = df["value"].diff().dropna()
    assert (diffs < 0).any(), "选中的列疑似是累计值（单调不减）"


def test_kg_column_aliases_normalise_both_schemas():
    """浙江用 id/type，江苏用 node_id/relationship_type，摄入后列集合必须一致。"""
    zj = load_region(PACKS_ROOT / "zhejiang")
    js = load_region(PACKS_ROOT / "jiangsu")
    assert "type" in zj.kg_relationships.columns
    assert "type" in js.kg_relationships.columns
    assert "id" in zj.kg_nodes.columns
    assert "id" in js.kg_nodes.columns
    assert js.kg_relationships["type"].notna().any()


def test_parse_period_handles_all_formats():
    assert parse_period("2021-01").month == 1
    assert parse_period("2024Q3").month == 7
    assert parse_period("2021").year == 2021
    assert parse_period("202101").month == 1
    assert parse_period("2024-01-15").day == 15
    assert parse_period("垃圾数据") is None
    assert parse_period(None) is None


# ---------------------------------------------------------------- 工具层


def test_tool_registry_and_schema_export():
    assert len(REGISTRY) >= 12
    openai = openai_tools()
    anthropic = anthropic_tools()
    assert len(openai) == len(anthropic) == len(REGISTRY)
    for spec in openai:
        fn = spec["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
    # 两个格式的 schema 必须同源
    by_name = {t["function"]["name"]: t for t in openai}
    for t in anthropic:
        assert t["input_schema"] == by_name[t["name"]]["function"]["parameters"]


def test_every_tool_description_is_substantive():
    """工具描述是模型唯一的说明书，太短说明没写清用途与边界。"""
    for name, spec in REGISTRY.items():
        assert len(spec.description) >= 40, f"{name} 的描述太短，模型看不懂该不该用它"


def test_dispatch_reports_error_instead_of_raising():
    """参数错误必须变成可读文本回给模型，而不是让整条链路崩掉。"""
    ctx = make_context("zhejiang")
    bad = dispatch("load_series", {"dataset_id": "不存在的表"}, ctx)
    assert bad.ok is False
    assert "没有数据集" in bad.text
    assert bad.duration_ms >= 0


def test_dispatch_handles_unknown_tool_and_bad_args():
    ctx = make_context("zhejiang")
    r1 = dispatch("tool_that_does_not_exist", {}, ctx)
    assert not r1.ok and "没有名为" in r1.text
    r2 = dispatch("load_series", {"wrong_param": 1}, ctx)
    assert not r2.ok and "参数校验失败" in r2.text


def test_tool_output_is_compact():
    """长列表必须被截断，否则会烧掉上下文。"""
    ctx = make_context("zhejiang")
    r = dispatch("decompose_series", {"dataset_id": "electricity_monthly", "include_points": True}, ctx)
    assert r.ok
    assert "省略" in r.text or len(r.text) <= 4200


# ---------------------------------------------------------------- 分析层


def test_stl_decompose_on_real_data():
    ctx = make_context("zhejiang")
    df = ctx.region.series("electricity_monthly")
    out = stl_decompose(df, ctx.region.datasets["electricity_monthly"].frequency)
    assert out.n == 51
    assert 0.0 <= out.trend_strength <= 1.0
    assert 0.0 <= out.seasonal_strength <= 1.0
    # 浙江月度用电季节强度应明显偏高
    assert out.seasonal_strength > 0.5
    # 分解必须可重构：趋势 + 季节 + 残差 ≈ 观测（对数尺度上）
    recon = out.trend + out.seasonal
    assert abs(float((out.values - recon).mean())) < float(out.values.mean())


def test_stl_refuses_short_series():
    """样本量不足必须报错，而不是给出不可靠的结果。"""
    import pandas as pd

    ctx = make_context("zhejiang")
    df = ctx.region.series("electricity_monthly").head(8)
    with pytest.raises(ValueError, match="不足 2 个完整季节周期"):
        stl_decompose(df, ctx.region.datasets["electricity_monthly"].frequency)


def test_yoy_at_matches_manual_calculation():
    ctx = make_context("zhejiang")
    df = ctx.region.series("electricity_monthly")
    out = stl_decompose(df, ctx.region.datasets["electricity_monthly"].frequency)
    idx = out.n - 1
    expected = (out.values[idx] - out.values[idx - 12]) / abs(out.values[idx - 12])
    assert yoy_at(out, idx) == pytest.approx(expected, rel=1e-9)
    assert yoy_at(out, 5) is None, "不足一年的期次不应给出同比"


# ---------------------------------------------------------------- 循环层


def test_extract_period_from_question():
    assert extract_period_from_question("2025年1月浙江用电下降是经济原因吗？") == "2025-01"
    assert extract_period_from_question("2024Q4 的情况如何") == "2024Q4"
    assert extract_period_from_question("2024年第三季度怎么样") == "2024Q3"
    assert extract_period_from_question("最近怎么样") is None


def test_rule_planner_end_to_end_and_trace():
    ctx = make_context("zhejiang")
    loop = AgentLoop(ctx, RulePlanner(), max_steps=12, trace_dir=Path(ctx.artifacts_dir).parent / "traces")
    result = loop.run("2025年1月浙江用电同比下降是经济原因吗？")

    assert result.ok, result.error
    assert result.conclusion is not None
    c = result.conclusion
    assert c.evidence, "必须有证据"
    assert c.counter_evidence, "必须有反证"
    assert c.uncertainty, "必须声明不确定性"
    assert 0.0 <= c.confidence <= 1.0
    assert c.causal_status is not CausalStatus.ESTABLISHED, "本项目不应产出已证因果"

    # 关键行为：问题里点明的期次必须被真正分析到
    assert "2025-01" in " ".join(e.period or "" for e in c.evidence)
    # 关键行为：不能把同比变化直接说成经济原因
    assert "不能" in c.conclusion or "不支持" in c.conclusion

    assert result.trace_path and Path(result.trace_path).exists()
    lines = [json.loads(x) for x in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]
    assert any(r.get("type") == "run_start" for r in lines)
    assert any(r.get("phase") == "act" for r in lines)
    assert any(r.get("type") == "run_end" for r in lines)


def test_scripted_planner_never_loops_forever():
    """规划器一直返回同一个工具时，循环必须靠 max_steps 收住。"""
    ctx = make_context("zhejiang")
    planner = ScriptedPlanner([("list_region_datasets", {})] * 20)
    loop = AgentLoop(ctx, planner, max_steps=5)
    result = loop.run("测试")
    assert result.n_steps <= 5
    assert result.conclusion is not None, "即使跑不完也必须给出诚实结论"


def test_synthesize_conclusion_flags_missing_context():
    """什么工具都没调时，置信度必须很低且明确说明缺什么。"""
    from powerecon.agent.planner import AgentState

    state = AgentState(question="q", region_name="浙江省")
    args = synthesize_conclusion(state)
    assert args["confidence"] <= 0.3
    assert args["uncertainty"]
    assert args["counter_evidence"]


def test_residual_negligible_is_not_reported_as_economic_signal():
    """残差接近零时，结论必须是"不构成经济信号"，不能硬凑异常叙事。"""
    ctx = make_context("zhejiang")
    r = dispatch("explain_anomaly", {"dataset_id": "electricity_monthly", "period": "2025-01"}, ctx)
    assert r.ok, r.error
    payload = r.payload.model_dump(mode="json")
    assert abs(payload["event"]["residual_pct"]) < 0.05
    assert "residual_negligible" in payload["rules_fired"]
    assert any("不存在需要额外解释的异常" in s for s in payload["exclusion_summary"])


def test_weather_context_declares_proxy():
    ctx = make_context("jiangsu")
    r = dispatch("get_weather_context", {"start": "2025-01", "end": "2025-03"}, ctx)
    assert r.ok
    payload = r.payload.model_dump(mode="json")
    assert payload["available"] is True
    assert payload["is_proxy"] is True, "单点气象代理必须显式声明"
    assert payload["proxy_note"]


def test_missing_context_is_declared_not_silently_empty():
    """没有天气数据的地区必须返回 available=false，而不是空列表。"""
    ctx = make_context("zhejiang")
    # 用一个不存在的区间触发"无记录"分支
    r = dispatch("get_weather_context", {"start": "1990-01", "end": "1990-03"}, ctx)
    assert r.ok
    payload = r.payload.model_dump(mode="json")
    assert payload["available"] is False
    assert payload["notes"]
