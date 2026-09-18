from __future__ import annotations

import os

import pandas as pd


def main() -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("未安装 UI 依赖，请执行 pip install '.[ui]'") from exc

    from .agents import PowerEconomyOrchestrator

    st.set_page_config(page_title="电力看经济智能体", layout="wide")
    st.title("基于特征解耦与知识增强的“电力看经济”智能体")
    config_path = st.sidebar.text_input(
        "配置文件", os.getenv("POWER_ECON_CONFIG", "configs/demo.yaml")
    )

    @st.cache_resource(show_spinner="加载模型与知识库…")
    def load_orchestrator(path: str):
        return PowerEconomyOrchestrator.from_config(path)

    try:
        orch = load_orchestrator(config_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    frame = orch.frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    min_date = frame["timestamp"].min().date()
    max_date = frame["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "分析时间段", value=(max(min_date, max_date - pd.Timedelta(days=30)), max_date)
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = str(date_range[0]), str(date_range[1])
    else:
        start, end = str(min_date), str(max_date)

    overview, diagnostics, report_tab, qa_tab = st.tabs(["态势总览", "异常诊断", "报告生成", "交互问答"])
    with overview:
        subset = frame.loc[
            (frame["timestamp"].dt.date >= pd.Timestamp(start).date())
            & (frame["timestamp"].dt.date <= pd.Timestamp(end).date())
        ]
        cols = st.columns(4)
        cols[0].metric("平均负荷(MW)", f"{subset['load_mw'].mean():.1f}")
        cols[1].metric("峰值(MW)", f"{subset['load_mw'].max():.1f}")
        cols[2].metric("数据点", f"{len(subset):,}")
        events = orch.monitor(start, end)
        cols[3].metric("代表性异常", str(len(events)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["load_mw"], name="实测负荷"))
        for event in events:
            fig.add_vline(x=pd.Timestamp(event.timestamp).timestamp() * 1000, line_dash="dash")
        fig.update_layout(height=430, xaxis_title="时间", yaxis_title="MW")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pd.DataFrame([e.model_dump(mode="json") for e in events]),
            use_container_width=True,
            hide_index=True,
        )

    with diagnostics:
        events = orch.monitor(start, end)
        if not events:
            st.info("当前时间段没有超过阈值的代表性异常。")
        else:
            options = {f"{e.timestamp} | {e.event_id} | {e.severity}": e for e in events}
            selected = st.selectbox("选择异常事件", list(options))
            if st.button("执行诊断", type="primary"):
                result = orch.diagnosis.diagnose(options[selected])
                st.subheader(result.narrative)
                st.write(result.economic_signal)
                st.dataframe(
                    pd.DataFrame([x.model_dump() for x in result.top_causes]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.dataframe(
                    pd.DataFrame([x.model_dump() for x in result.feature_contributions]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.warning(result.uncertainty)

    with report_tab:
        max_diag = st.slider("最大诊断事件数", 0, 20, 8)
        if st.button("生成报告"):
            result = orch.generate_report(start, end, max_diag)
            st.markdown(orch.report.to_markdown(result))

    with qa_tab:
        question = st.text_area("问题", "最近异常主要由什么造成？是否能说明经济走弱？")
        if st.button("提问"):
            answer = orch.ask(question, "streamlit")
            st.write(answer.answer)
            st.caption(f"置信度：{answer.confidence:.2f}")
            with st.expander("证据"):
                for item in answer.evidence:
                    st.write("-", item)


if __name__ == "__main__":
    main()
