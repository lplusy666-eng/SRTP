# API 与 MCP 使用

## 1. FastAPI 启动

```bash
power-econ serve -c configs/demo.yaml --host 0.0.0.0 --port 8000
```

环境变量也可指定配置：

```bash
export POWER_ECON_CONFIG=/absolute/path/to/configs/demo.yaml
```

交互文档：

```text
http://127.0.0.1:8000/docs
```

## 2. 健康检查

```http
GET /health
```

示例响应：

```json
{
  "status": "ok",
  "config": "/path/configs/demo.yaml",
  "features_ready": true,
  "model_ready": true
}
```

## 3. 最新预测

```http
GET /v1/forecast/latest
```

返回未来每个步长的 `lower_mw`、`median_mw`、`upper_mw` 和五组门控权重。

```bash
curl "http://127.0.0.1:8000/v1/forecast/latest"
```

## 4. 异常列表

```http
GET /v1/anomalies?start=2024-05-01&end=2024-05-10&limit=10
```

```bash
curl "http://127.0.0.1:8000/v1/anomalies?start=2024-05-01&end=2024-05-10&limit=10"
```

返回字段包括：

- `event_id`
- `timestamp`
- `observed_load_mw`
- `expected_load_mw`
- `residual_mw`
- `forecast_score`
- `vae_score`
- `anomaly_score`
- `threshold`
- `severity`
- `gate_weights`

## 5. 单事件诊断

```http
GET /v1/diagnoses/{event_id}
```

```bash
curl "http://127.0.0.1:8000/v1/diagnoses/6786696aebfd6c7c"
```

必须先通过异常接口生成/登记事件，或数据库中已有该事件。

## 6. 生成报告

```http
POST /v1/reports
Content-Type: application/json
```

请求：

```json
{
  "start": "2024-05-01",
  "end": "2024-05-10",
  "max_diagnoses": 5
}
```

curl：

```bash
curl -X POST "http://127.0.0.1:8000/v1/reports" \
  -H "Content-Type: application/json" \
  -d '{"start":"2024-05-01","end":"2024-05-10","max_diagnoses":5}'
```

## 7. 问答

```http
POST /v1/qa
Content-Type: application/json
```

请求：

```json
{
  "question": "这次异常是否能够说明经济走弱？",
  "session_id": "research-001"
}
```

```bash
curl -X POST "http://127.0.0.1:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"question":"这次异常是否能够说明经济走弱？","session_id":"research-001"}'
```

## 8. 模型摘要

```http
GET /v1/model/summary
```

返回训练、异常和 nowcast 指标以及产物路径。

---

## 9. Python 客户端示例

```python
from __future__ import annotations

import httpx

BASE_URL = "http://127.0.0.1:8000"

with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
    health = client.get("/health")
    health.raise_for_status()
    print(health.json())

    events = client.get(
        "/v1/anomalies",
        params={"start": "2024-05-01", "end": "2024-05-10", "limit": 3},
    )
    events.raise_for_status()
    event_list = events.json()

    if event_list:
        event_id = event_list[0]["event_id"]
        diagnosis = client.get(f"/v1/diagnoses/{event_id}")
        diagnosis.raise_for_status()
        print(diagnosis.json())

    answer = client.post(
        "/v1/qa",
        json={
            "question": "最近异常主要由什么造成？",
            "session_id": "python-client",
        },
    )
    answer.raise_for_status()
    print(answer.json()["answer"])
```

同样示例保存在 `examples/api_client.py`。

---

## 10. 生产 API 建议

当前 API 是研究原型。生产部署需至少增加：

- OAuth2/JWT 或网关鉴权；
- TLS；
- IP/用户限流；
- 请求和结果审计；
- 敏感字段脱敏；
- 超时、熔断和任务队列；
- PostgreSQL 等并发数据库；
- 模型进程与 API 进程资源隔离；
- 版本化端点和灰度发布。

---

# MCP

## 11. 安装与启动

```bash
pip install -e ".[mcp]"
power-econ mcp -c configs/demo.yaml --transport stdio
```

当前工程固定使用 MCP Python SDK v1 系列兼容范围：

```text
mcp>=1.27,<2
```

## 12. 提供的工具

### `list_anomalies`

参数：

```json
{
  "start": "2024-05-01",
  "end": "2024-05-10",
  "limit": 20
}
```

### `diagnose_event`

```json
{
  "event_id": "6786696aebfd6c7c"
}
```

### `generate_power_economy_report`

```json
{
  "start": "2024-05-01",
  "end": "2024-05-10",
  "max_diagnoses": 5
}
```

### `ask_power_economy`

```json
{
  "question": "异常是否能说明经济走弱？",
  "session_id": "mcp-demo"
}
```

### `latest_load_forecast`

无参数，返回最新多步区间预测。

## 13. 客户端配置

模板：

```json
{
  "mcpServers": {
    "power-economy-agent": {
      "command": "power-econ",
      "args": [
        "mcp",
        "--config",
        "/ABSOLUTE/PATH/power_econ_agent/configs/demo.yaml",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

完整文件：`examples/mcp_client_config.json`。

Windows 使用虚拟环境可执行文件绝对路径，例如：

```json
{
  "command": "C:\\project\\power_econ_agent\\.venv\\Scripts\\power-econ.exe"
}
```

## 14. MCP 调试

确认独立命令能启动：

```bash
POWER_ECON_CONFIG=$(pwd)/configs/demo.yaml power-econ mcp --transport stdio
```

stdio 传输会等待 MCP 客户端输入，看起来“没有输出”是正常现象。若启动即报错，优先检查：

- 是否安装 `.[mcp]`；
- 配置是否为绝对路径；
- 模型和特征是否存在；
- 客户端启动命令是否使用正确虚拟环境。
