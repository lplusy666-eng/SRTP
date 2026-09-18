"""工具包。import 本包即完成全部工具注册。

新增工具的步骤只有两步：
1. 在某个 tools/*.py 里用 @tool 装饰器定义；
2. 在本文件下方 import 该模块。

不需要改任何调度代码 —— JSON Schema 和 function-calling 格式都是自动导出的。
"""

from .base import (
    REGISTRY,
    ToolContext,
    ToolResult,
    ToolSpec,
    anthropic_tools,
    dispatch,
    get_spec,
    list_specs,
    openai_tools,
    render_for_llm,
    tool,
    tool_catalog_markdown,
)

# 以下 import 的副作用就是注册工具，顺序无关。
from . import context_tools, data_tools, decompose_tools, detect_tools  # noqa: F401,E402
from . import diagnosis_tools, economy_tools, knowledge_tools, report_tools  # noqa: F401,E402

__all__ = [
    "REGISTRY",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "anthropic_tools",
    "dispatch",
    "get_spec",
    "list_specs",
    "openai_tools",
    "render_for_llm",
    "tool",
    "tool_catalog_markdown",
]
