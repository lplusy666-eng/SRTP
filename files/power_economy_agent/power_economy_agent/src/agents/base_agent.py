"""
智能体基类 + MCP 风格共享上下文
================================
借鉴 MCP(Model Context Protocol) 思想，实现一个轻量协同机制：
- SharedContext：所有智能体共享的上下文/黑板，存放中间产物（数据、异常、证据、结论）
- Tool：可被智能体调用的能力单元（如"检索知识""查图谱""调用模型"）
- BaseAgent：每个智能体声明自己的角色、可用工具，从共享上下文读写。
协同器(coordinator)按流程调度各智能体，实现"感知→诊断→生成"的闭环。
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_logger

log = get_logger("agent")


class SharedContext:
    """多智能体共享黑板（MCP 中的 context/resources）。"""
    def __init__(self):
        self.store = {}
        self.trace = []   # 协同过程留痕，用于可信输出

    def set(self, key, value):
        self.store[key] = value

    def get(self, key, default=None):
        return self.store.get(key, default)

    def log_step(self, agent, action, detail=""):
        entry = {"agent": agent, "action": action, "detail": detail}
        self.trace.append(entry)
        log.info("[%s] %s %s", agent, action, ("| " + detail) if detail else "")


class Tool:
    """能力单元（MCP 中的 tool）。"""
    def __init__(self, name, func, description=""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class BaseAgent:
    def __init__(self, name, role, ctx: SharedContext, tools=None):
        self.name = name
        self.role = role
        self.ctx = ctx
        self.tools = {t.name: t for t in (tools or [])}

    def use(self, tool_name, *args, **kwargs):
        self.ctx.log_step(self.name, f"调用工具 {tool_name}")
        return self.tools[tool_name](*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError
