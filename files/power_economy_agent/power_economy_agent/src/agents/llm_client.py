"""
LLM 客户端
==========
统一封装大模型调用。兼容 OpenAI 接口（DeepSeek / 通义千问兼容模式 / 智谱 / OpenAI 等）。
当 cfg.llm.enabled=False 或无Key时，自动降级为"离线模板模式"，
用规则拼装结构化文本，保证整条流水线在无大模型时也能跑通并产出可读结果。
"""
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_logger

log = get_logger("llm")


class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg["llm"]
        self.enabled = self.cfg.get("enabled", False)
        self.client = None
        if self.enabled:
            key = os.environ.get(self.cfg["api_key_env"], "")
            if not key:
                log.warning("未检测到环境变量 %s，降级为离线模板模式", self.cfg["api_key_env"])
                self.enabled = False
            else:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(base_url=self.cfg["base_url"], api_key=key)
                    log.info("LLM 已启用: %s @ %s", self.cfg["model"], self.cfg["base_url"])
                except Exception as e:
                    log.warning("LLM 初始化失败(%s)，降级为离线模板模式", e)
                    self.enabled = False
        if not self.enabled:
            log.info("LLM 运行于离线模板模式（设置 llm.enabled=true 并配置Key以启用真实大模型）")

    def chat(self, system: str, user: str, offline_fn=None) -> str:
        """system/user 提示 -> 回复。离线时调用 offline_fn(system,user) 生成兜底文本。"""
        if self.enabled and self.client is not None:
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg["model"],
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.cfg.get("temperature", 0.3),
                    max_tokens=self.cfg.get("max_tokens", 1200),
                )
                return resp.choices[0].message.content
            except Exception as e:
                log.warning("LLM 调用失败(%s)，本次降级为离线模板", e)
        if offline_fn is not None:
            return offline_fn(system, user)
        return "[离线模式] " + user[:200]
