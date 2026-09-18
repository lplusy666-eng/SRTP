"""通用工具：配置加载、路径处理、日志。"""
import os
import yaml
import logging
from pathlib import Path

# 项目根目录（configs 的上一级）
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = None) -> dict:
    """加载 YAML 配置。"""
    path = path or (ROOT / "configs" / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def abspath(rel: str) -> str:
    """把配置里的相对路径转成基于项目根目录的绝对路径。"""
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str(ROOT / rel)


def ensure_dir(file_path: str):
    """确保某文件所在目录存在。"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "PEA") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s | %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
