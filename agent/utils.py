import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text: str) -> str:
    """去掉 <think>...</think> 推理块（Qwen3 等模型会输出）。"""
    return _THINK_RE.sub("", text).strip()