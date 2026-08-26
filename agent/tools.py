from __future__ import annotations

import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent
from .llm import LLMClient

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text: str) -> str:
    """去掉 <think>...</think> 推理块（Qwen3 等模型会输出）。"""
    return _THINK_RE.sub("", text).strip()

class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
        aliases: Optional[List[str]] = None,
        root_only: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.aliases = list(aliases or [])
        self.root_only = root_only

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: Dict[str, Any]) -> Any:
        return self.func(**arguments)

def _tool_read_file(
    path: str,
    line_start: int = 1,
    line_end: Optional[int] = None,
) -> str:
    """读取文本文件，可指定行范围；失败时返回 ERROR 文本让模型自行调整。"""

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"ERROR: 文件不存在: {path}"
    except IsADirectoryError:
        return f"ERROR: {path} 是目录，请改用 list_directory"
    except OSError as exc:
        return f"ERROR: 读取 {path} 失败: {exc!r}"
    if not lines:
        return "(空文件)"
    start = max(1, line_start)
    end = len(lines) if line_end is None else min(len(lines), line_end)
    if start > end:
        return (
            f"ERROR: line_start({start}) > line_end({end})，"
            f"文件共 {len(lines)} 行"
        )
    body = "".join(lines[start - 1:end])
    if end < len(lines):
        body += f"\n...(共 {len(lines)} 行，已按 line_end 截断)"
    return body

def _tool_list_directory(path: str = ".") -> str:
    """列出目录条目（文件/子目录），失败返回 ERROR。"""
    try:
        entries = sorted(os.listdir(path))
    except FileNotFoundError:
        return f"ERROR: 目录不存在: {path}"
    except NotADirectoryError:
        return f"ERROR: {path} 不是目录"
    except OSError as exc:
        return f"ERROR: 列举 {path} 失败: {exc!r}"
    lines = []
    for name in entries:
        full = os.path.join(path, name)
        kind = "dir" if os.path.isdir(full) else "file"
        lines.append(f"{kind:<5}{name}")
    return "\n".join(lines) if lines else "(空目录)"


def _tool_search_files(pattern: str, path: str = ".") -> str:
    """按 glob 通配符模式递归搜索文件（只返回文件路径）。"""
    try:
        matches = sorted(
            p for p in glob.glob(os.path.join(path, "**", pattern), recursive=True)
            if os.path.isfile(p)
        )
    except OSError as exc:
        return f"ERROR: 搜索失败: {exc!r}"
    if not matches:
        return f"（没有匹配 {pattern!r} 的文件）"
    return "\n".join(matches)

def _call_subagent(task: str, client: LLMClient, trace: Optional[List[str]] = None) -> str:
    system_prompt = "You are a sub-agent responsible for completing subtasks delegated by the main AI agent. You are capable of solving complex tasks and completing assigned subtasks accurately. You have access to common file-reading and file-search tools."
    sub_agent = Agent(name="sub", system_prompt=system_prompt, llm=client, is_main_agent=False, max_tokens=10240, tools=build_file_tools(), trace=trace)
    content = sub_agent.run(task)
    content = strip_think(content)

    return content

def build_subagent_tools() -> List[Tool]:
    return [
        Tool(
            name="call_subagent",
            description=(
                "调用 SubAgent，使用子智能体完成一个子任务。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "子智能体需要完成的任务，需要把完成子任务所需要的所有必要资料和条件传递进来"}
                },
                "required": ["task"],
            },
            func=_call_subagent,
        )
    ]

def build_file_tools() -> List[Tool]:
    """内置文件工具集：主/子 agent 通用（read_file / write_file / list_directory / search_files）。"""
    return [
        Tool(
            name="read_file",
            description=(
                "读取文本文件内容（带行号，可指定行范围），用于查看代码、文档、配置等本地文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径"},
                    "line_start": {"type": "integer", "description": "起始行号（从 1 开始，默认 1）"},
                    "line_end": {"type": "integer", "description": "结束行号（默认读到底）"},
                },
                "required": ["path"],
            },
            func=_tool_read_file,
        ),
        Tool(
            name="list_directory",
            description="列出目录下的条目（文件/子目录），用于了解项目结构。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前目录"},
                },
                "required": [],
            },
            func=_tool_list_directory,
        ),
        Tool(
            name="search_files",
            description="按 glob 通配符模式递归搜索文件路径（如 *.py、**/test_*.py），用于定位文件。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，如 *.py"},
                    "path": {"type": "string", "description": "搜索起点目录，默认当前目录"},
                },
                "required": ["pattern"],
            },
            func=_tool_search_files,
        ),
    ]
