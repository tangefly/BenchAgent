from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .utils import strip_think

def parse_json_arguments(raw: str) -> Dict[str, Any]:
    """解析模型返回的工具参数 JSON 字符串（解析失败或非对象时返回空 dict）。"""
    try:
        arguments = json.loads(raw or "{}")
        return arguments if isinstance(arguments, dict) else {}
    except json.JSONDecodeError:
        return {}

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        is_main_agent: bool,
        tools: Optional[List[Any]] = None,
        max_iters: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        trace: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools: Dict[str, Any] = {t.name: t for t in (tools or [])}
        self.max_iters = max_iters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools_json = self._build_tool_json(tools)
        # agent 调用链: 每次模型请求都发送, 末位始终是当前 agent。
        # 默认从当前 agent 名开始(主 agent 即 ["main"]); 子 agent 由父级传入
        # 完整调用链(如 ["main","sub"])。LMInfer 按 trace 末位判定 KV 段类型
        # (main / sub), 子 agent 若带 ["main"] 会被当成 main 段, 覆盖主段。
        self.trace = list(trace) if trace is not None else [self.name]

    def run(self, task: str) -> str:
        messages: List[Dict[str, Any]] = [self._first_message(task)]

        for step in range(1, self.max_iters + 1):
            assistant = self.llm.chat(
                messages,
                max_tokens=self.max_tokens,
                tools=self.tools_json,
                trace=self.trace
            )

            assistant_content = assistant.get("content")
            if isinstance(assistant_content, str):
                assistant_content = strip_think(assistant_content)

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                final = assistant_content or ""
                print(self.trace)
                return final

            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                arguments = parse_json_arguments(fn.get("arguments"))
                result = self._run_tool(name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,  # 文本协议后端用它格式化「[工具结果] 工具名: ...」
                    "content": result,
                })
            if self.name == "main":
                self.trace.append("main")

        raise RuntimeError(f"[{self.name}] 达到最大迭代次数 {self.max_iters}，任务未完成")

    def _first_message(self, task: str) -> Dict[str, Any]:
        return {"role": "user", "content": f"【系统设定】\n{self.system_prompt}\n\n【任务】\n{task}"}

    def _resolve_tool(self, name: str):
        tool = self.tools.get(name)
        if tool is not None:
            return tool, None
        for t in self.tools.values():
            if name in t.aliases:
                return t, name
        return None, None

    def _run_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        print(f"[{self.name} Tool Call] {name}, {arguments}")
        tool, alias_for = self._resolve_tool(name)
        if tool is None:
            alias_names = sorted(a for t in self.tools.values() for a in t.aliases)
            hint = (
                f"；另外可以直接用子代理名 {alias_names} 调用（效果等同 call_sub_agent）"
                if alias_names else ""
            )
            return f"ERROR: 未知工具 {name!r}，可用工具: {sorted(self.tools)}{hint}"
        if alias_for is not None:
            if "name" in (tool.parameters.get("properties") or {}) and "name" not in arguments:
                arguments = dict(arguments)
                arguments["name"] = alias_for
        if tool.root_only and not self.is_root:
            return (
                f"ERROR: 工具 {name!r} 只能由 main agent 调用，sub agent 禁止继续向下"
                f"分派（最多两级 agent）；请直接作答，或改用你自带的普通工具。"
            )
        # Add LLMClient to SubAgent Args
        if name == "call_subagent":
            self.trace.append("sub")  # 子 agent 的请求链以 "sub" 结尾(LMInfer 按末位定段)
            arguments["client"] = self.llm
            arguments["trace"] = list(self.trace)  # 完整调用链传给子 agent, 其请求直接沿用
        try:
            result = tool.call(arguments)
        except Exception as exc:
            result = f"ERROR: 工具 {name} 执行失败: {exc!r}"
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        return result
        
    def _build_tool_json(self, tools_list):
        tools_json = []
        for tool in (tools_list or []):
            tools_json.append(tool.schema())
        return tools_json