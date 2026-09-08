from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import parse_json_arguments
from agent.llm import LLMClient
from agent.tools import build_subagent_tools


def build_task(document_path: str, questions: List[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    return f"""
You are the MainAgent. You must answer the following questions by delegating work to SubAgents.

Questions:
{numbered}

Document path for every SubAgent:
{document_path}

Hard execution constraints:
1. In your FIRST assistant response, call exactly {len(questions)} call_subagent tools in the SAME response.
2. Each tool call must handle exactly one question.
3. Do not answer any question directly in the first response.
4. Each call_subagent task must include the exact document path and the exact question.
5. After all tool results are returned, produce only a compact JSON object with keys answer1 through answer{len(questions)}.
6. Do not include markdown, code fences, or explanatory text in the final answer.
""".strip()


def main_system_prompt() -> str:
    return (
        "You are an AI agent that coordinates several SubAgents. When asked to delegate "
        "multiple independent questions, you should issue all required call_subagent tool "
        "calls in one assistant message. After the tool results return, synthesize the final "
        "answer exactly in the requested format."
    )


def make_first_message(system_prompt: str, task: str) -> Dict[str, Any]:
    # Keep the same message style as scripts/agent_infer.py for comparable prompts.
    return {"role": "user", "content": f"[System]\n{system_prompt}\n\n[Task]\n{task}"}


def get_stats(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/stats" if base_url.rstrip('/').endswith('/v1') else f"{base_url.rstrip('/')}/v1/stats"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": repr(exc)}


def call_subagent_tool(call: Dict[str, Any], client: LLMClient, trace: List[str]) -> str:
    fn = call.get("function") or {}
    arguments = parse_json_arguments(fn.get("arguments") or "{}")
    task = arguments.get("task")
    if not isinstance(task, str) or not task.strip():
        return f"ERROR: invalid call_subagent arguments: {json.dumps(arguments, ensure_ascii=False)}"

    tool = build_subagent_tools()[0]
    sub_trace = list(trace)
    sub_trace.append("sub")
    result = tool.call({"task": task, "client": client, "trace": sub_trace})
    trace.append("sub")
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def run(args: argparse.Namespace) -> None:
    questions = args.question or [
        "Who played the role of Ken Neville in Alias - the Bad Man?",
        "How did Ken Neville gain the trust of Rance Collins in Alias - the Bad Man?",
        "Why did Ken Neville initially hide his identity in the town?",
    ]

    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        agent_mode=True,
        enable_thinking=args.enable_thinking,
    )
    base_url = client.base_url
    tools_json = [tool.schema() for tool in build_subagent_tools()]
    main_trace: List[str] = ["main"]
    messages: List[Dict[str, Any]] = [make_first_message(main_system_prompt(), build_task(args.document, questions))]

    print("[stats before]")
    print(json.dumps(get_stats(base_url), ensure_ascii=False, indent=2))

    try:
        print("[main request 1: expect multiple call_subagent tool calls]")
        assistant = client.chat(
            messages,
            tools=tools_json,
            tool_choice="required",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            trace=main_trace,
        )
        tool_calls = assistant.get("tool_calls") or []
        print(f"[main tool calls] {len(tool_calls)}")
        print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
        if len(tool_calls) != len(questions):
            raise RuntimeError(
                f"Expected {len(questions)} tool calls in one main response, got {len(tool_calls)}. "
                "Try lowering temperature or making the prompt stricter."
            )

        messages.append({
            "role": "assistant",
            "content": assistant.get("content"),
            "tool_calls": tool_calls,
        })

        for idx, call in enumerate(tool_calls, start=1):
            print(f"[sub agent {idx} request]")
            result = call_subagent_tool(call, client, main_trace)
            print(f"[sub agent {idx} result]")
            print(result)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

        main_trace.append("main")
        print("[main request 2: final answer, should reuse all recent sub KV segments]")
        final = client.chat(
            messages,
            tools=tools_json,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            trace=main_trace,
        )
        print("[final reused_prompt_tokens]")
        print(client.last_reused_tokens)
        print("[final usage]")
        print(json.dumps(client.last_usage, ensure_ascii=False, indent=2))
        print("[final answer]")
        print(final.get("content") or "")
        print("[stats after]")
        print(json.dumps(get_stats(base_url), ensure_ascii=False, indent=2))
    finally:
        if args.release_kv:
            client.release_kv()
            print("[released kv]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a main -> {sub, sub, ...} -> main flow to test LMInfer multi-sub KV reuse."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--document", default="/home/tanger/workspace/BenchAgent/data/documents/doc1.txt")
    parser.add_argument("--question", action="append", help="Question to delegate. Repeat for multiple questions.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-kv", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
