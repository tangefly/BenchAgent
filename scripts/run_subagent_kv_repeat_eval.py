from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import parse_json_arguments
from agent.llm import LLMClient
from agent.tools import build_subagent_tools
from agent.utils import strip_think


DEFAULT_DATASET = ROOT / "data" / "subagent_kv_repeat_questions.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return strip_think(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_marked_output(text: str) -> str:
    normalized = normalize_text(text)
    begin = "BEGIN_SUBAGENT_OUTPUT"
    end = "END_SUBAGENT_OUTPUT"
    start = normalized.find(begin)
    stop = normalized.rfind(end)
    if start == -1 or stop == -1 or stop < start:
        return normalized
    return normalized[start : stop + len(end)].strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def positional_line_accuracy(expected: str, actual: str) -> float:
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    if not expected_lines:
        return 1.0 if not actual_lines else 0.0
    matched = sum(
        1 for idx, line in enumerate(expected_lines)
        if idx < len(actual_lines) and actual_lines[idx] == line
    )
    return matched / len(expected_lines)


def chinese_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def numbered_body_lines(text: str) -> List[str]:
    return [line for line in extract_marked_output(text).splitlines() if line.startswith("L")]


def compare_texts(expected_raw: str, actual_raw: str) -> Dict[str, Any]:
    expected = extract_marked_output(expected_raw)
    actual = extract_marked_output(actual_raw)
    distance = levenshtein_distance(expected, actual)
    denom = max(len(expected), 1)
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    return {
        "exact_match": expected == actual,
        "expected_chars": len(expected),
        "actual_chars": len(actual),
        "levenshtein_distance": distance,
        "char_accuracy": max(0.0, 1.0 - distance / denom),
        "sequence_ratio": matcher.ratio(),
        "line_accuracy": positional_line_accuracy(expected, actual),
        "expected_sha256": sha256_text(expected),
        "actual_sha256": sha256_text(actual),
        "has_begin_marker": "BEGIN_SUBAGENT_OUTPUT" in actual,
        "has_end_marker": "END_SUBAGENT_OUTPUT" in actual,
    }


def usage_value(usage: Dict[str, Any], key: str) -> Optional[int]:
    value = usage.get(key)
    return value if isinstance(value, int) else None


def main_system_prompt() -> str:
    return (
        "You are a coordinator for a KVCache repeatability experiment. "
        "Call exactly one SubAgent with the user-provided SubAgent task. "
        "After the SubAgent returns, output the SubAgent result verbatim. "
        "Do not summarize, correct, reformat, translate, or add any text."
    )


def build_main_task(subagent_prompt: str) -> str:
    return f"""Please call exactly one SubAgent with the following task.

SubAgent task:
{subagent_prompt}

After the SubAgent returns, your final answer must be an exact verbatim copy of the full SubAgent output.
Do not add labels, Markdown, code fences, commentary, or any text before or after the copied content.
Preserve every line break and character as much as possible.""".strip()


def make_first_message(task: str) -> Dict[str, Any]:
    return {"role": "user", "content": f"【系统设定】\n{main_system_prompt()}\n\n【任务】\n{task}"}


def call_subagent_tool(call: Dict[str, Any], client: LLMClient, trace: List[str]) -> str:
    fn = call.get("function") or {}
    arguments = parse_json_arguments(fn.get("arguments") or "{}")
    task = arguments.get("task")
    if not isinstance(task, str) or not task.strip():
        raise RuntimeError(f"Invalid call_subagent arguments: {json.dumps(arguments, ensure_ascii=False)}")

    tool = build_subagent_tools()[0]
    sub_trace = list(trace)
    sub_trace.append("sub")
    result = tool.call({"task": task, "client": client, "trace": sub_trace})
    trace.append("sub")
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def selected_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if args.case_id:
        wanted = set(args.case_id)
        indexed = [(idx, row) for idx, row in indexed if row.get("case_id") in wanted]
    if args.length_bucket:
        wanted_buckets = set(args.length_bucket)
        indexed = [(idx, row) for idx, row in indexed if row.get("length_bucket") in wanted_buckets]
    if args.target_tokens:
        wanted_tokens = set(args.target_tokens)
        indexed = [(idx, row) for idx, row in indexed if row.get("target_repeat_tokens") in wanted_tokens]
    if args.content_type:
        wanted_types = set(args.content_type)
        indexed = [(idx, row) for idx, row in indexed if row.get("content_type") in wanted_types]
    if args.start:
        indexed = indexed[args.start :]
    if args.limit is not None:
        indexed = indexed[: args.limit]
    return indexed


def chat_timed(
    client: LLMClient,
    messages: List[Dict[str, Any]],
    trace: List[str],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[Dict[str, Any], float, Dict[str, int], int]:
    start = time.perf_counter()
    message = client.chat(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        trace=trace,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return message, elapsed_ms, dict(client.last_usage), client.last_reused_tokens


def run_one(row: Dict[str, Any], dataset_index: int, args: argparse.Namespace) -> Dict[str, Any]:
    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        agent_mode=True,
        enable_thinking=args.enable_thinking,
    )
    tools_json = [tool.schema() for tool in build_subagent_tools()]
    trace: List[str] = ["main"]
    messages: List[Dict[str, Any]] = [make_first_message(build_main_task(row["subagent_prompt"]))]

    try:
        main_first, main_first_ms, main_first_usage, main_first_reused = chat_timed(
            client,
            messages,
            trace=trace,
            tools=tools_json,
            tool_choice="required",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        tool_calls = main_first.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise RuntimeError(f"Expected exactly one call_subagent tool call, got {len(tool_calls)}")

        messages.append(
            {
                "role": "assistant",
                "content": main_first.get("content"),
                "tool_calls": tool_calls,
            }
        )

        sub_start = time.perf_counter()
        sub_output = call_subagent_tool(tool_calls[0], client, trace)
        sub_elapsed_ms = (time.perf_counter() - sub_start) * 1000.0
        sub_usage = dict(client.last_usage)
        actual_subagent_output_tokens = usage_value(sub_usage, "completion_tokens")
        sub_reused = client.last_reused_tokens
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "name": "call_subagent",
                "content": sub_output,
            }
        )

        trace.append("main")
        main_final, main_final_ms, main_final_usage, main_final_reused = chat_timed(
            client,
            messages,
            trace=trace,
            tools=tools_json,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        repeated_output = strip_think(main_final.get("content") or "")
        metrics = compare_texts(sub_output, repeated_output)

        sub_marked_output = extract_marked_output(sub_output)
        sub_body_lines = numbered_body_lines(sub_output)
        min_body_lines = row.get("min_body_lines")
        min_output_chars = row.get("min_output_chars")

        result: Dict[str, Any] = {
            "dataset_index": dataset_index,
            "case_id": row.get("case_id"),
            "length_bucket": row.get("length_bucket"),
            "target_repeat_tokens": row.get("target_repeat_tokens"),
            "min_body_lines": min_body_lines,
            "min_chars_per_line": row.get("min_chars_per_line"),
            "min_output_chars": min_output_chars,
            "content_type": row.get("content_type"),
            "topic": row.get("topic"),
            "session_id": client.session_id,
            "trace": trace,
            "metrics": metrics,
            "timing_ms": {
                "main_first": main_first_ms,
                "subagent": sub_elapsed_ms,
                "main_final": main_final_ms,
            },
            "usage": {
                "main_first": main_first_usage,
                "subagent": sub_usage,
                "main_final": main_final_usage,
            },
            "actual_tokens": {
                "subagent_output_tokens": actual_subagent_output_tokens,
                "target_repeat_tokens": row.get("target_repeat_tokens"),
                "main_repeat_output_tokens": usage_value(main_final_usage, "completion_tokens"),
            },
            "subagent_output_shape": {
                "marked_chars": len(sub_marked_output),
                "marked_chinese_chars": chinese_char_count(sub_marked_output),
                "numbered_body_lines": len(sub_body_lines),
                "meets_min_body_lines": (
                    len(sub_body_lines) >= min_body_lines
                    if isinstance(min_body_lines, int)
                    else None
                ),
                "meets_min_output_chars": (
                    len(sub_marked_output) >= min_output_chars
                    if isinstance(min_output_chars, int)
                    else None
                ),
            },
            "reused_prompt_tokens": {
                "main_first": main_first_reused,
                "subagent": sub_reused,
                "main_final": main_final_reused,
            },
            "sub_output_sha256": sha256_text(extract_marked_output(sub_output)),
            "main_repeat_sha256": sha256_text(extract_marked_output(repeated_output)),
        }
        if args.include_text:
            result["sub_output"] = sub_output
            result["main_repeated_output"] = repeated_output
        return result
    finally:
        if args.release_kv and client.session_id:
            client.release_kv()


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in results if "metrics" in row]
    errors = [row for row in results if "error" in row]

    def bucket_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(rows),
            "exact_match_rate": mean(1.0 if row["metrics"]["exact_match"] else 0.0 for row in rows),
            "char_accuracy": mean(row["metrics"]["char_accuracy"] for row in rows),
            "sequence_ratio": mean(row["metrics"]["sequence_ratio"] for row in rows),
            "line_accuracy": mean(row["metrics"]["line_accuracy"] for row in rows),
            "subagent_output_tokens": mean(
                row["actual_tokens"]["subagent_output_tokens"]
                for row in rows
                if row["actual_tokens"]["subagent_output_tokens"] is not None
            ),
            "subagent_marked_chars": mean(
                row["subagent_output_shape"]["marked_chars"] for row in rows
            ),
            "subagent_numbered_body_lines": mean(
                row["subagent_output_shape"]["numbered_body_lines"] for row in rows
            ),
            "meets_min_body_lines_rate": mean(
                1.0 if row["subagent_output_shape"]["meets_min_body_lines"] else 0.0
                for row in rows
                if row["subagent_output_shape"]["meets_min_body_lines"] is not None
            ),
            "meets_min_output_chars_rate": mean(
                1.0 if row["subagent_output_shape"]["meets_min_output_chars"] else 0.0
                for row in rows
                if row["subagent_output_shape"]["meets_min_output_chars"] is not None
            ),
            "main_repeat_output_tokens": mean(
                row["actual_tokens"]["main_repeat_output_tokens"]
                for row in rows
                if row["actual_tokens"]["main_repeat_output_tokens"] is not None
            ),
            "main_final_reused_prompt_tokens": mean(
                row["reused_prompt_tokens"]["main_final"] for row in rows
            ),
            "main_final_latency_ms": mean(row["timing_ms"]["main_final"] for row in rows),
        }

    by_length: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_length[str(row.get("length_bucket"))].append(row)
        by_type[str(row.get("content_type"))].append(row)

    return {
        "num_completed": len(completed),
        "num_errors": len(errors),
        "overall": bucket_summary(completed),
        "by_length_bucket": {
            key: bucket_summary(value) for key, value in sorted(by_length.items())
        },
        "by_content_type": {
            key: bucket_summary(value) for key, value in sorted(by_type.items())
        },
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / "subagent_kv_repeat" / f"results_{timestamp}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SubAgent KVCache repeatability by comparing SubAgent output and MainAgent verbatim repeat."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--length-bucket", action="append")
    parser.add_argument("--target-tokens", type=int, action="append")
    parser.add_argument("--content-type", action="append")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.dataset)
    indexed_rows = selected_rows(rows, args)
    output_path = args.output or default_output_path()
    summary_path = args.summary_output or output_path.with_suffix(".summary.json")
    results: List[Dict[str, Any]] = []

    print("[dataset]", args.dataset)
    print("[selected]", len(indexed_rows))
    print("[output]", output_path)
    print("[summary_output]", summary_path)

    for ordinal, (dataset_index, row) in enumerate(indexed_rows, start=1):
        print(
            f"[progress] {ordinal}/{len(indexed_rows)} "
            f"{row.get('case_id')} {row.get('length_bucket')} {row.get('content_type')}"
        )
        try:
            result = run_one(row, dataset_index, args)
            metrics = result["metrics"]
            print(
                "[result] "
                f"exact={metrics['exact_match']} "
                f"char_acc={metrics['char_accuracy']:.4f} "
                f"line_acc={metrics['line_accuracy']:.4f} "
                f"sub_tokens={result['actual_tokens']['subagent_output_tokens']} "
                f"sub_chars={result['subagent_output_shape']['marked_chars']} "
                f"sub_lines={result['subagent_output_shape']['numbered_body_lines']} "
                f"reused={result['reused_prompt_tokens']['main_final']}"
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "dataset_index": dataset_index,
                "case_id": row.get("case_id"),
                "length_bucket": row.get("length_bucket"),
                "target_repeat_tokens": row.get("target_repeat_tokens"),
                "content_type": row.get("content_type"),
                "topic": row.get("topic"),
                "error": repr(exc),
            }
            print("[error]", json.dumps(result, ensure_ascii=False))

        results.append(result)
        write_jsonl_row(output_path, result)

    summary = summarize(results)
    summary.update(
        {
            "dataset": str(args.dataset),
            "output": str(output_path),
            "num_selected": len(indexed_rows),
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
