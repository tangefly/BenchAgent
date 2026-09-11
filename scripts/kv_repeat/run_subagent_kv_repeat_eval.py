from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import parse_json_arguments
from agent.llm import LLMClient
from agent.tools import build_subagent_tools
from agent.utils import strip_think
from scripts.browsecomp.run_browsecomp import score_prediction

METRIC_NAMES = ("rouge1", "rouge2", "rougeL", "token_f1", "exact_match")


DEFAULT_DATASET = Path("/home/tanger/workspace/datasets/subagent_kv_repeat_docs/questions.jsonl")


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


def chinese_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def numbered_body_lines(text: str) -> List[str]:
    return [line for line in extract_marked_output(text).splitlines() if line.startswith("L")]


def compare_texts(expected_raw: str, actual_raw: str) -> Dict[str, Any]:
    expected = extract_marked_output(expected_raw)
    actual = extract_marked_output(actual_raw)
    return score_prediction(actual, expected)


def trim_incomplete_sentence(text: str, finish_reason: Optional[str]) -> tuple[str, Dict[str, Any]]:
    """Conservative punctuation heuristic for the dataset's one-fact-per-line output.

    Only length-stopped generations are trimmed. For multiline notes, retain the
    prefix ending at the last line with terminal punctuation. This avoids treating
    a decimal or an abbreviation inside a truncated final line as a sentence end.
    For a single-line response, use the last punctuation followed by whitespace.
    """
    info = {"applied": False, "removed_chars": 0, "reason": "not_length_stopped"}
    if finish_reason != "length":
        return text, info
    terminal = r"[.!?。！？][\"'”’）)]*"
    if re.search(terminal + r"\s*$", text):
        info["reason"] = "ends_with_sentence_punctuation"
        return text, info
    if "\n" in text:
        boundary, offset = 0, 0
        for line in text.splitlines(keepends=True):
            offset += len(line)
            if re.search(terminal + r"\s*$", line):
                boundary = offset
    else:
        matches = list(re.finditer(terminal + r"(?=\s|$)", text))
        boundary = matches[-1].end() if matches else 0
    retained = text[:boundary].rstrip()
    info.update(applied=retained != text, removed_chars=len(text)-len(retained),
                reason="removed_incomplete_tail" if retained else "no_complete_sentence")
    return retained, info


@lru_cache(maxsize=2)
def output_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)


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


STRICT_REPEAT_INSTRUCTION = (
    "Copy the entire content of the preceding call_subagent tool result, from its first "
    "character to its last character, exactly once. This is a transcription task. "
    "The tool result is a fixed excerpt and may end in the middle of a word or sentence. "
    "Stop at that exact point: do not finish the word or sentence or continue the document. "
    "Include every line in the original order, including repeated facts. Do not remove "
    "duplicates, omit the beginning or middle, correct facts, or summarize. "
    "Output only the copied content, without an introduction, quotation marks, or code fences."
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


def call_subagent_tool(call: Dict[str, Any], client: LLMClient, trace: List[str],
                       max_tokens: int, temperature: float, expected_task: str,
                       document_text: Optional[str] = None) -> str:
    fn = call.get("function") or {}
    arguments = parse_json_arguments(fn.get("arguments") or "{}")
    task = arguments.get("task")
    if not isinstance(task, str) or not task.strip():
        raise RuntimeError(f"Invalid call_subagent arguments: {json.dumps(arguments, ensure_ascii=False)}")

    if fn.get("name") != "call_subagent":
        raise RuntimeError("Expected call_subagent")
    # Bind execution to the dataset task: model paraphrases cannot alter the experiment.
    task = expected_task
    sub_trace = list(trace)
    sub_trace.append("sub")
    if document_text is not None:
        # Private source input is never appended to main's messages.
        task += "\n\n<source_document>\n" + document_text + "\n</source_document>"
    # A single generation avoids tool calls and extra iterations consuming budget.
    result = client.chat(
        [{"role": "user", "content": task}],
        trace=sub_trace, temperature=temperature, max_tokens=max_tokens,
    )
    trace.append("sub")
    if result.get("tool_calls"):
        raise RuntimeError("SubAgent unexpectedly returned tool calls")
    return strip_think(result.get("content") or "")


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
    document_text = None
    if row.get("document_path"):
        dataset_dir = args.dataset.resolve().parent
        document_path = (dataset_dir / row["document_path"]).resolve()
        if not document_path.is_relative_to(dataset_dir):
            raise ValueError("Document must be inside the copied dataset")
        document_bytes = document_path.read_bytes()
        if hashlib.sha256(document_bytes).hexdigest() != row["document_sha256"]:
            raise ValueError(f"Document checksum mismatch: {document_path}")
        document_text = document_bytes.decode("utf-8")
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
        sub_max_tokens = args.sub_max_tokens if args.sub_max_tokens is not None else row.get("target_repeat_tokens")
        if not isinstance(sub_max_tokens, int) or sub_max_tokens <= 0:
            raise ValueError("A positive sub token budget is required")
        sub_output = call_subagent_tool(
            tool_calls[0], client, trace, sub_max_tokens,
            args.sub_temperature, row["subagent_prompt"], document_text,
        )
        sub_finish_reason = client.last_finish_reason
        sub_output_raw = sub_output
        trim_info = {"applied": False, "removed_chars": 0, "reason": "disabled"}
        if getattr(args, "trim_incomplete_sentence", False):
            sub_output, trim_info = trim_incomplete_sentence(sub_output, sub_finish_reason)
            if not sub_output:
                raise ValueError("Sub output contains no complete sentence; increase sub budget")
        returned_tokens = None
        if getattr(args, "tokenizer_path", None):
            returned_tokens = len(output_tokenizer(str(args.tokenizer_path)).encode(
                sub_output, add_special_tokens=False))
        sub_elapsed_ms = (time.perf_counter() - sub_start) * 1000.0
        sub_usage = dict(client.last_usage)
        actual_subagent_output_tokens = usage_value(sub_usage, "completion_tokens")
        sub_reused = client.last_reused_tokens
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "name": "call_subagent",
                "content": (
                    "BEGIN_KV_EXCERPT\n" + sub_output + "\nEND_KV_EXCERPT"
                    if getattr(args, "repeat_instruction", "baseline") == "bounded"
                    else sub_output
                ),
            }
        )

        if getattr(args, "repeat_instruction", "baseline") == "bounded":
            messages.append({"role": "user", "content": (
                "Transcribe only the text between BEGIN_KV_EXCERPT and END_KV_EXCERPT "
                "in the tool result above. Do not output the markers. Copy every character "
                "and every line, including duplicates and the final unfinished word. "
                "Do not complete the final fragment. Do not call any tool. "
                "Do not summarize, correct, explain, or add anything."
            )})
        if getattr(args, "repeat_instruction", "baseline") == "strict":
            messages.append({"role": "user", "content": STRICT_REPEAT_INSTRUCTION})
        trace.append("main")
        main_final, main_final_ms, main_final_usage, main_final_reused = chat_timed(
            client,
            messages,
            trace=trace,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        repeated_output = strip_think(main_final.get("content") or "")
        metrics = compare_texts(sub_output, repeated_output)
        task_check = None
        if row.get("content_type") == "deterministic_records":
            lines = sub_output.splitlines()
            complete = list(lines)
            partial_tail = None
            if complete and not re.fullmatch(r"R[0-9]{3}\|value=[0-9]{3}", complete[-1]):
                partial_tail = complete.pop()
            correct = sum(
                line == f"R{i:03d}|value={(i * 37 + row['seed']) % 1000:03d}"
                for i, line in enumerate(complete, 1)
            )
            task_check = {
                "complete_lines": len(complete),
                "correct_lines": correct,
                "complete_line_accuracy": correct / len(complete) if complete else None,
                "partial_tail": partial_tail,
            }

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
            "document_id": row.get("document_id"),
            "document_path": row.get("document_path"),
            "document_sha256": row.get("document_sha256"),
            "session_id": client.session_id,
            "sub_max_tokens": sub_max_tokens,
            "repeat_instruction": getattr(args, "repeat_instruction", "baseline"),
            "sub_temperature": args.sub_temperature,
            "requested_subagent_task": parse_json_arguments(
                tool_calls[0]["function"].get("arguments") or "{}"
            ).get("task"),
            "executed_subagent_task": row["subagent_prompt"],
            "sub_finish_reason": sub_finish_reason,
            "sentence_trimming": dict(trim_info, generated_chars=len(sub_output_raw),
                                      returned_chars=len(sub_output)),
            "main_finish_reason": client.last_finish_reason,
            "sub_budget_reached": actual_subagent_output_tokens == sub_max_tokens,
            "trace": trace,
            "metrics": metrics,
            "raw_exact_match": sub_output == repeated_output,
            "sub_task_check": task_check,
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
                "subagent_returned_text_tokens": returned_tokens,
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
            result["sub_output_raw"] = sub_output_raw
            result["main_repeated_output"] = repeated_output
        if getattr(args, "text_control", False):
            # Same sub output and same main messages, without an agent session or grafts.
            control = LLMClient(base_url=args.base_url, api_key=args.api_key,
                                model=args.model, timeout=args.timeout,
                                agent_mode=False, enable_thinking=args.enable_thinking)
            try:
                plain, latency, usage, reused = chat_timed(
                    control, messages, trace=[], temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                text = strip_think(plain.get("content") or "")
                result["text_control"] = {
                    "metrics": compare_texts(sub_output, text),
                    "raw_exact_match": sub_output == text,
                    "finish_reason": control.last_finish_reason,
                    "usage": usage, "reused_prompt_tokens": reused,
                    "latency_ms": latency,
                }
                if args.include_text:
                    result["text_control"]["output"] = text
            finally:
                control.session.close()
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
            "text_control_metrics": {
                name: mean(row["text_control"]["metrics"][name] for row in rows
                           if "text_control" in row)
                for name in METRIC_NAMES
            },
            "sub_budget_reached_rate": mean(float(row["sub_budget_reached"]) for row in rows),
            "sub_returned_text_tokens": mean(
                row["actual_tokens"].get("subagent_returned_text_tokens") for row in rows
                if row["actual_tokens"].get("subagent_returned_text_tokens") is not None
            ),
            "sentence_trimmed_rate": mean(
                float(row.get("sentence_trimming", {}).get("applied", False)) for row in rows
            ),
            "sub_output_tokens_min": min((row["actual_tokens"]["subagent_output_tokens"] for row in rows
                                          if row["actual_tokens"]["subagent_output_tokens"] is not None), default=None),
            "sub_output_tokens_max": max((row["actual_tokens"]["subagent_output_tokens"] for row in rows
                                          if row["actual_tokens"]["subagent_output_tokens"] is not None), default=None),
            "metrics": {
                name: mean(row["metrics"][name] for row in rows)
                for name in METRIC_NAMES
            },
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
    parser.add_argument("--model", default="exp-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--sub-max-tokens", type=int, default=None,
                        help="Override sub budget; defaults to row target_repeat_tokens")
    parser.add_argument("--sub-temperature", type=float, default=0.0)
    parser.add_argument("--repeat-instruction", choices=("baseline", "strict", "bounded"), default="baseline",
                        help="Strict adds a transcription reminder after the tool result")
    parser.add_argument("--trim-incomplete-sentence", action=argparse.BooleanOptionalAction,
                        default=True, help="Drop a length-truncated trailing sentence before main")
    parser.add_argument("--tokenizer-path", type=Path, default=None,
                        help="Local tokenizer for counting returned text tokens after trimming")
    parser.add_argument("--text-control", action="store_true",
                        help="Also repeat the same sub output without agent KV grafts")
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
                f"rouge1={metrics['rouge1']:.4f} "
                f"rouge2={metrics['rouge2']:.4f} "
                f"rougeL={metrics['rougeL']:.4f} "
                f"token_f1={metrics['token_f1']:.4f} "
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
            "repeat_instruction": args.repeat_instruction,
            "text_control": args.text_control,
            "trim_incomplete_sentence": args.trim_incomplete_sentence,
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
