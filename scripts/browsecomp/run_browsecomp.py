from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import Agent
from agent.llm import LLMClient
from agent.tools import build_subagent_tools


def load_sample(metadata_path: Path, index: int) -> Dict[str, Any]:
    samples = load_samples(metadata_path)
    if index < 0 or index >= len(samples):
        raise IndexError(f"sample index {index} out of range; metadata has {len(samples)} rows")
    return samples[index]


def load_samples(metadata_path: Path) -> List[Dict[str, Any]]:
    samples = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"metadata must be a JSON array: {metadata_path}")
    return samples


def build_task(sample: Dict[str, Any]) -> str:
    docs = "\n".join(
        f"{idx}. {doc_path}" for idx, doc_path in enumerate(sample["evidence_docs"])
    )
    return f"""
You are the MainAgent for one BrowseComp-plus deep research task.

query_id: {sample["query_id"]}

Query:
{sample["query"]}

Evidence documents:
{docs}

Use SubAgents when document inspection is needed. The available documents are independent evidence candidates, so the efficient pattern is usually a fan-out/fan-in chain:
MainAgent -> {{SubAgent(document 0), SubAgent(document 1), ...}} -> MainAgent.

Planning guidance:
1. You may decide how many SubAgents are needed, but prefer broad document coverage when the answer depends on several criteria spread across different evidence documents.
2. For independent document checks, prefer issuing multiple call_subagent tool calls in the same assistant response instead of waiting for one result before starting the next.
3. In this BrowseComp-plus item, each listed evidence document is a curated candidate. Unless one document clearly settles the full answer, inspect the remaining listed documents before final synthesis.
4. Do not answer the query directly before you have enough document-backed evidence. If more document evidence is needed, call SubAgents first.
5. Each call_subagent task should handle exactly one evidence document. Do not combine multiple documents in one SubAgent task.
6. Create SubAgent tasks in listed document order when practical, so document_index values stay easy to compare.
7. Every SubAgent invocation must include the query_id, the full query, the document_index, and the exact document_path.
8. Each SubAgent must inspect exactly its assigned document_path, must not read any other document, and must not use outside knowledge.
9. Each SubAgent should return compact evidence relevant to the query, preferably as a valid JSON object with keys document_index, document_path, relevant, candidate_answer, findings, and missing.
10. Synthesize the final answer using only returned SubAgent tool results.

For each SubAgent, use this task shape exactly, filling in that document's index and path:
You are a document-level research SubAgent for BrowseComp-plus.
Your job is to inspect exactly one evidence document and extract information relevant to the query.
query_id: {sample["query_id"]}
document_index: <document_index>
document_path: <document_path>
Query:
{sample["query"]}
Hard constraints:
- You must use read_file on exactly this document_path.
- Do not read any other document.
- Do not use outside knowledge.
- Do not try to solve the whole multi-document task unless this single document is sufficient.
- Extract only facts that are supported by this document and useful for answering the query.
- Keep the response compact. Long copied passages are not allowed.
Return only a valid JSON object with exactly these keys: document_index, document_path, relevant, candidate_answer, findings, missing.

Final Answer Requirements:
Return only a valid JSON object with exactly these keys:
{{
  "query_id": "{sample['query_id']}",
  "prediction": "final answer string",
  "support": "brief explanation tying the relevant SubAgent tool results to the answer"
}}

No markdown, no code fences, no extra text.
""".strip()


def _tokenize(value: str) -> List[str]:
    """Tokenize normalized text into word-like tokens."""
    return re.findall(r"\w+", value.lower())


def normalize_answer(value: str) -> str:
    return " ".join(_tokenize(value))


def _ngram_counts(tokens: List[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def _f1_from_overlap(common: int, pred_count: int, ref_count: int) -> float:
    if pred_count == 0 and ref_count == 0:
        return 1.0
    if pred_count == 0 or ref_count == 0 or common == 0:
        return 0.0
    precision = common / pred_count
    recall = common / ref_count
    return (2 * precision * recall) / (precision + recall)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    pred_ngrams = _ngram_counts(pred_tokens, n)
    ref_ngrams = _ngram_counts(ref_tokens, n)
    common = sum((pred_ngrams & ref_ngrams).values())
    return _f1_from_overlap(common, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_len(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[idx] = previous[idx - 1] + 1
            else:
                current[idx] = max(previous[idx], current[idx - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    common = _lcs_len(pred_tokens, ref_tokens)
    return _f1_from_overlap(common, len(pred_tokens), len(ref_tokens))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score (word overlap on normalized text)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_prediction(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "rouge1": rouge_n(prediction, reference, 1),
        "rouge2": rouge_n(prediction, reference, 2),
        "rougeL": rouge_l(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "exact_match": float(normalize_answer(prediction) == normalize_answer(reference)),
    }


def parse_prediction(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    prediction = data.get("prediction", "")
    return prediction if isinstance(prediction, str) else ""


def selected_indices(total: int, args: argparse.Namespace) -> List[int]:
    if args.indices:
        indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
    elif args.all:
        indices = list(range(args.start, total))
    elif args.limit is not None:
        indices = list(range(args.start, min(total, args.start + args.limit)))
    else:
        indices = [args.index]

    invalid = [idx for idx in indices if idx < 0 or idx >= total]
    if invalid:
        raise IndexError(f"indices out of range for {total} rows: {invalid[:10]}")
    return indices


def mean_scores(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        count += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-plus items as MainAgent -> per-document SubAgents -> MainAgent."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/home/tanger/workspace/datasets/browsecomp-plus-100/metadata1.json"),
    )
    parser.add_argument("--index", type=int, default=0, help="Single row index used when no batch option is set.")
    parser.add_argument("--start", type=int, default=0, help="Start row for --all or --limit.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many rows from --start.")
    parser.add_argument("--all", action="store_true", help="Run all rows from --start.")
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated row indices to run, e.g. 0,4,9. Overrides --all/--limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSONL path for per-sample model results. Defaults to a timestamped "
            "file under outputs/browsecomp/."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "JSON path for aggregate metrics/final score. Defaults to the "
            "per-sample output path with .summary.json."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue batch evaluation after a sample fails.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sub-max-tokens", type=int, default=10240)
    parser.add_argument("--sub-max-iters", type=int, default=4)
    parser.add_argument("--final-max-tokens", type=int, default=10240)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-sub-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-kv", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def run_one(sample: Dict[str, Any], index: int, args: argparse.Namespace) -> Dict[str, Any]:
    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        agent_mode=True,
        enable_thinking=args.enable_thinking,
    )
    system_prompt = (
        "You are an AI agent that coordinates document-level SubAgents for "
        "BrowseComp-plus. Use your judgment to decide how many SubAgents are "
        "needed. When several independent evidence documents may contain different "
        "criteria, prefer batching those SubAgent calls in one assistant message, "
        "then synthesize the final answer only from returned tool results."
    )
    main_agent = Agent(
        name="main",
        system_prompt=system_prompt,
        llm=client,
        is_main_agent=True,
        tools=build_subagent_tools(),
        max_iters=max(len(sample["evidence_docs"]) + 3, 4),
        temperature=args.temperature,
        max_tokens=args.final_max_tokens,
    )

    print(f"[sample] index={index} query_id={sample['query_id']} docs={len(sample['evidence_docs'])}")

    try:
        answer = main_agent.run(build_task(sample))
        print("[final reused_prompt_tokens]")
        print(client.last_reused_tokens)
        print("[final usage]")
        print(json.dumps(client.last_usage, ensure_ascii=False, indent=2))
        print("[prediction]")
        print(answer)

        gold = sample["answer"]
        prediction = parse_prediction(answer)
        metrics = score_prediction(prediction, gold)
        if args.show_gold:
            print("[gold]")
            print(gold)
            print("[metrics]")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return {
            "index": index,
            "query_id": sample["query_id"],
            "query": sample["query"],
            "gold": gold,
            "prediction": prediction,
            "raw_answer": answer,
            "metrics": metrics,
            "final_reused_prompt_tokens": client.last_reused_tokens,
            "final_usage": client.last_usage,
        }
    finally:
        if args.release_kv and client.session_id:
            client.release_kv()
            print("[released kv]")


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def default_output_path(indices: List[int]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(indices) == 1:
        selection = f"idx{indices[0]}"
    else:
        selection = f"n{len(indices)}_from{indices[0]}_to{indices[-1]}"
    return ROOT / "outputs" / "browsecomp" / f"results_{selection}_{timestamp}.jsonl"


def resolve_output_paths(args: argparse.Namespace, indices: List[int]) -> tuple[Path, Path]:
    output = args.output or default_output_path(indices)
    summary_output = args.summary_output or output.with_suffix(".summary.json")
    return output, summary_output


def run(args: argparse.Namespace) -> None:
    samples = load_samples(args.metadata)
    indices = selected_indices(len(samples), args)
    output_path, summary_output_path = resolve_output_paths(args, indices)
    results: List[Dict[str, Any]] = []

    print("[output]")
    print(str(output_path))
    print("[summary_output]")
    print(str(summary_output_path))

    for ordinal, index in enumerate(indices, start=1):
        print(f"[progress] {ordinal}/{len(indices)}")
        try:
            result = run_one(samples[index], index, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "index": index,
                "query_id": samples[index].get("query_id"),
                "error": repr(exc),
            }
            print("[error]")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        results.append(result)
        write_jsonl_row(output_path, result)

    summary = {
        "metadata": str(args.metadata),
        "num_requested": len(indices),
        "num_completed": sum("metrics" in row for row in results),
        "num_errors": sum("error" in row for row in results),
        "metrics": mean_scores(results),
    }

    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(parse_args())
