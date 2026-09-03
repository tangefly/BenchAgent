from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import Agent
from agent.llm import LLMClient
from agent.tools import build_subagent_tools


def load_sample(metadata_path: Path, index: int) -> Dict[str, Any]:
    samples = json.loads(metadata_path.read_text(encoding="utf-8"))
    if index < 0 or index >= len(samples):
        raise IndexError(f"sample index {index} out of range; metadata has {len(samples)} rows")
    return samples[index]


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

def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_prediction(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    prediction = data.get("prediction", "")
    return prediction if isinstance(prediction, str) else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one BrowseComp-plus item as MainAgent -> per-document SubAgents -> MainAgent."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/home/tanger/workspace/datasets/browsecomp-plus-100/metadata1.json"),
    )
    parser.add_argument("--index", type=int, default=0)
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


def run(args: argparse.Namespace) -> None:
    sample = load_sample(args.metadata, args.index)
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

    print(f"[sample] index={args.index} query_id={sample['query_id']} docs={len(sample['evidence_docs'])}")

    try:
        answer = main_agent.run(build_task(sample))
        print("[final reused_prompt_tokens]")
        print(client.last_reused_tokens)
        print("[final usage]")
        print(json.dumps(client.last_usage, ensure_ascii=False, indent=2))
        print("[prediction]")
        print(answer)

        if args.show_gold:
            gold = sample["answer"]
            prediction = parse_prediction(answer)
            print("[gold]")
            print(gold)
            print("[exact_match]")
            print(normalize_answer(prediction) == normalize_answer(gold))
    finally:
        if args.release_kv and client.session_id:
            client.release_kv()
            print("[released kv]")


if __name__ == "__main__":
    run(parse_args())
