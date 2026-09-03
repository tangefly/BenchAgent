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
from agent.tools import build_file_tools


def load_sample(metadata_path: Path, index: int) -> Dict[str, Any]:
    samples = json.loads(metadata_path.read_text(encoding="utf-8"))
    if index < 0 or index >= len(samples):
        raise IndexError(f"sample index {index} out of range; metadata has {len(samples)} rows")
    return samples[index]


def build_doc_task(query_id: str, query: str, doc_index: int, doc_path: str) -> str:
    return f"""
You are a document-level research SubAgent for BrowseComp-plus.

Your job is to inspect exactly one evidence document and extract information relevant to the query.

query_id: {query_id}

document_index: {doc_index}
document_path: {doc_path}

Query:
{query}

Hard constraints:
1. You must use read_file on exactly this document_path: {doc_path}
2. Do not read any other document.
3. Do not use outside knowledge.
4. Do not try to solve the whole multi-document task unless this single document is sufficient.
5. Extract only facts that are supported by this document and useful for answering the query.
6. Keep the response compact. Long copied passages are not allowed.

Return only a valid JSON object with exactly these keys:
{{
  "document_index": {doc_index},
  "document_path": "{doc_path}",
  "relevant": true,
  "candidate_answer": "answer candidate if this document supports one, otherwise empty string",
  "findings": ["short supported fact 1", "short supported fact 2"],
  "missing": ["query criteria not addressed by this document"]
}}
""".strip()


def build_final_messages(sample: Dict[str, Any], doc_reports: List[str]) -> List[Dict[str, Any]]:
    reports = "\n\n".join(
        f"[Document report {idx}]\n{report}" for idx, report in enumerate(doc_reports)
    )
    system = (
        "You are the MainAgent for a BrowseComp-plus deep research task. "
        "Synthesize the final answer using only the document-level SubAgent reports. "
        "Do not use outside knowledge."
    )
    user = f"""
query_id: {sample["query_id"]}

Query:
{sample["query"]}

SubAgent document reports:
{reports}

Return only a valid JSON object with exactly these keys:
{{
  "query_id": "{sample['query_id']}",
  "prediction": "final answer string",
  "support": "brief explanation tying the relevant reports to the answer"
}}

No markdown, no code fences, no extra text.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        default=Path("/home/tanger/workspace/datasets/browsecomp-plus-100/metadata.json"),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sub-max-tokens", type=int, default=1024)
    parser.add_argument("--sub-max-iters", type=int, default=4)
    parser.add_argument("--final-max-tokens", type=int, default=1024)
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
    trace: List[str] = ["main"]
    doc_reports: List[str] = []

    print(f"[sample] index={args.index} query_id={sample['query_id']} docs={len(sample['evidence_docs'])}")

    try:
        for doc_index, doc_path in enumerate(sample["evidence_docs"]):
            task = build_doc_task(sample["query_id"], sample["query"], doc_index, doc_path)
            sub_trace = trace + ["sub"]
            print(f"[subagent {doc_index}] document={doc_path}")
            sub_agent = Agent(
                name="sub",
                system_prompt=(
                    "You are a document-level research SubAgent. Use file tools to inspect "
                    "only the assigned document and extract compact query-relevant evidence."
                ),
                llm=client,
                is_main_agent=False,
                tools=build_file_tools(),
                max_iters=args.sub_max_iters,
                temperature=args.temperature,
                max_tokens=args.sub_max_tokens,
                trace=sub_trace,
            )
            result = sub_agent.run(task)
            doc_reports.append(result)
            trace.append("sub")
            if args.show_sub_results:
                print(f"[subagent {doc_index} result]")
                print(result)

        trace.append("main")
        final = client.chat(
            build_final_messages(sample, doc_reports),
            temperature=args.temperature,
            max_tokens=args.final_max_tokens,
            trace=trace,
        )
        answer = final.get("content") or ""
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
