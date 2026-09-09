#!/usr/bin/env python3
"""Iterative BrowseComp research, searching only each sample's evidence_docs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import LLMClient
from agent.research import ResearchEngine, SourceStore
from agent.research_fast import FastResearchEngine
from scripts.browsecomp.run_browsecomp import score_prediction, mean_scores


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path("/home/tanger/workspace/datasets/browsecomp-plus-100/metadata.json"))
    parser.add_argument("--index", type=int, default=0, help="First sample index")
    parser.add_argument("--limit", type=positive, default=1)
    parser.add_argument("--all", action="store_true", help="Run all samples starting at --index")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--agent-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Use --no-agent-mode for a plain OpenAI-compatible backend")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-kv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--engine", choices=("fast", "legacy"), default="fast")
    parser.add_argument("--max-followups", type=int, default=5, help="Targeted single-document reinspections after every document is analyzed (fast engine)")
    parser.add_argument("--sub-tool-rounds", type=int, default=2, help="Sub tool-round cap per task; default 2, 0 uses supplied passages only")
    parser.add_argument("--packet-chars", type=positive, default=32000)
    parser.add_argument("--max-requests", type=positive, default=60)
    parser.add_argument("--max-main-turns", type=positive, default=24)
    parser.add_argument("--max-worker-turns", type=positive, default=8)
    parser.add_argument("--max-tokens", type=positive, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sample_paths(sample, metadata):
    """Resolve only explicitly listed evidence; never index metadata/gold or sibling samples."""
    paths = sample["evidence_docs"]
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError("evidence_docs must be a nonempty list of paths")
    return [Path(p) if Path(p).is_absolute() else metadata.resolve().parent / p for p in paths]


def run(args):
    samples = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not 0 <= args.index < len(samples):
        raise ValueError("metadata must be an array and index must be in range")
    stop = len(samples) if args.all else min(len(samples), args.index + args.limit)
    output = args.output or ROOT / "outputs" / "research" / datetime.now().strftime("results_%Y%m%d_%H%M%S_%f.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.index, stop):
        sample = samples[index]
        store = None
        client = LLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model, timeout=args.timeout,
                           agent_mode=args.agent_mode, enable_thinking=args.enable_thinking)
        engine = None
        try:
            # Both retrieval state and LLM session are fresh for every sample.
            store = SourceStore(sample_paths(sample, args.metadata))
            if args.engine == "fast":
                engine = FastResearchEngine(client, store, max_followups=args.max_followups,
                                            packet_chars=args.packet_chars, max_tokens=args.max_tokens,
                                            temperature=args.temperature, max_sub_tool_rounds=args.sub_tool_rounds,
                                            max_main_turns=args.max_main_turns, max_requests=args.max_requests)
            else:
                engine = ResearchEngine(client, store, max_requests=args.max_requests,
                                        max_main_turns=args.max_main_turns, max_worker_turns=args.max_worker_turns,
                                        max_tokens=args.max_tokens, temperature=args.temperature)
            result = engine.run(sample["query"])
            row = {"index": index, "query_id": sample["query_id"], **result,
                   "source_mode": "sample_evidence_docs", "prediction": result["answer"]}
            # Gold is used only after research, never included in prompts or the index.
            row.update(gold=sample["answer"], metrics=score_prediction(result["answer"], sample["answer"]))
        except Exception as exc:
            row = {"index": index, "query_id": sample.get("query_id"), "error": repr(exc),
                   "events": engine.events if engine else [],
                   "evidence": (list(engine.evidence.values()) if isinstance(engine.evidence, dict) else engine.evidence) if engine else []}
        finally:
            if store is not None:
                store.db.close()
            if args.release_kv and client.session_id:
                try:
                    client.release_kv()
                except Exception as exc:
                    print(f"[release warning] {exc}", file=sys.stderr)
            client.session.close()
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k in {"index", "status", "prediction", "error", "metrics"}}, ensure_ascii=False))
    summary = {"metadata": str(args.metadata), "num_requested": len(rows),
               "num_completed": sum("error" not in r for r in rows),
               "num_errors": sum("error" in r for r in rows),
               "num_answered": sum(r.get("status") == "answered" for r in rows), "metrics": mean_scores(rows)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[output] {output}")


if __name__ == "__main__":
    run(parse_args())
