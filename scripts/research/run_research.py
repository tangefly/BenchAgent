#!/usr/bin/env python3
"""Iterative BrowseComp research, searching only each sample's evidence_docs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
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
    # 选样本: 下标与 query_id 二选一(同时给会互相覆盖语义, 直接拒绝)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--index", type=int, default=None, help="First sample index (default 0)")
    select.add_argument("--query-id", default=None,
                        help="Select the sample whose metadata query_id matches (exact string compare)")
    parser.add_argument("--limit", type=positive, default=1)
    parser.add_argument("--all", action="store_true",
                        help="Run all samples starting at --index/--query-id")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--model-name", default=None,
                        help="结果文件名里用的模型名(默认取 --model, 非法文件名字符替换成 -)")
    parser.add_argument("--agent-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Use --no-agent-mode for a plain OpenAI-compatible backend")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-kv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--engine", choices=("fast", "legacy"), default="fast")
    parser.add_argument("--log-level", choices=("basic", "full"), default="basic",
                        help="Console logs: basic progress/results (default), or full model/tool details; JSONL events remain complete")
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
    parser.add_argument("--compact-output", type=Path, default=None,
                        help="精简版(每样本一行, 只留判定字段)输出路径, 默认与完整版同名加 .compact")
    return parser.parse_args()


def sample_paths(sample, metadata):
    """Resolve only explicitly listed evidence; never index metadata/gold or sibling samples."""
    paths = sample["evidence_docs"]
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError("evidence_docs must be a nonempty list of paths")
    return [Path(p) if Path(p).is_absolute() else metadata.resolve().parent / p for p in paths]


def start_index(samples, index, query_id):
    """--index / --query-id -> 起始下标. query_id 必须唯一命中, 否则直接报错."""
    if query_id is not None:
        hits = [i for i, s in enumerate(samples)
                if isinstance(s, dict) and str(s.get("query_id")) == str(query_id)]
        if len(hits) != 1:
            raise ValueError(f"query_id {query_id!r} 匹配到 {len(hits)} 个样本, 需要恰好 1 个")
        return hits[0]
    return 0 if index is None else index


def default_output_path(model_name):
    """结果文件名: results_<模型名>_<时间戳>.jsonl(模型名里的非法文件名字符换成 -)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name)).strip("-") or "model"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ROOT / "outputs" / "research" / f"results_{safe}_{stamp}.jsonl"


# 精简版只留人工核对要看的字段: 完整版的 events/reports/evidence/sources 常有几百 KB
COMPACT_FIELDS = ("index", "query_id", "status", "prediction", "gold", "metrics",
                  "requests", "gaps", "error")


def compact_row(row):
    """一条样本的精简视图(每样本一行, 便于 jq/grep 和人眼扫)."""
    return {k: row[k] for k in COMPACT_FIELDS if k in row}


def run(args):
    samples = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("metadata must be an array")
    start = start_index(samples, args.index, args.query_id)
    if not 0 <= start < len(samples):
        raise ValueError("index must be in range")
    if args.query_id is not None:
        print(f"[SELECT] query_id={args.query_id} -> index={start}", flush=True)
    stop = len(samples) if args.all else min(len(samples), start + args.limit)
    output = args.output or default_output_path(args.model_name or args.model)
    compact_output = args.compact_output or output.with_suffix(".compact.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(start, stop):
        sample = samples[index]
        store = None
        client = LLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model, timeout=args.timeout,
                           agent_mode=args.agent_mode, enable_thinking=args.enable_thinking)
        engine = None
        try:
            # Both retrieval state and LLM session are fresh for every sample.
            store = SourceStore(sample_paths(sample, args.metadata))
            print(f"[SAMPLE] index={index} query_id={sample['query_id']} documents={len(store.sources)}", flush=True)
            if args.engine == "fast":
                engine = FastResearchEngine(client, store, max_followups=args.max_followups,
                                            packet_chars=args.packet_chars, max_tokens=args.max_tokens,
                                            temperature=args.temperature, max_sub_tool_rounds=args.sub_tool_rounds,
                                            max_main_turns=args.max_main_turns, max_requests=args.max_requests,
                                            log_level=args.log_level)
            else:
                engine = ResearchEngine(client, store, max_requests=args.max_requests,
                                        max_main_turns=args.max_main_turns, max_worker_turns=args.max_worker_turns,
                                        max_tokens=args.max_tokens, temperature=args.temperature,
                                        log_level=args.log_level)
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
        with compact_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_row(row), ensure_ascii=False) + "\n")
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k in {"index", "status", "prediction", "error", "metrics"}}, ensure_ascii=False))
    summary = {"metadata": str(args.metadata), "model": args.model,
               "model_name": args.model_name or args.model,
               "num_requested": len(rows),
               "num_completed": sum("error" not in r for r in rows),
               "num_errors": sum("error" in r for r in rows),
               "num_answered": sum(r.get("status") == "answered" for r in rows), "metrics": mean_scores(rows)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[output] {output}")
    print(f"[compact_output] {compact_output}")


if __name__ == "__main__":
    run(parse_args())
