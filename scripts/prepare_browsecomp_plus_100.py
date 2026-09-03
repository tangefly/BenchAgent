#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "unknown"


def write_dataset(input_path: Path, output_dir: Path, limit: int) -> None:
    evidence_root = output_dir / "evidence_docs"
    metadata_path = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)

    records = []
    with input_path.open("r", encoding="utf-8") as source:
        for row_idx, line in enumerate(source):
            if row_idx >= limit:
                break
            if not line.strip():
                continue

            item = json.loads(line)
            query_id = str(item["query_id"])
            query_dir = evidence_root / safe_name(query_id)
            query_dir.mkdir(parents=True, exist_ok=True)

            evidence_paths = []
            for doc_idx, doc in enumerate(item.get("evidence_docs", [])):
                docid = safe_name(doc.get("docid", f"doc_{doc_idx:04d}"))
                text_path = query_dir / f"{doc_idx:04d}_{docid}.txt"
                text_path.write_text(doc.get("text", ""), encoding="utf-8")
                evidence_paths.append(str(text_path))

            records.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "answer": item["answer"],
                    "evidence_docs": evidence_paths,
                }
            )

    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/tanger/workspace/datasets/browsecomp-plus/decrypted.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/tanger/workspace/datasets/browsecomp-plus-100"),
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    write_dataset(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
