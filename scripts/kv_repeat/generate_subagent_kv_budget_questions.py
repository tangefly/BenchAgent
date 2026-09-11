"""Generate paired deterministic tasks; only the sub output budget varies."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def build_rows(seeds=(17, 29), budgets=(64, 128, 256, 512, 1024)):
    for seed in seeds:
        task = (
            "执行确定性记录计算任务，不使用工具，不解释，不输出标题或代码围栏。\n"
            f"对 i 从 1 到 512 按顺序逐条计算 v=(i*37+{seed})%1000。\n"
            "每条只输出一行，格式为 Rxxx|value=yyy，其中 xxx 是 i 的三位十进制数，"
            "yyy 是 v 的三位十进制数，不足三位补零。\n"
            "输出全部512条记录，不省略、不汇总。"
        )
        for budget in budgets:
            yield {
                "case_id": f"KV_BUDGET_{seed}_{budget}",
                "length_bucket": f"{budget}_tokens",
                "target_repeat_tokens": budget,
                "content_type": "deterministic_records",
                "topic": "modular_arithmetic",
                "seed": seed,
                "record_count": 512,
                "subagent_prompt": task,
            }


if __name__ == "__main__":
    out = ROOT / "data" / "subagent_kv_budget_questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(build_rows())
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows to {out}")
