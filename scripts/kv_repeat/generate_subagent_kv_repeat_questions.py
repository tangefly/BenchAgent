from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "subagent_kv_repeat_questions.jsonl"

TOKEN_BUCKETS = [64, 128, 256, 512, 1024]
CASES_PER_BUCKET = 40

# Token counts are tokenizer dependent. These character/line lower bounds are
# intentionally conservative so the generated text is unlikely to undershoot the
# target completion-token bucket on Qwen-style Chinese tokenization.
LENGTH_SPECS: Dict[int, Dict[str, int]] = {
    64: {"min_body_lines": 8, "min_chars_per_line": 24, "min_output_chars": 260},
    128: {"min_body_lines": 14, "min_chars_per_line": 28, "min_output_chars": 520},
    256: {"min_body_lines": 24, "min_chars_per_line": 34, "min_output_chars": 1000},
    512: {"min_body_lines": 44, "min_chars_per_line": 38, "min_output_chars": 2000},
    1024: {"min_body_lines": 88, "min_chars_per_line": 40, "min_output_chars": 4200},
}

TOPICS = [
    ("城市交通", "记录早高峰公交、地铁、骑行和步行之间的衔接细节"),
    ("图书馆管理", "记录借阅、归还、预约、盘点和读者服务的流程"),
    ("实验室安全", "记录试剂领取、设备检查、异常上报和清洁步骤"),
    ("仓库盘点", "记录入库、出库、批次号、货架位置和复核过程"),
    ("远程会议", "记录议题、发言顺序、决议、风险和待办事项"),
    ("校园餐厅", "记录备餐、排队、结算、补货和卫生检查情况"),
    ("软件发布", "记录版本冻结、构建、测试、灰度和回滚条件"),
    ("医院分诊", "记录挂号、初筛、候诊、检查和随访提醒"),
    ("社区活动", "记录报名、物资、场地、志愿者和值班安排"),
    ("天气观测", "记录温度、湿度、风向、降水和能见度变化"),
]

FORMATS = [
    "structured_lines",
    "key_value_records",
    "chinese_paragraph_lines",
    "mixed_symbols",
]

LINE_STYLE_HINTS = {
    "structured_lines": "每行写成 L001: 字段A=...；字段B=...；字段C=...；备注=...。",
    "key_value_records": "每行写成 rec_id=L001 | stage=... | actor=... | observation=... | next=...。",
    "chinese_paragraph_lines": "每行写成 L001: 一句完整中文陈述，包含时间、地点、对象、动作和结果。",
    "mixed_symbols": "每行写成 L001: tag=...; code=...; note=中文描述; status=OK/WAIT/RETRY。",
}


def build_prompt(case_id: int, target_tokens: int, topic: str, requirement: str, fmt: str) -> str:
    spec = LENGTH_SPECS[target_tokens]
    min_body_lines = spec["min_body_lines"]
    min_chars_per_line = spec["min_chars_per_line"]
    min_output_chars = spec["min_output_chars"]
    checksum_hint = f"KV-{case_id:03d}-{target_tokens}-{fmt}-{min_body_lines}L"
    return f"""请生成一段用于 KVCache 复述实验的 SubAgent 输出内容。目标是制造足够长、可逐字比对的文本。

硬性输出边界：
1. 只输出 BEGIN_SUBAGENT_OUTPUT 到 END_SUBAGENT_OUTPUT 之间的正文。
2. 第一行必须完全等于：BEGIN_SUBAGENT_OUTPUT
3. 最后一行必须完全等于：END_SUBAGENT_OUTPUT
4. 不要输出解释、标题、Markdown、代码围栏、字数统计或任何边界外文本。

硬性元数据行：
5. 第二行必须完全等于：case_id=KV_REPEAT_{case_id:03d}
6. 第三行必须完全等于：target_repeat_tokens={target_tokens}
7. 第四行必须完全等于：checksum_hint={checksum_hint}
8. 第五行必须完全等于：min_body_lines={min_body_lines}
9. 第六行必须完全等于：min_chars_per_line={min_chars_per_line}

硬性长度要求：
10. 在元数据行之后、END_SUBAGENT_OUTPUT 之前，必须至少输出 {min_body_lines} 条正文记录行。
11. 每条正文记录行必须以连续编号开头，从 L001 开始，到至少 L{min_body_lines:03d} 结束，不能跳号。
12. 每条正文记录行在编号之后必须至少包含 {min_chars_per_line} 个中文字符；数字、英文、标点和空格不计入这个中文字符下限。
13. BEGIN_SUBAGENT_OUTPUT 到 END_SUBAGENT_OUTPUT 之间的总正文长度必须至少约 {min_output_chars} 个字符；如果写到 L{min_body_lines:03d} 后明显不足，请继续写 L{min_body_lines + 1:03d}、L{min_body_lines + 2:03d} 等补充行，直到满足长度要求。
14. 不要因为无法精确计算 token 数而缩短输出；宁可略长，也不能明显短于 target_repeat_tokens={target_tokens} 对应规模。

内容要求：
15. 内容主题：{topic}，{requirement}。
16. 内容格式：{fmt}。格式提示：{LINE_STYLE_HINTS[fmt]}
17. 每行内容必须具体、稳定、可复述，包含明确实体、动作、数值或状态，避免文学化、押韵、随机闲聊和含糊代词。
18. 不要总结，不要分段说明，不要把多条记录压缩在同一行。

MainAgent 后续会被要求逐字复述 BEGIN_SUBAGENT_OUTPUT 到 END_SUBAGENT_OUTPUT 之间的全部内容，因此请严格保持编号、字段和值的清晰一致。""".strip()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    case_id = 1
    for target_tokens in TOKEN_BUCKETS:
        spec = LENGTH_SPECS[target_tokens]
        for _ in range(CASES_PER_BUCKET):
            topic, requirement = TOPICS[(case_id - 1) % len(TOPICS)]
            fmt = FORMATS[((case_id - 1) // len(TOPICS)) % len(FORMATS)]
            rows.append(
                {
                    "case_id": f"KV_REPEAT_{case_id:03d}",
                    "target_repeat_tokens": target_tokens,
                    "length_bucket": f"{target_tokens}_tokens",
                    "min_body_lines": spec["min_body_lines"],
                    "min_chars_per_line": spec["min_chars_per_line"],
                    "min_output_chars": spec["min_output_chars"],
                    "content_type": fmt,
                    "topic": topic,
                    "subagent_prompt": build_prompt(case_id, target_tokens, topic, requirement, fmt),
                }
            )
            case_id += 1

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
