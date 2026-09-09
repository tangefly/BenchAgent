# BrowseComp 本地研究：精简版

运行入口：`scripts/research/run_research.py`，默认引擎：`agent/research_fast.py`。

只读取 `metadata.json` 当前条目的 `evidence_docs`，不联网、不扫描其他文档。每条样本新建索引和 LLM 会话。标准答案仅在研究完成后评分，不进入模型上下文。

## 为什么替换旧流程

旧研究引擎把计划、搜索、阅读、逐条记录、查询证据库和审查都拆成模型调用。实际失败日志显示，模型反复在 `record_evidence` 中输出额外字段，被严格校验拒绝，60 次请求仅保存两条证据。对已经给定少量候选文档的任务，这种流程开销过高。

精简版把本地数据处理交给代码，模型负责取证与综合：

1. main 调用一次 `research`，首次自动覆盖当前样本所有文档。
2. 程序准备阅读材料：短文档完整提供，长文档提供标题及检索命中的片段。没有模型侧逐页读取和逐条录入操作。
3. sub 一次返回跨文档的一组事实及原文引用；额外 JSON 字段会被忽略。引用按空白归一化后检查是否来自提供的片段，校验失败则直接把源材料交给 main 判断，不让 sub 反复修复。
4. main 综合并输出答案。需要更多信息时，在 JSON 中返回 `follow_up`（目标问题、关键词、来源 ID），程序再执行一次 sub 提取。重复的阅读材料不会再调用 sub。
5. 默认最多两次补查。取消强制独立 reviewer 和“所有条件全部登记才能回答”的门槛；main 必须说明跨文档关系、反例和未解决条件。保留有证据支持的候选答案，标记 `partial`，不因某个条件尚不明确就清空预测。

正常无补查路径为 3 次模型请求：`main -> sub -> main`；一次补查为 5 次，两次为 7 次。异常格式/未正常收尾时可能增加一次终结请求，默认上限 8 次，不再是 60 次。字符预算限制输入规模，但不等于 tokenizer 的精确 token 数。

## 运行

```bash
# 默认 metadata: /home/tanger/workspace/datasets/browsecomp-plus-100/metadata.json
python3 scripts/research/run_research.py --index 0

# 连续 5 条 / 全部样本
python3 scripts/research/run_research.py --index 0 --limit 5
python3 scripts/research/run_research.py --all

# 禁止补查，只做一轮跨文档提取和综合
python3 scripts/research/run_research.py --index 0 --max-followups 0

# 增加阅读材料的字符预算
python3 scripts/research/run_research.py --index 0 --packet-chars 48000

# 普通 OpenAI 兼容服务
python3 scripts/research/run_research.py --index 0 --no-agent-mode \
  --base-url http://localhost:8000/v1 --model Qwen3-8B

# 仅用于对照：原来的多工具研究引擎
python3 scripts/research/run_research.py --index 0 --engine legacy
```

精简引擎使用 `--max-followups`（默认 2）、`--packet-chars`（默认 32000）、`--max-tokens`（默认 4096）。`--max-requests`、`--max-main-turns`、`--max-worker-turns` 只影响 legacy 引擎。

`--release-kv` 在每条样本结束时释放其服务端 KV。实际缓存复用量由 LMInfer 实现决定，以输出中的统计为准。

## 输出和限制

每条 JSONL 记录包括 `prediction`、`gold`、`metrics`、`status`、`gaps`、`synthesis`、引用来源 `citations`、提取证据 `evidence`、材料覆盖范围 `reports`、`trace` 和逐轮 `events`（包含 usage、复用 token、耗时）。同时生成汇总 `.summary.json`。

`answered` 表示模型报告没有未解决缺口，**不是程序证明答案正确**。`partial` 保留候选答案和缺口，仍正常评分；没有候选答案为 `insufficient`。程序仅校验引用是否属于所给材料，不能保证事实推导和实体连接正确。精简版取消了强制独立核查，准确率和效率需要联合评测。

搜索采用 SQLite FTS5/BM25 词汇检索，不是向量检索。首轮每份文档都有阅读配额，避免非英文短文档因英语关键词无法命中而被漏掉。长文档仍可能遗漏相关内容；main 可用本地名称、别名和关键词请求针对性补查。

旧 `run_browsecomp.py`、`run_browsecomp_evidence.py` 及通用 Agent 未修改。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
