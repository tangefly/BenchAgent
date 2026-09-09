# BrowseComp 本地研究：精简版

运行入口：`scripts/research/run_research.py`，默认引擎：`agent/research_fast.py`。

只读取 `metadata.json` 当前条目的 `evidence_docs`，不联网、不扫描其他文档。每条样本新建索引和 LLM 会话。标准答案仅在研究完成后评分，不进入模型上下文。

## 为什么替换旧流程

旧研究引擎把计划、搜索、阅读、逐条记录、查询证据库和审查都拆成模型调用。实际失败日志显示，模型反复在 `record_evidence` 中输出额外字段，被严格校验拒绝，60 次请求仅保存两条证据。对已经给定少量候选文档的任务，这种流程开销过高。

精简版把本地数据处理交给代码，模型负责取证与综合：

1. main 拆解问题条件，每轮通过 `research(query, source_ids)` 只分派一篇文档。`source_ids` 必须只有一个来源 ID；省略时程序选择下一篇未处理文档。main 不能把全部文档交给同一个 sub。
2. 程序准备该文档的阅读材料：短文档完整提供，长文档提供标题及检索命中的片段。
3. sub 返回与问题相关的事实、原文引用和缺口，可用 `search_documents` 和 `read_passages` 补查，但程序禁止访问其他文档。默认最多两轮工具交互，之后返回 findings/gaps，由 main 接手。引用按空白归一化后检查是否来自提供的片段；校验失败时把该文档的源材料交给 main 判断。
4. main 接收结果后给出简短阶段判断，继续安排下一篇未处理文档。每轮保留分派工具；同一回复仅执行第一个分派，后续调用须等 main 读完结果再决定。
5. 所有文档都返回结果后，main 才能给出最终答案，或针对某一篇文档定向复查。默认最多额外复查五次；不同问题可以使用相同材料，相同任务和材料的重复请求会被拦截。请求或 main 轮数预算耗尽而尚未形成最终答案时，返回 `insufficient` 并保留证据和事件。

两篇文档的典型调用链是 `main -> sub(S1) -> main -> sub(S2) -> main(final)`。每次 sub 都是新的单文档任务上下文，main 保留历次结果并负责跨文档综合。只有一篇文档时允许一个 sub 返回后直接综合。sub 内部搜索/阅读不会改变 trace，返回 main 时才追加 main。字符预算限制输入规模，但不等于 tokenizer 的精确 token 数。

## 运行

`--log-level basic` 为默认值，控制台只保留样本进度、任务分派、sub 提取的事实与缺口、错误和最终答案。
`--log-level full` 额外显示每次模型请求、main 阶段分析、完整工具参数/结果、sub 原始输出及完整处理结果。
两种模式均保留完整 JSONL 事件记录，不改变模型请求、证据提取或评分。fast 和 legacy 引擎均支持此参数。

```bash
python3 scripts/research/run_research.py --index 10 --log-level basic
python3 scripts/research/run_research.py --index 10 --log-level full
```

```bash
# 默认 metadata: /home/tanger/workspace/datasets/browsecomp-plus-100/metadata.json
python3 scripts/research/run_research.py --index 0

# 连续 5 条 / 全部样本
python3 scripts/research/run_research.py --index 0 --limit 5
python3 scripts/research/run_research.py --all

# 每篇文档各分析一次，全部完成后综合，不额外复查
python3 scripts/research/run_research.py --index 0 --max-followups 0

# 默认每个 sub 最多两轮工具交互，随后交回 main
python3 scripts/research/run_research.py --index 0
# 可选：恢复不调用 sub 工具的提取模式
python3 scripts/research/run_research.py --index 0 --sub-tool-rounds 0

# 增加阅读材料的字符预算
python3 scripts/research/run_research.py --index 0 --packet-chars 48000

# 普通 OpenAI 兼容服务
python3 scripts/research/run_research.py --index 0 --no-agent-mode \
  --base-url http://localhost:8000/v1 --model Qwen3-8B

# 仅用于对照：原来的多工具研究引擎
python3 scripts/research/run_research.py --index 0 --engine legacy
```

精简引擎使用 `--max-followups`（默认 5，即全部文档处理完后最多额外复查 5 次）、`--sub-tool-rounds`（默认 2）、`--packet-chars`（默认 32000）、`--max-tokens`（默认 4096）。总分派预算为文档数量加复查次数。`--max-requests`（默认 60）和 `--max-main-turns`（默认 24）对两个引擎均生效；`--max-worker-turns` 仅影响 legacy 引擎。文档较多时需要相应调大请求及 main 轮数预算。

`--release-kv` 在每条样本结束时释放其服务端 KV。实际缓存复用量由 LMInfer 实现决定，以输出中的统计为准。

## 输出和限制

每条 JSONL 记录包括 `prediction`、`gold`、`metrics`、`status`、`gaps`、`synthesis`、引用来源 `citations`、提取证据 `evidence`、材料覆盖范围 `reports`、`trace` 和逐轮 `events`（包含 usage、复用 token、耗时）。同时生成汇总 `.summary.json`。

`answered` 表示模型报告没有未解决缺口，**不是程序证明答案正确**。`partial` 保留候选答案和缺口，仍正常评分；没有候选答案为 `insufficient`。程序仅校验引用是否属于所给材料，不能保证事实推导和实体连接正确。精简版取消了强制独立核查，准确率和效率需要联合评测。

搜索采用 SQLite FTS5/BM25 词汇检索，不是向量检索。每轮只有当前文档使用阅读配额，避免非英文短文档因英语关键词无法命中而被漏掉。长文档仍可能遗漏相关内容；main 可用本地名称、别名和关键词请求针对性补查。

旧 `run_browsecomp.py`、`run_browsecomp_evidence.py` 及通用 Agent 未修改。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## sub 原生工具

- `search_documents(queries, source_ids?, limit?)`：自主决定一次提交的查询数量；来源被限定为当前分配文档。返回原文片段、来源 ID 和字符范围，sub 可直接引用，无须再调用读取工具。
- `read_passages(passages)`：自主决定一次读取的指定来源片段数量，每项包含 `source_id`、可选 `start` 和 `length`。只能读取当前分配文档，不接受其他来源或文件路径。

同一回复中的所有 sub 工具调用都会执行，批量数组也不会截取前几项。`limit` 由 sub 决定每个查询返回多少个命中；默认 3，但没有硬上限。读取长度按请求执行，工具总结果不再按 12000 字符截断。完全相同的调用使用缓存返回，不强制 sub 收尾。默认 `--sub-tool-rounds 2`，达到上限后要求 sub 返回已有发现和缺口；可调大该值，但总请求数仍受 `--max-requests` 限制。工具得到的新片段进入引用校验与 coverage 记录。

例如模型可生成原生 tool call 参数：

```json
{"queries": ["graduation June 2003", "Sana'a capital"], "source_ids": ["S2", "S6"], "limit": 3}
```

事件日志中 `role=researcher` 且 `event=tool` 的记录包含工具名、参数、结果和 trace。控制台会显示 `[research-fast tool] sub search_documents passages=...`。

控制台额外显示 `[main analysis]`（模型可见的阶段判断）、`[main -> sub #N]`（子任务）和 `[sub #N -> main]`（返回结果）。JSONL 的 `events` 对应保存 `main_decision`、`delegation`、`sub_result` 事件；`reports` 包含 `task_id`、`task`、`document_id` 和 `remaining_documents`，最终结果还包含 `processed_sources` 与 `remaining_sources`，可以逐文档追踪。阶段判断是模型输出的简短说明，不保证每轮模型都会生成；程序保证每次分派结束后先回到 main。

文档已处理表示该文档的 sub 返回了分析结果，不代表长文的每个字符都已读取；实际阅读范围见 `coverage`。长文缺口仍需 main 发起针对该文档的复查。
