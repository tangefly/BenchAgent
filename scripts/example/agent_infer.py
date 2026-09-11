"""MainAgent -> SubAgent 串行调用 demo(LMInfer agent 模式 / 跨请求 KV 复用).

与 multi_subagent_kv_reuse.py 的区别: 这里主 agent 严格**一次只调用一个子 agent**,
形成 MainAgent → SubAgent → MainAgent → ... 的长链, 用来观察子 agent 输出 KV 在
主 agent 后续请求里的复用(服务端 --reuse-agent-kv / --reuse-agent-kv-append)。

用法(先启动服务):
  lminfer serve /home/tanger/workspace/models/Qwen3-8B \
      --served-model-name Qwen3-8B --reuse-agent-kv --port 8000
  python scripts/example/agent_infer.py --model Qwen3-8B

  # Ministral-3(FP8 多模态 checkpoint, 工具调用是 [TOOL_CALLS]name[ARGS]{json})
  lminfer serve /home/tanger/workspace/models/Ministral-3-8B-Instruct-2512 \
      --served-model-name Ministral-3-8B --max-model-len 40960 \
      --reuse-agent-kv-append --graft-rope-rebase \
      --repair-window-begin 0.1 --repair-window-end 0.1 \
      --enable-auto-tool-choice --port 8000
  python scripts/example/agent_infer.py --model Ministral-3-8B
"""
from __future__ import annotations

import argparse

from agent.llm import LLMClient
from agent.agent import Agent
from agent.tools import *

ALL_QUESTIONS = [
    "Who played the role of Ken Neville in *Alias – the Bad Man*?",
    "How did Ken Neville gain the trust of Rance Collins in *Alias – the Bad Man*?",
    "Why did Ken Neville initially hide his identity in the town?",
    "What key contrast can be drawn between the fates of the two films described?",
    "What common narrative arc do the two films share regarding the female lead's relationship with the hero?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MainAgent -> SubAgent 串行调用 demo")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3-8B",
                        help="必须与服务端 --served-model-name 一致; "
                             "Ministral-3 填 Ministral-3-8B")
    parser.add_argument("--document",
                        default="/home/tanger/workspace/BenchAgent/data/documents/doc1.txt")
    parser.add_argument("--questions", type=int, default=len(ALL_QUESTIONS),
                        help="只跑前 N 个问题(调试用), 默认全部")
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args()


def build_task(file_path: str, questions: list[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    keys = "\n".join(f'"answer{i}": "Answer to question {i}"'
                     for i in range(1, len(questions) + 1))
    return f"""
There are {len(questions)} questions:

{numbered}

All questions must be answered using the information contained in:

`{file_path}`

You are the MainAgent. You must process the questions through a strict sequential:

MainAgent → SubAgent → MainAgent → SubAgent → ... → MainAgent

execution chain.

Hard execution constraints:

1. Process the questions strictly in order: Q1 → Q2 → ... .

2. For each question, the MainAgent MUST invoke exactly one SubAgent and MUST wait for that SubAgent's response before processing the next question.

3. Each assistant reply may contain AT MOST ONE tool call. NEVER invoke multiple SubAgents in the same reply.

4. Every SubAgent invocation MUST explicitly include:

   * the current question;
   * the document path:
     `{file_path}`

5. The document path MUST be passed to the SubAgent on EVERY invocation. Do not rely on previous conversation context or previous SubAgent calls.

6. The SubAgent MUST use the specified document as the source for answering the current question. It must not assume that the document content has already been provided.

7. The SubAgent MUST answer ONLY the current question. It must not answer future questions or invoke another agent.

8. The SubAgent response MUST be concise. Give only the information necessary to answer the current question. Avoid unnecessary explanation, background, reasoning, repetition, or restatement of the question.

9. The MainAgent must retain the returned answer and then proceed to the next question only after the SubAgent response has been received.

10. SubAgent calls MUST be strictly sequential and MUST NOT be parallelized.

11. Do not skip any question or answer a question directly without first invoking its SubAgent.

12. The logical execution sequence MUST be:

Q1:
MainAgent → SubAgent(question=Q1, document_path=...) → answer1

Q2:
MainAgent → SubAgent(question=Q2, document_path=...) → answer2

... (repeat for every question)

Final Answer Requirements:

After all SubAgent calls are completed, the MainAgent MUST return the answers in a valid JSON object.

The final answer MUST be extremely concise. Each answer should contain only the minimum information needed to correctly answer its corresponding question. Do not include unnecessary explanations, reasoning, background information, or repeated context.

The final response MUST contain exactly these {len(questions)} keys:

{{
{keys}
}}

The keys MUST correspond exactly to Q1–Q{len(questions)}.

The final response must contain ONLY the JSON object.

Do NOT include:

* Markdown
* Code fences
* Explanations
* Reasoning
* Additional keys
* Additional text before or after the JSON
* Unnecessary details in any answer

Keep every answer as short as possible while preserving correctness.
"""


def main() -> None:
    args = parse_args()

    # 1. LMInfer / vLLM 客户端: agent_mode=True 时客户端自动维护 session_id,
    #    服务端据此把 main/sub 的多次调用关联到同一会话, 才能跨请求复用 KV
    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        agent_mode=True,
        timeout=args.timeout,
    )

    system_prompt = (
        "You are an AI agent capable of solving complex tasks. You can independently plan, execute "
        "tasks step by step, and verify the results. You have access to tools for calling SubAgents, "
        "allowing you to decompose a task into multiple subtasks and delegate them to SubAgents. "
        "SubAgents have access to common file operations, such as file reading and file searching.\n"
        "\n"
        "**Tool call discipline (hard constraint):** In EVERY reply, you must invoke AT MOST ONE "
        "tool call. If you need to call a SubAgent several times, do it strictly one at a time: "
        "call the first SubAgent, wait for its returned result, then in a SEPARATE reply call the "
        "next one. Never include two or more tool calls in the same reply."
    )

    main_agent = Agent(name="main", system_prompt=system_prompt, llm=client,
                       is_main_agent=True, max_tokens=10240,
                       tools=build_subagent_tools())

    questions = ALL_QUESTIONS[:args.questions]
    task = build_task(args.document, questions)

    try:
        answer = main_agent.run(task)
    finally:
        pass
        # client.release_kv()

    print("[answer]")
    print(answer)
    # agent 模式观测字段: 最后一次请求跳过 prefill 的 token 数
    # (含拼接进来的子 agent 输出 KV); 恒为 0 说明服务端没开 --reuse-agent-kv*
    print(f"[reused_prompt_tokens] {client.last_reused_tokens}")
    print(f"[usage] {client.last_usage}")


if __name__ == "__main__":
    main()
