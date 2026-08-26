from agent.llm import LLMClient
from agent.agent import Agent
from agent.tools import *

# 1. vLLM Client

client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3-8B",
    agent_mode=True
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

main_agent = Agent(name="main", system_prompt=system_prompt, llm=client, is_main_agent=True, max_tokens=10240, tools=build_subagent_tools())

task = """
There are five questions:

1. Who played the role of Ken Neville in *Alias – the Bad Man*?
2. How did Ken Neville gain the trust of Rance Collins in *Alias – the Bad Man*?
3. Why did Ken Neville initially hide his identity in the town?
4. What key contrast can be drawn between the fates of the two films described?
5. What common narrative arc do the two films share regarding the female lead's relationship with the hero?

Each question is based on `/home/tanger/workspace/BenchAgent/data/documents/doc1.txt`.

Please complete the tasks using the following agent invocation chain:

**main → sub → main → sub → ... → main**

Strict execution rules (hard constraints, do not deviate):

1. The main agent answers the five questions **strictly one by one**: call a SubAgent for
   question 1, wait for its answer, then call a SubAgent for question 2, and so on.
2. **Each reply may contain AT MOST ONE tool call.** Calling two or more SubAgents in a single
   reply is FORBIDDEN.
3. Never proceed to the next question before the previous SubAgent's answer has been returned
   to you.
4. The SubAgent should provide the final answer as concisely as possible. After completing all
   five questions, the main agent should summarize the answers to all five questions.

**Final Output Requirements:**

The main agent must return the final result as a valid JSON object with exactly the following structure:

```json
{
    "answer1": "Answer to question 1",
    "answer2": "Answer to question 2",
    "answer3": "Answer to question 3",
    "answer4": "Answer to question 4",
    "answer5": "Answer to question 5"
}
```

The keys must be exactly `answer1`, `answer2`, `answer3`, `answer4`, and `answer5`, corresponding to questions 1–5 respectively.

Do not include any additional keys, explanations, Markdown, or text outside the JSON object. The final response must contain only the JSON object.
"""

try:
    answer = main_agent.run(task)
finally:
    # 整个任务结束后再释放会话 KV: 主/子 agent 共用同一 client,
    # 中途释放会清空会话 KV 段, 使 --reuse-agent-kv-append 失效
    client.release_kv()

print("[answer]")
print(answer)
