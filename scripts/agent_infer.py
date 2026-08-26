from agent.llm import LLMClient
from agent.agent import Agent
from agent.tools import *

# 1. vLLM Client

client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3-8B"
)

system_prompt = "You are an AI agent capable of solving complex tasks. You can independently plan, execute tasks step by step, and verify the results. You have access to tools for calling SubAgents, allowing you to decompose a task into multiple subtasks and delegate them to SubAgents. SubAgents have access to common file operations, such as file reading and file searching."

main_agent = Agent(name="main", system_prompt=system_prompt, llm=client, max_tokens=10240, tools=build_subagent_tools())

# task = "/home/tanger/workspace/BenchAgent/data/longbench.jsonl 里面含有 5 个问题，每个问题都能从所给的 file_path 文档中找到答案，请你依次调用 SubAgent，将 5 个任务逐个求解，逐个派发给 SubAgent，最后做一次答案汇总"

# task = "列出 /home/tanger/workspace/BenchAgent/data 下的所有文件"

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

The main agent should call a SubAgent to answer each question sequentially, one at a time. The SubAgent should provide the final answer as concisely as possible. After completing all five questions, the main agent should summarize the answers to all five questions.

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

answer = main_agent.run(task)

print("[answer]")
print(answer)
