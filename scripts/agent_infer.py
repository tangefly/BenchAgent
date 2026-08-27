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

file_path = "/home/tanger/workspace/BenchAgent/data/documents/doc1.txt"

task = f"""
There are five questions:

1. Who played the role of Ken Neville in *Alias – the Bad Man*?
2. How did Ken Neville gain the trust of Rance Collins in *Alias – the Bad Man*?
3. Why did Ken Neville initially hide his identity in the town?
4. What key contrast can be drawn between the fates of the two films described?
5. What common narrative arc do the two films share regarding the female lead's relationship with the hero?

All five questions must be answered using the information contained in:

`{file_path}`

You are the MainAgent. You must process the five questions through a strict sequential:

MainAgent → SubAgent → MainAgent → SubAgent → ... → MainAgent

execution chain.

Hard execution constraints:

1. Process the questions strictly in order: Q1 → Q2 → Q3 → Q4 → Q5.

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

Q3:
MainAgent → SubAgent(question=Q3, document_path=...) → answer3

Q4:
MainAgent → SubAgent(question=Q4, document_path=...) → answer4

Q5:
MainAgent → SubAgent(question=Q5, document_path=...) → answer5

Final Answer Requirements:

After all five SubAgent calls are completed, the MainAgent MUST return the five answers in a valid JSON object.

The final answer MUST be extremely concise. Each answer should contain only the minimum information needed to correctly answer its corresponding question. Do not include unnecessary explanations, reasoning, background information, or repeated context.

The final response MUST contain exactly these five keys:

{{
"answer1": "Answer to question 1",
"answer2": "Answer to question 2",
"answer3": "Answer to question 3",
"answer4": "Answer to question 4",
"answer5": "Answer to question 5"
}}

The keys MUST correspond exactly to Q1–Q5.

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

try:
    answer = main_agent.run(task)
finally:
    pass
    client.release_kv()

print("[answer]")
print(answer)
