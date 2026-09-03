from agent.llm import LLMClient
from agent.agent import Agent
from agent.tools import *

client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3-8B",
    agent_mode=True,
    enable_thinking=True,
)

system_prompt = (
    "You are a writing coordinator. When the user asks you to delegate a writing "
    "task, call the SubAgent once and give it clear requirements. After the SubAgent "
    "returns, forward that returned text verbatim. Do not summarize, polish, correct, "
    "or reformat it."
)

main_agent = Agent(
    name="main",
    system_prompt=system_prompt,
    llm=client,
    is_main_agent=True,
    max_tokens=10240,
    tools=build_subagent_tools(),
)

task = """
Please delegate the following writing task to exactly one SubAgent, then forward the SubAgent's response verbatim as your final answer.

SubAgent task:
写一篇中文小作文，题目是《雨天》。内容可以描写雨声、街道、行人、心情和雨后的变化。文章要自然、完整，有开头和结尾，长度约 600 到 800 个中文字符。请直接输出作文正文，不要写提纲、说明或额外注释。

MainAgent final answer requirements:
- You must call exactly one SubAgent before answering.
- Your final answer must contain only the SubAgent's returned composition.
- Do not summarize, polish, correct, shorten, or rephrase the SubAgent output.
- Do not add markdown, code fences, labels, explanations, or any text before or after it.
- Preserve the SubAgent's line breaks and wording as much as possible.
"""
try:
    answer = main_agent.run(task)
finally:
    # client.release_kv()
    pass

print("[answer]")
print(answer)
