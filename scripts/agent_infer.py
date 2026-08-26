from agent.llm import LLMClient
from agent.agent import Agent
from agent.tools import *

# 1. vLLM Client

client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3-8B"
)

system_prompt = "你是一个 AI 智能体，具备解决复杂任务的能力，能够自主规划，逐步完成任务，并且验证结果。"
main_agent = Agent(name="main", system_prompt=system_prompt, llm=client, max_tokens=10240, tools=build_subagent_tools())
main_agent.run("计算 100 以内的素数。")
