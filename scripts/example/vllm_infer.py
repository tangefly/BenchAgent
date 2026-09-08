from openai import OpenAI
import json

from agent.llm import LLMClient

# 1. vLLM Client

client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen3-8B"
)

# 2. Define actual Python tools

def get_weather(city: str):
    """
    实际执行工具的 Python 函数。
    这里模拟一个天气 API。
    """

    weather_data = {
        "Tokyo": {
            "temperature": 28,
            "weather": "Sunny",
        },
        "Beijing": {
            "temperature": 30,
            "weather": "Cloudy",
        },
        "Shanghai": {
            "temperature": 31,
            "weather": "Rainy",
        },
    }

    return weather_data.get(
        city,
        {
            "temperature": None,
            "weather": "Unknown",
        },
    )


# 工具名 -> Python 函数
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}

# 3. Tool schema

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather information for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# 4. Initial conversation

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Use tools when necessary."
        ),
    },
    {
        "role": "user",
        "content": "What's the weather like in Tokyo?",
    },
]

# 5. First inference

response = client.chat(
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.0,
    trace=["main"]
)

client.release_kv()

print(response)