"""Minimal test to identify the issue."""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

async def main():
    print("Creating model...", flush=True)
    model = OpenAIModel(
        model_name='qwen3:0.6b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    
    print("Creating agent...", flush=True)
    agent = Agent(model=model)
    
    print("Running query...", flush=True)
    result = await agent.run('Say hello')
    print(f"Result: {result.output}", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
