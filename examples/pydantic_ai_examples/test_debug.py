"""Debug test with timeout."""

import asyncio
import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

async def main():
    print("Creating httpx client with 5 second timeout...", flush=True)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    
    print("Creating OllamaProvider with custom http_client...", flush=True)
    provider = OllamaProvider(
        base_url='http://localhost:11434/v1',
        http_client=http_client
    )
    
    print("Creating OpenAIModel...", flush=True)
    model = OpenAIModel(
        model_name='qwen3:8b',
        provider=provider
    )
    
    print("Creating agent...", flush=True)
    agent = Agent(model=model)
    
    print("Running query...", flush=True)
    try:
        result = await agent.run('Say hello')
        print(f"Result: {result.output}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
    finally:
        await http_client.aclose()

if __name__ == '__main__':
    asyncio.run(main())