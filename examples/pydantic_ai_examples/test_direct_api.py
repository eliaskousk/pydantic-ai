"""Test direct OpenAI client with Ollama."""

import asyncio
from openai import AsyncOpenAI

async def main():
    print("Creating AsyncOpenAI client...", flush=True)
    client = AsyncOpenAI(
        base_url='http://localhost:11434/v1',
        api_key='not-needed'
    )
    print("Client created", flush=True)
    
    print("Making API call...", flush=True)
    response = await client.chat.completions.create(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content': 'Say hello'}],
        max_tokens=10
    )
    print(f"Response: {response.choices[0].message.content}", flush=True)

if __name__ == '__main__':
    asyncio.run(main())