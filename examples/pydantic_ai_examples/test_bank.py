"""Debug version of bank_support.py to identify the hanging issue."""

import asyncio
import sys
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider


class DatabaseConn:
    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            if include_pending:
                return 123.45
            else:
                return 100.00
        else:
            raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their card or not')
    risk: int = Field(description='Risk level of query', ge=0, le=10)


print("Creating Ollama model...", flush=True)
ollama_model = OpenAIModel(
    model_name='qwen3:8b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
print("Model created successfully", flush=True)

print("Creating support agent...", flush=True)
support_agent = Agent(
    model=ollama_model,
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)
print("Agent created successfully", flush=True)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    print("Getting customer name...", flush=True)
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    print(f"Customer name: {customer_name}", flush=True)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    print(f"Getting balance (include_pending={include_pending})...", flush=True)
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    print(f"Balance: ${balance:.2f}", flush=True)
    return f'${balance:.2f}'


async def main():
    print("Starting main function...", flush=True)
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    print("Dependencies created", flush=True)
    
    print("\nRunning query: 'What is my balance?'", flush=True)
    try:
        result = await support_agent.run('What is my balance?', deps=deps)
        print(f"Result: {result.output}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Starting script...", flush=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
    except Exception as e:
        print(f"Unhandled error: {e}", flush=True)
        import traceback
        traceback.print_exc()