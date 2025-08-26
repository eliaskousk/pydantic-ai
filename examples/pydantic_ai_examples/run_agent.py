import logfire
import nest_asyncio

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

nest_asyncio.apply()

agent = Agent('google-vertex:gemini-2.5-flash')


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
            #> The capital of
            #> The capital of the UK is
            #> The capital of the UK is London.

if __name__ == '__main__':
    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.output)
    # > The capital of Italy is Rome.

    import asyncio
    asyncio.run(main())
