import logfire

from pydantic_ai import Agent

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

provider = GoogleProvider(vertexai=True,
                          project='stromasys-projects',
                          location='us-east5')
model = GoogleModel('gemini-2.5-flash', provider=provider)


agent = Agent(model)

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),
)
print(result2.output)
#> Albert Einstein's most famous equation is (E = mc^2).
