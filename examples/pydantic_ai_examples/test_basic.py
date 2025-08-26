"""
Basic PydanticAI example demonstrating core functionality.

This example shows:
- Creating an agent with a model
- Using tools to extend agent capabilities
- Getting structured output with Pydantic models
- Dependency injection for tools
"""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider


# Define a structured output model
class WeatherReport(BaseModel):
    """Structured weather report output."""
    
    location: str = Field(description="The location for the weather report")
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g., sunny, cloudy, rainy)")
    recommendation: str = Field(description="What to wear or bring based on the weather")


# Define dependencies for the agent
@dataclass
class WeatherDeps:
    """Dependencies that tools can access."""
    
    default_unit: str = "celsius"
    include_forecast: bool = False


# Create the agent with OpenAIModel using OllamaProvider
# Make sure Ollama is running locally: ollama run qwen2.5:3b
provider = OllamaProvider(
    base_url="http://localhost:11434/v1",  # Default Ollama URL
)
model = OpenAIModel(
    model_name="qwen3:0.6b",
    provider=provider,
)

agent = Agent(
    model,
    deps_type=WeatherDeps,
    output_type=WeatherReport,
    system_prompt=(
        "You are a helpful weather assistant. "
        "Provide weather information and recommendations based on the conditions."
    ),
)


# Add a tool to get weather data (mock implementation)
@agent.tool
async def get_weather_data(ctx: RunContext[WeatherDeps], location: str) -> str:
    """
    Get weather data for a location.
    
    Args:
        ctx: The run context with dependencies
        location: The location to get weather for
    
    Returns:
        Weather data as a string
    """
    # In a real app, this would call a weather API
    # For demo purposes, we'll return mock data
    unit = ctx.deps.default_unit
    
    # Mock weather data based on location
    weather_data = {
        "london": {"temp": 15, "conditions": "cloudy with occasional rain"},
        "miami": {"temp": 28, "conditions": "sunny and humid"},
        "tokyo": {"temp": 20, "conditions": "partly cloudy"},
        "new york": {"temp": 18, "conditions": "clear skies"},
    }
    
    # Default weather if location not found
    data = weather_data.get(
        location.lower(),
        {"temp": 22, "conditions": "mild and pleasant"}
    )
    
    temp_str = f"{data['temp']}°C" if unit == "celsius" else f"{data['temp'] * 9/5 + 32:.1f}°F"
    
    result = f"Current weather in {location}: {temp_str}, {data['conditions']}"
    
    if ctx.deps.include_forecast:
        result += " (Forecast: Similar conditions expected tomorrow)"
    
    return result


# Add another tool for recommendations
@agent.tool
def get_clothing_recommendation(ctx: RunContext[WeatherDeps], temperature: float, conditions: str) -> str:
    """
    Get clothing recommendations based on weather.
    
    Args:
        temperature: Temperature in Celsius
        conditions: Weather conditions description
    
    Returns:
        Clothing recommendation
    """
    recommendations = []
    
    # Temperature-based recommendations
    if temperature < 10:
        recommendations.append("warm coat and gloves")
    elif temperature < 20:
        recommendations.append("light jacket or sweater")
    else:
        recommendations.append("light clothing")
    
    # Condition-based recommendations
    if "rain" in conditions.lower():
        recommendations.append("umbrella or raincoat")
    elif "sunny" in conditions.lower():
        recommendations.append("sunglasses and sunscreen")
    elif "snow" in conditions.lower():
        recommendations.append("winter boots and warm layers")
    
    return f"Recommended: {', '.join(recommendations)}"


async def main():
    """Run the example."""
    print("=" * 60)
    print("PydanticAI Basic Example - Weather Assistant")
    print("Using OpenAIModel with OllamaProvider (qwen3:0.6b)")
    print("Make sure Ollama is running: ollama run qwen3:0.6b")
    print("=" * 60)
    
    # Example 1: Basic usage with default dependencies
    print("\n1. Basic weather query:")
    print("-" * 40)
    
    deps = WeatherDeps()
    result = await agent.run(
        "What's the weather like in London and what should I wear?",
        deps=deps
    )
    
    print(f"Location: {result.output.location}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    print(f"Recommendation: {result.output.recommendation}")
    
    # Example 2: Using different dependencies
    print("\n2. Weather with forecast enabled:")
    print("-" * 40)
    
    deps_with_forecast = WeatherDeps(
        default_unit="celsius",
        include_forecast=True
    )
    
    result = await agent.run(
        "Tell me about the weather in Miami",
        deps=deps_with_forecast
    )
    
    print(f"Location: {result.output.location}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    print(f"Recommendation: {result.output.recommendation}")
    
    # Example 3: Multiple locations (agent will handle intelligently)
    print("\n3. Comparing multiple locations:")
    print("-" * 40)
    
    result = await agent.run(
        "Compare the weather in Tokyo - I need to know what to pack",
        deps=deps
    )
    
    print(f"Location: {result.output.location}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    print(f"Recommendation: {result.output.recommendation}")
    
    # Show usage statistics
    print("\n" + "=" * 60)
    print("Usage Statistics:")
    print(f"Total tokens: {result.usage().total_tokens}")
    print(f"Model: {model.model_name}")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())