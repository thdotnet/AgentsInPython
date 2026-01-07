import asyncio
import os
from typing import Annotated

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ai_function


@ai_function
def GetWeather(city: Annotated[str, "City name to check weather for"]) -> Annotated[str, "Returns the current weather summary for the given city."]:
	"""Simple tool function used by the agent.

	In a real scenario, you would call a weather API (e.g., Open-Meteo)
	and format the response. Here we return a friendly stub for demo purposes.
	"""
	city = (city or "").strip() or "your location"
	return f"It's sunny and around 20°C today in {city}."


async def simple_agent_with_tools(
	deployment_name: str,
	joker_instructions: str,
	joker_name: str,
) -> None:
	"""Python equivalent of the C# SimpleAgentWitTools sample.

	- Creates an agent with a Python function tool (`GetWeather`)
	- Runs the agent once and prints the response
	- Cleans up the agent automatically via async context manager
	"""

	# Ensure Azure AI Projects env vars for the agent client
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		# Create the agent with the tool; use provided name/instructions
		async with client.create_agent(
			name=joker_name or "WeatherAgent",
			instructions=joker_instructions or "You are a helpful assistant",
			tools=[GetWeather],
		) as agent:
			response = await agent.run("What is the weather like in Amsterdam?")
			print(response.text or "<no assistant reply>")


def main() -> None:
	asyncio.run(
		simple_agent_with_tools(
			deployment_name="gpt-4.1",
			joker_instructions="You are a helpful assistant",
			joker_name="WeatherAgent",
		)
	)


if __name__ == "__main__":
	main()

