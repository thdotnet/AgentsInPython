import asyncio
import os
from typing import Annotated

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ai_function


@ai_function
def get_weather(
	location: Annotated[str, "City name to get the forecast for"],
	unit: Annotated[str, "Temperature unit: celsius or fahrenheit"] = "celsius",
) -> str:
	"""Return a simple mock weather report for the given location."""
	unit_symbol = "C" if unit.lower().startswith("c") else "F"
	return f"It is 22°{unit_symbol} and sunny today in {location}."


async def using_agent_as_a_function(deployment_name: str) -> None:
	# Ensure Azure AI Projects endpoint and model deployment are set
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		# Create the weather agent with the python function tool
		async with client.create_agent(
			name="WeatherAgent",
			instructions="You answer questions about the weather.",
			tools=[get_weather],
		) as weather_agent:

			# Wrap the weather agent itself as a callable tool
			weather_agent_tool = weather_agent.as_tool(name="weather_agent")

			# Create a second agent that can call the first agent as a tool
			async with client.create_agent(
				name="FrenchAgent",
				instructions="Vous êtes un assistant serviable qui répond en français.",
				tools=[weather_agent_tool],
			) as french_agent:
				response = await french_agent.run("What is the weather like in Amsterdam?")
				print(response.text or "<no assistant reply>")


def main() -> None:
	asyncio.run(using_agent_as_a_function("gpt-4.1"))


if __name__ == "__main__":
	main()

