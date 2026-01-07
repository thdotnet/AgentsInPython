import asyncio
import os
from typing import List

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ChatMessage, Role


async def multi_turn_async(deployment_name: str, joker_instructions: str, joker_name: str) -> None:
	"""
	Python equivalent of the C# MultiTurn behavior using Agent Framework.

	- Creates an agent with given instructions and name
	- Maintains a conversation "thread" by appending messages
	- Runs twice on the same logical thread and prints both responses
	- Agent is cleaned up automatically
	"""
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		async with client.create_agent(
			name=joker_name,
			instructions=joker_instructions,
		) as agent:
			messages: List[ChatMessage] = []

			# Turn 1
			messages.append(ChatMessage(role=Role.USER, text="Tell me a joke about a pirate."))
			res1 = await agent.run(messages)
			print(res1.text or "<no assistant reply>")

			# Maintain thread by appending assistant reply
			if res1.text:
				messages.append(ChatMessage(role=Role.ASSISTANT, text=res1.text))

			# Turn 2
			messages.append(
				ChatMessage(
					role=Role.USER,
					text=(
						"Now add some emojis to the joke and tell it in the voice of a pirate's parrot."
					),
				)
			)
			res2 = await agent.run(messages)
			print(res2.text or "<no assistant reply>")


def main() -> None:
	asyncio.run(
		multi_turn_async(
			deployment_name="gpt-4.1",
			joker_instructions="You are good at telling jokes.",
			joker_name="JokerAgent",
		)
	)


if __name__ == "__main__":
	main()

