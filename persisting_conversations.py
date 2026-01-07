import asyncio
import json
import os
import tempfile
from typing import List

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ChatMessage, Role


async def persisting_conversations(deployment_name: str, joker_instructions: str, joker_name: str) -> None:
	"""
	Python equivalent of the C# PersistingConversations sample.

	- Creates an agent with given instructions and name
	- Starts a conversation and gets an assistant reply
	- Serializes the in-memory conversation (messages) to JSON and saves it to a temp file
	- Reloads from JSON to resume the conversation
	- Runs again on the resumed conversation and prints the response
	- Agent is cleaned up automatically via async context manager
	"""

	# Ensure endpoint + model are available via env for the Azure client
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		async with client.create_agent(name=joker_name, instructions=joker_instructions) as agent:
			# Start a new logical thread as an in-memory list of messages
			messages: List[ChatMessage] = []

			# Turn 1: user asks for a short pirate joke
			messages.append(ChatMessage(role=Role.USER, text="Tell me a short pirate joke."))
			res1 = await agent.run(messages)
			print(res1.text or "<no assistant reply>")

			# Append assistant reply to maintain the logical thread
			if res1.text:
				messages.append(ChatMessage(role=Role.ASSISTANT, text=res1.text))

			# Serialize the conversation to JSON (role/text only)
			serialized = [
				{
					"role": (m.role.value if hasattr(m.role, "value") else str(m.role)).lower(),
					"text": m.text,
				}
				for m in messages
			]

			# Save to a local temp file (replace with DB or blob storage in production)
			file_path = os.path.join(tempfile.gettempdir(), "agent_thread.json")
			with open(file_path, "w", encoding="utf-8") as f:
				json.dump(serialized, f, ensure_ascii=False)

			# Load back from disk
			with open(file_path, "r", encoding="utf-8") as f:
				loaded = json.load(f)

			# Deserialize into ChatMessage list tied to the same agent type
			resumed_messages: List[ChatMessage] = []
			for item in loaded:
				role_str = str(item.get("role", "user")).lower()
				text = item.get("text", "")
				if role_str in ("assistant", "agent"):
					role_enum = Role.ASSISTANT
				else:
					role_enum = Role.USER
				resumed_messages.append(ChatMessage(role=role_enum, text=text))

			# Continue the conversation with resumed thread
			resumed_messages.append(
				ChatMessage(role=Role.USER, text="Now tell that joke in the voice of a pirate.")
			)
			res2 = await agent.run(resumed_messages)
			print(res2.text or "<no assistant reply>")


def main() -> None:
	asyncio.run(
		persisting_conversations(
			deployment_name="gpt-4.1",
			joker_instructions="You are good at telling jokes.",
			joker_name="JokerAgent",
		)
	)


if __name__ == "__main__":
	main()

