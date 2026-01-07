import asyncio
import os
from typing import Annotated

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ai_function, ChatMessage, Role
from agent_framework._middleware import chat_middleware, ChatContext


@ai_function
def GetDateTime() -> Annotated[str, "Returns the current date/time in ISO format"]:
	import datetime
	return datetime.datetime.now().isoformat()


async def adding_middleware_to_agents(
	deployment_name: str,
	joker_instructions: str,
	joker_name: str,
) -> None:
	"""
	Python equivalent of the C# AddingMiddlewareToAgents sample.

	- Creates an agent with a Python function tool (`GetDateTime`)
	- Adds a custom run middleware
	- Runs the agent once and prints the response
	- Cleans up the agent
	"""

	# Ensure Azure AI Projects endpoint and model deployment are available via env vars
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		# Define a custom chat middleware using the framework decorator.
		# The middleware receives a ChatContext with messages and result.
		@chat_middleware
		async def CustomAgentChatMiddleware(context: ChatContext, next):
			# Pre-processing: gently nudge the assistant to be concise by
			# appending a small hint to the last user turn if available.
			if hasattr(context, "messages") and isinstance(context.messages, list) and context.messages:
				last = context.messages[-1]
				try:
					role = getattr(last, "role", None)
					text = getattr(last, "text", None)
					if isinstance(text, str) and (role == "user" or role == Role.USER):
						last.text = text + " (kindly be concise)"
				except Exception:
					# Non-fatal: if structure differs, just continue
					pass

			# Continue the pipeline
			await next(context)

			# Post-processing: no-op, but could normalize context.result

		# Create the agent with the tool and middleware
		async with client.create_agent(
			name=joker_name,
			instructions=joker_instructions,
			tools=[GetDateTime],
			middleware=CustomAgentChatMiddleware,
		) as agent:
			response = await agent.run("What's the current time?")
			print(response.text or "<no assistant reply>")


def main() -> None:
	asyncio.run(
		adding_middleware_to_agents(
			deployment_name="gpt-4.1",
			joker_instructions="You are good at telling jokes.",
			joker_name="JokerAgent",
		)
	)


if __name__ == "__main__":
	main()

