from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
import traceback


def simple_agent(
	endpoint: str,
	deployment_name: str,
	agent_name: str,
	instructions: str,
	user_message: str = "Tell me a joke about a pirate.",
):
	print("Initializing AIProjectClient...")
	client = AIProjectClient(endpoint=endpoint, credential=AzureCliCredential())

	agent = None
	try:
		print("Creating agent...")
		agent = client.agents.create_agent(
			model=deployment_name,
			name=agent_name,
			instructions=instructions,
		)

		print("Creating thread...")
		thread = client.agents.threads.create()

		print("Posting user message...")
		client.agents.messages.create(
			thread_id=thread.id,
			role="user",
			content=user_message,
		)

		print("Running agent and waiting for completion...")
		client.agents.runs.create_and_process(
			thread_id=thread.id,
			agent_id=agent.id,
		)

		print("Fetching messages...")
		messages = client.agents.messages.list(thread_id=thread.id)

		last_text = None
		for m in messages:
			if getattr(m, "role", None) in ("assistant", "agent", "MessageRole.AGENT"):
				for block in getattr(m, "content", []):
					if getattr(block, "type", None) == "text":
						text_obj = getattr(block, "text", None)
						value = getattr(text_obj, "value", None) if text_obj else None
						if value:
							last_text = value

		if last_text:
			print(last_text)
			return last_text
		else:
			# Fallback: return the first assistant message as string
			for m in messages:
				if getattr(m, "role", None) in ("assistant", "agent", "MessageRole.AGENT"):
					print(m)
					return str(m)
			return None

	finally:
		if agent is not None:
			print("Deleting agent...")
			client.agents.delete_agent(agent.id)


def main() -> None:
	JokerInstructions = "You are good at telling jokes."
	JokerName = "JokerAgent"

	endpoint = "https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default"
	deployment_name = "gpt-4.1"

	simple_agent(
		endpoint=endpoint,
		deployment_name=deployment_name,
		agent_name=JokerName,
		instructions=JokerInstructions,
		user_message="Tell me a joke about a pirate.",
	)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print("ERROR:", e)
		traceback.print_exc()
		raise

