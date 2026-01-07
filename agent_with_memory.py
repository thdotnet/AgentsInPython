import time
import traceback

from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, AgentKind


def _run_and_print_response(openai_client, model: str, instructions: str, conversation_id: str, user_text: str) -> None:
	"""
	Helper: create a response tied to a given conversation and print the output text.
	Uses the OpenAI Responses API which is supported in the Azure AI Projects client.
	"""
	resp = openai_client.responses.create(
		model=model,
		instructions=instructions,
		conversation={"id": conversation_id},
		input=user_text,
	)
	if getattr(resp, "output_text", None):
		print(resp.output_text)
	else:
		print(resp)


def agent_with_memory(endpoint: str, deployment_name: str) -> None:
	"""
	Python version of the C# AgentWithMemory sample using azure-ai-projects.

	- Creates an agent with instructions and name
	- Starts a thread, sends two messages introducing personal trip details
	- Waits briefly to allow memory indexing
	- Asks what the agent already knows about the upcoming trip
	- Demonstrates persisted state by "serializing" (saving thread id) and "deserializing" (reusing it)
	- Starts a new thread in the same agent/memory scope and asks for a summary
	- Deletes the agent at the end
	"""
	print("Initializing AIProjectClient...")
	client = AIProjectClient(endpoint=endpoint, credential=AzureCliCredential())

	options_instructions = (
		"You are a friendly travel assistant. "
		"Use known memories about the user when responding, and do not invent details."
	)
	options_name = "AgentWithMemory"

	agent = None
	try:
		print("Creating agent...")
		# Create the agent (assistant) with the given model
		definition = PromptAgentDefinition()
		definition["kind"] = AgentKind.PROMPT
		definition["model"] = deployment_name
		definition["instructions"] = options_instructions

		agent = client.agents.create(
			name=options_name,
			definition=definition,
		)

		# Use the OpenAI-compatible client for conversations/responses
		oc = client.get_openai_client()

		print("Creating conversation...")
		conv = oc.conversations.create()

		# Initial messages to seed personal details
		_run_and_print_response(
			oc,
			model=deployment_name,
			instructions=options_instructions,
			conversation_id=conv.id,
			user_text=(
				"Hi there! My name is Taylor and I'm planning a hiking trip "
				"to Patagonia in November."
			),
		)
		_run_and_print_response(
			oc,
			model=deployment_name,
			instructions=options_instructions,
			conversation_id=conv.id,
			user_text=(
				"I'm travelling with my sister and we love finding scenic viewpoints."
			),
		)

		print("\nWaiting briefly for Mem0 to index the new memories...\n")
		time.sleep(2)

		_run_and_print_response(
			oc,
			model=deployment_name,
			instructions=options_instructions,
			conversation_id=conv.id,
			user_text=("What do you already know about my upcoming trip?"),
		)

		print("\n>> Serialize and deserialize the thread to demonstrate persisted state\n")
		serialized_conversation_id = conv.id  # Serialize by storing the conversation id
		# Reuse the saved conversation id for persistence demonstration
		restored_conversation_id = serialized_conversation_id

		_run_and_print_response(
			oc,
			model=deployment_name,
			instructions=options_instructions,
			conversation_id=restored_conversation_id,
			user_text=("Can you recap the personal details you remember?"),
		)

		print("\n>> Start a new conversation that shares the same Mem0 scope\n")
		new_conv = oc.conversations.create()
		_run_and_print_response(
			oc,
			model=deployment_name,
			instructions=options_instructions,
			conversation_id=new_conv.id,
			user_text=("Summarize what you already know about me."),
		)
	finally:
		if agent is not None:
			print("Deleting agent...")
			client.agents.delete(agent.id)


def main() -> None:
	endpoint = "https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default"
	deployment_name = "gpt-4.1"
	agent_with_memory(endpoint=endpoint, deployment_name=deployment_name)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print("ERROR:", e)
		traceback.print_exc()
		raise

