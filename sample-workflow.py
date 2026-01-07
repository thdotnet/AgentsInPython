import asyncio
import os
from contextlib import AsyncExitStack

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import (
	WorkflowBuilder,
	WorkflowOutputEvent,
	AgentRunUpdateEvent,
)


async def sample_workflow(deployment_name: str) -> None:
	# Ensure endpoint + model are available via env for the Azure client
	os.environ.setdefault(
		"AZURE_AI_PROJECT_ENDPOINT",
		"https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
	)
	os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

	async with AzureCliCredential() as credential:
		client = AzureAIAgentClient(credential=credential)

		# Use AsyncExitStack so agents are cleaned up automatically
		async with AsyncExitStack() as stack:
			french_agent = await stack.enter_async_context(
				client.create_agent(
					name="FrenchAgent",
					instructions=(
						"You are a translation assistant that translates the provided text to French."
					),
				)
			)

			spanish_agent = await stack.enter_async_context(
				client.create_agent(
					name="SpanishAgent",
					instructions=(
						"You are a translation assistant that translates the provided text to Spanish."
					),
				)
			)

			quality_agent = await stack.enter_async_context(
				client.create_agent(
					name="QualityAgent",
					instructions=(
						"You are a multilingual translation quality reviewer. Check the translations for grammar accuracy, "
						"tone consistency, and cultural fit compared to the original English text. "
						"Give a brief summary with a quality rating (Excellent / Good / Needs Review). "
						"Example output: Quality Excellent Feedback: Accurate translation, friendly tone preserved, minor punctuation tweaks only."
					),
				)
			)

			summary_agent = await stack.enter_async_context(
				client.create_agent(
					name="SummaryAgent",
					instructions=(
						"You are a localization summary assistant. Summarize the localization results below. "
						"For each language, list: - Translation quality - Tone feedback - Any corrections made. "
						"Then, give an overall summary in 3-5 lines."
					),
				)
			)

			# Build the workflow with sequential edges
			workflow = (
				WorkflowBuilder()
				.set_start_executor(french_agent)
				.add_edge(french_agent, spanish_agent)
				.add_edge(spanish_agent, quality_agent)
				.add_edge(quality_agent, summary_agent)
				.build()
			)

			user_text = (
				"English texts for beginners to practice reading and comprehension online and for free. "
				"Practicing your comprehension of written English will both improve your vocabulary and understanding "
				"of grammar and word order. The texts below are designed to help you develop while giving you an instant "
				"evaluation of your progress."
			)

			# Stream execution; print agent updates and final output
			last_executor = None
			async for evt in workflow.run_stream(user_text):
				if isinstance(evt, AgentRunUpdateEvent):
					if evt.executor_id != last_executor:
						if last_executor is not None:
							print()
						print(f"{evt.executor_id}:", end=" ", flush=True)
						last_executor = evt.executor_id
					print(evt.data, end="", flush=True)
				elif isinstance(evt, WorkflowOutputEvent):
					print()
					print("\nWorkflow completed with summary:\n")
					print(evt.data)


def main() -> None:
	asyncio.run(sample_workflow("gpt-4.1"))


if __name__ == "__main__":
	main()

