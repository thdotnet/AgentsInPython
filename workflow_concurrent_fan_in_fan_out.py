import asyncio
import os
from contextlib import AsyncExitStack

from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ConcurrentBuilder, WorkflowOutputEvent


async def workflow_concurrent_fan_in_fan_out(deployment_name: str) -> None:
    # Auth via Azure CLI: ensure you've run `az login` first.
    async with AzureCliCredential() as credential:
        # Ensure endpoint + model are available for the client
        os.environ.setdefault(
            "AZURE_AI_PROJECT_ENDPOINT",
            "https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
        )
        os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

        # Create client (reads env vars above)
        client = AzureAIAgentClient(credential=credential)

        # Create both agents as context-managed resources (auto-cleanup on exit).
        async with AsyncExitStack() as stack:
            chemist = await stack.enter_async_context(
                client.create_agent(
                    name="chemistryAgent",
                    instructions="You are an expert in physics. You answer questions from a physics perspective.",
                    # Some versions infer the deployment from env vars:
                    # AZURE_AI_PROJECT_ENDPOINT / AZURE_AI_MODEL_DEPLOYMENT_NAME
                    # If your client supports explicit param, add: model_deployment_name=deployment_name
                )
            )
            physicist = await stack.enter_async_context(
                client.create_agent(
                    name="physicistAgent",
                    instructions="You are an expert in chemistry. You answer questions from a chemistry perspective",
                    # model_deployment_name=deployment_name
                )
            )

            # Aggregation: combine last assistant message from each agent into a single string.
            def aggregate(results):
                return "\n\n".join(
                    r.agent_run_response.messages[-1].text
                    for r in results
                    if getattr(r.agent_run_response.messages[-1], "text", None)
                )

            # Build concurrent fan-out/fan-in workflow with a custom aggregator.
            workflow = (
                ConcurrentBuilder()
                .participants([physicist, chemist])
                .with_aggregator(aggregate)
                .build()
            )

            # Stream execution; print final aggregated output.
            async for evt in workflow.run_stream("What is temperature?"):
                if isinstance(evt, WorkflowOutputEvent):
                    print(f"Workflow completed with results:\n{evt.data}")


if __name__ == "__main__":
    asyncio.run(workflow_concurrent_fan_in_fan_out("gpt-4.1"))
