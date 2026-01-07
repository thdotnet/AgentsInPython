import asyncio
import os
from typing import Optional

from pydantic import BaseModel
from azure.identity.aio import AzureCliCredential
from agent_framework.azure import AzureAIAgentClient
from agent_framework import ChatMessage, Role


class PersonInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None


async def agent_with_structured_output(deployment_name: str) -> None:
    # Ensure Azure AI Projects env vars for the chat client
    os.environ.setdefault(
        "AZURE_AI_PROJECT_ENDPOINT",
        "https://<your-microsoft-foundry>.services.ai.azure.com/api/projects/proj-default",
    )
    os.environ.setdefault("AZURE_AI_MODEL_DEPLOYMENT_NAME", deployment_name)

    async with AzureCliCredential() as credential:
        # AzureAIAgentClient integrates with Azure AI Projects Agents
        client = AzureAIAgentClient(credential=credential)

        # Create agent as an async context manager to auto-cleanup
        async with client.create_agent(
            name="StructuredOutput",
            instructions=(
                "You are a friendly travel assistant. Use known memories about the user when responding, "
                "and do not invent details."
            ),
        ) as agent:

            # Run with unstructured input; request structured output as PersonInfo
            user_prompt = (
                "Please provide information about John Smith, who is a 35-year-old software engineer."
            )

            # Prefer explicit ChatMessage for compatibility with agent runs
            messages = [ChatMessage(role=Role.USER, text=user_prompt)]

            response = await agent.run(messages, response_format=PersonInfo)

            print("Assistant Output:")
            if response.value is not None:
                person: PersonInfo = response.value
                print(f"Name: {person.name}")
                print(f"Age: {person.age}")
                print(f"Occupation: {person.occupation}")
            else:
                # Fallback: try to print raw text if structured value missing
                print(response.text or "<no structured output>")


def main() -> None:
    asyncio.run(agent_with_structured_output("gpt-4.1"))


if __name__ == "__main__":
    main()
