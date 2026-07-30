from enum import Enum
from pydantic import BaseModel, Field
from app.models.chat import Message
from app.services.llm.factory import factory

class RouteType(str, Enum):
    RAG = "RAG"
    CHAT = "CHAT"

class RoutingDecision(BaseModel):
    route: RouteType = Field(
        description="Select RAG if the query asks for specific knowledge, project info, files, passphrases, or codes. Select CHAT for general conversation, greetings, generic questions, or explaining general concepts."
    )
    justification: str = Field(
        description="A brief justification of why this route was selected."
    )

async def route_message(
    content: str,
    history: list[Message] | None = None,
    provider_name: str | None = None
) -> str:
    """
    Routes a message to either 'RAG' or 'CHAT' by querying the active provider.
    """
    provider = factory.get_provider(provider_name)
    return await provider.route_message(content, history)
