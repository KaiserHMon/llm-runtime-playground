from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any
from pydantic import BaseModel
from app.models.chat import Message as DBMessage

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]

class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: List[ToolCall] = []
    parts: List[Dict[str, Any]] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> LLMResponse:
        """
        Generates a response from the LLM given a conversation history.
        """
        pass

    @abstractmethod
    def generate_response_stream(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams the response content from the LLM.
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Counts the tokens in the given text.
        """
        pass

    @abstractmethod
    async def summarize_messages(self, previous_summary: str | None, messages: List[DBMessage]) -> str:
        """
        Summarizes the messages to update the conversation summary.
        """
        pass

    @abstractmethod
    def format_tool_response(self, tool_name: str, result: str) -> List[Dict[str, Any]]:
        """
        Formats a tool execution result into the provider's specific message parts representation.
        """
        pass

    @abstractmethod
    async def route_message(self, content: str, history: List[DBMessage] | None = None) -> str:
        """
        Decides whether the message should be routed to 'RAG' or 'CHAT'.
        """
        pass

