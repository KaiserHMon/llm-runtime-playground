import asyncio
from typing import AsyncGenerator, List, Dict, Any
from app.models.chat import Message as DBMessage, MessageRole
from app.services.llm.base import LLMProvider, LLMResponse, ToolCall

class MockProvider(LLMProvider):
    async def generate_response(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> LLMResponse:
        # Find the last user message
        last_user_msg = ""
        for msg in reversed(history):
            if msg.role == MessageRole.USER:
                last_user_msg = msg.content or ""
                break

        # Check if the last message in history is a tool response
        # If it is, we don't trigger the tool call again
        last_msg_is_tool = False
        if history:
            last_msg_is_tool = history[-1].role == MessageRole.TOOL

        # Trigger tool calls mock
        tool_calls = []
        if not last_msg_is_tool:
            if "weather" in last_user_msg.lower():
                tool_calls.append(ToolCall(name="get_weather", args={"location": "Buenos Aires"}))
            elif "time" in last_user_msg.lower() or "hora" in last_user_msg.lower():
                tool_calls.append(ToolCall(name="get_current_datetime", args={}))

        if tool_calls:
            return LLMResponse(
                content=None,
                tool_calls=tool_calls,
                parts=[{"text": "I need to call a tool to answer that."}],
                input_tokens=15,
                output_tokens=10
            )

        # Plain text mock response
        response_text = f"[Mock Provider] ¡Hola! Esta es una respuesta simulada en caliente. Me dijiste: '{last_user_msg}'"
        if rag_context:
            response_text += f"\n\nContexto RAG recuperado:\n{rag_context}"
            
        return LLMResponse(
            content=response_text,
            tool_calls=[],
            parts=[{"text": response_text}],
            input_tokens=len(last_user_msg) // 4,
            output_tokens=len(response_text) // 4
        )

    async def generate_response_stream(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        response = await self.generate_response(history, summary, rag_context)
        content = response.content or "[Mock Provider] Tool calls requested."
        # Yield words with a small delay to simulate streaming
        for word in content.split(" "):
            yield word + " "
            await asyncio.sleep(0.05)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4 if text else 0

    async def summarize_messages(self, previous_summary: str | None, messages: List[DBMessage]) -> str:
        new_msgs_content = ", ".join([m.content for m in messages if m.content])
        if previous_summary:
            return f"{previous_summary} Además, se habló de: {new_msgs_content[:50]}..."
        return f"Resumen simulado: Se habló de {new_msgs_content[:50]}..."

    def format_tool_response(self, tool_name: str, result: str) -> List[Dict[str, Any]]:
        # Return a simple mock part representing the tool result
        return [{"text": f"[Mock Tool Response for {tool_name}]: {result}"}]

    async def route_message(self, content: str, history: List[DBMessage] | None = None) -> str:
        # Check current content and history content for RAG-related terms
        keywords = ["code", "passphrase", "antigravity", "secreto", "secret", "ventilación", "ventilation"]
        text_to_check = content.lower()
        if any(kw in text_to_check for kw in keywords):
            return "RAG"
        
        if history:
            for msg in history:
                if msg.content and any(kw in msg.content.lower() for kw in keywords):
                    return "RAG"
                    
        return "CHAT"

