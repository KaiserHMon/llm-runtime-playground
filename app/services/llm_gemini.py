import inspect
from typing import AsyncGenerator, List, Dict, Any
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.chat import Message as DBMessage, MessageRole
from app.services.tools import registry
from app.services.llm_base import LLMProvider, LLMResponse, ToolCall

class GeminiProvider(LLMProvider):
    def __init__(self, model_id: str = "gemini-flash-lite-latest"):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = model_id
        self.system_prompt = """You are a helpful AI assistant.
Be concise and direct in your answers."""

    def _get_tools(self) -> list[types.Tool]:
        """
        Transforms registered Python tools into Gemini SDK types.Tool objects.
        """
        declarations = []
        for name, func in registry._tools.items():
            sig = inspect.signature(func)
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name == "db":
                    continue
                    
                param_type = types.Type.STRING
                if param.annotation is int:
                    param_type = types.Type.INTEGER
                elif param.annotation is float:
                    param_type = types.Type.NUMBER
                elif param.annotation is bool:
                    param_type = types.Type.BOOLEAN
                    
                properties[param_name] = types.Schema(
                    type=param_type,
                    description=f"The {param_name} parameter."
                )
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                    
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=func.__doc__.strip().split("\n")[0] if func.__doc__ else "",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=properties,
                        required=required if required else None
                    )
                )
            )
        return [types.Tool(function_declarations=declarations)] if declarations else []

    def _build_context(self, db_messages: list[DBMessage]) -> list[types.Content]:
        """
        Builds the conversational context for the Gemini SDK.
        Formats database messages as google.genai.types.Content.
        """
        contents = []
        for msg in db_messages:
            if msg.role not in (MessageRole.USER, MessageRole.MODEL, MessageRole.TOOL):
                continue
                
            role = "user" if msg.role in (MessageRole.USER, MessageRole.TOOL) else "model"
            
            if msg.parts:
                parts = [types.Part.model_validate(p) for p in msg.parts]
            else:
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                    
                if msg.role == MessageRole.MODEL and msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(types.Part.from_function_call(
                            name=tc.get("name", ""),
                            args=tc.get("args", {})
                        ))
                        
                if msg.role == MessageRole.TOOL and msg.tool_name:
                    parts.append(types.Part.from_function_response(
                        name=msg.tool_name,
                        response={"result": msg.content}
                    ))
                
            if parts:
                contents.append(
                    types.Content(
                        role=role,
                        parts=parts
                    )
                )
                
        return contents

    async def generate_response(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> LLMResponse:
        contents = self._build_context(history)
        
        system_instruction = self.system_prompt
        if summary:
            system_instruction = f"{system_instruction}\n\n[Summary of the conversation so far:\n{summary}]"
        if rag_context:
            system_instruction = (
                f"{system_instruction}\n\n"
                f"You have access to the following retrieved document context to help answer the user's question. "
                f"Cite the sources you use using numeric footnotes (e.g., [1], [2]) corresponding to the numbers in the context below.\n\n"
                f"Context:\n{rag_context}"
            )

        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                tools=self._get_tools()
            )
        )
        
        tool_calls = []
        if response.function_calls:
            tool_calls = [
                ToolCall(name=fc.name, args=fc.args or {})
                for fc in response.function_calls
            ]
            
        parts = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            parts = [p.model_dump(mode="json") for p in response.candidates[0].content.parts]
            
        input_tokens = None
        output_tokens = None
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        return LLMResponse(
            content=response.text,
            tool_calls=tool_calls,
            parts=parts,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    async def generate_response_stream(
        self,
        history: List[DBMessage],
        summary: str | None = None,
        rag_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        contents = self._build_context(history)
        
        system_instruction = self.system_prompt
        if summary:
            system_instruction = f"{system_instruction}\n\n[Summary of the conversation so far:\n{summary}]"
        if rag_context:
            system_instruction = (
                f"{system_instruction}\n\n"
                f"You have access to the following retrieved document context to help answer the user's question. "
                f"Cite the sources you use using numeric footnotes (e.g., [1], [2]) corresponding to the numbers in the context below.\n\n"
                f"Context:\n{rag_context}"
            )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                tools=self._get_tools()
            )
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        response = await self.client.aio.models.count_tokens(
            model=self.model_id,
            contents=text,
        )
        return response.total_tokens if response.total_tokens is not None else 0

    async def summarize_messages(self, previous_summary: str | None, messages: List[DBMessage]) -> str:
        if not messages:
            return previous_summary or ""

        formatted_messages = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = msg.content or ""
            if msg.tool_calls:
                content += f"\n[Requested Tool Calls: {msg.tool_calls}]"
            if msg.tool_name:
                content += f"\n[Executed Tool: {msg.tool_name}]"
            formatted_messages.append(f"{role.upper()}: {content}")
        
        new_history_text = "\n".join(formatted_messages)

        if previous_summary:
            prompt = (
                f"You are an assistant responsible for maintaining an ongoing summary of a conversation.\n\n"
                f"Existing Summary:\n{previous_summary}\n\n"
                f"New messages to incorporate:\n{new_history_text}\n\n"
                f"Provide a single, updated, and consolidated summary of the conversation so far, integrating the new messages into the existing summary. Keep it concise and direct."
            )
        else:
            prompt = (
                f"You are an assistant responsible for summarizing the following conversation history.\n\n"
                f"Messages to summarize:\n{new_history_text}\n\n"
                f"Provide a concise and direct summary of the conversation so far."
            )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        
        return response.text.strip() if response.text else ""

    def format_tool_response(self, tool_name: str, result: str) -> List[Dict[str, Any]]:
        part_dict = types.Part.from_function_response(
            name=tool_name,
            response={"result": result}
        ).model_dump(mode="json")
        return [part_dict]
