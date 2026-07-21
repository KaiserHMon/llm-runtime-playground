import inspect
from typing import Any
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.chat import Message as DBMessage, MessageRole
from app.services.tools import registry

# Initialize the Gemini client using the API key from our settings
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Using flash by default as it is the fastest and most cost-effective model for an MVP
MODEL_ID = "gemini-flash-lite-latest"

# Hardcoding the system prompt for this initial phase.
# CONCEPTS > CODE: The system prompt defines the "personality" and behavioral boundaries 
# of the model. Never leave it empty in production.
SYSTEM_PROMPT = """You are a helpful AI assistant.
Be concise and direct in your answers."""

def get_tools_for_gemini() -> list[types.Tool]:
    """
    Transforms registered Python tools into Gemini SDK types.Tool objects.
    
    Why this helper is needed (CONCEPTS > CODE):
    When passing Python functions directly to the SDK, the SDK generates the schema 
    via Pydantic. However, some tools require a database session parameter `db: AsyncSession`.
    Since AsyncSession is a backend-only object that the LLM cannot supply, Pydantic fails 
    to generate a valid JSON schema for it.
    
    This function manually constructs the types.FunctionDeclaration objects and filters 
    out the `db` parameter, while mapping basic Python type annotations (int, float, bool) 
    to their respective Gemini API schema types (INTEGER, NUMBER, BOOLEAN).
    """
    declarations = []
    for name, func in registry._tools.items():
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Filter out the internal DB connection parameter to keep it hidden from the LLM schema
            if param_name == "db":
                continue
                
            # Default to string schema type
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
            # If the parameter does not have a default value, it is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                # Extract only the first line of docstring as description
                description=func.__doc__.strip().split("\n")[0] if func.__doc__ else "",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=required if required else None
                )
            )
        )
    # Return a single Tool container with all function declarations
    return [types.Tool(function_declarations=declarations)] if declarations else []

async def count_tokens(text: str) -> int:
    """
    Asynchronously count tokens in a given text using Gemini SDK.
    """
    if not text:
        return 0
    response = await client.aio.models.count_tokens(
        model=MODEL_ID,
        contents=text,
    )
    return response.total_tokens if response.total_tokens is not None else 0

def build_context(db_messages: list[DBMessage]) -> list[types.Content]:
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

async def generate_response(history: list[DBMessage]) -> Any:
    """
    Builds the historical context and calls Gemini asynchronously.
    """
    # Build history
    contents = build_context(history)
    
    # Call the model using the async SDK client
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            tools=get_tools_for_gemini()
        )
    )
    
    return response

async def generate_response_stream(history: list[DBMessage]):
    """
    Builds context and yields chunks of the response from Gemini asynchronously.
    """
    contents = build_context(history)
    
    response_stream = await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            tools=get_tools_for_gemini()
        )
    )
    
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text
