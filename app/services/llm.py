from google import genai
from google.genai import types
from app.core.config import settings
from app.models.chat import Message as DBMessage, MessageRole

# Initialize the Gemini client using the API key from our settings
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Using flash by default as it is the fastest and most cost-effective model for an MVP
MODEL_ID = "gemini-flash-lite-latest"

# Hardcoding the system prompt for this initial phase.
# CONCEPTS > CODE: The system prompt defines the "personality" and behavioral boundaries 
# of the model. Never leave it empty in production.
SYSTEM_PROMPT = """You are a helpful AI assistant.
Be concise and direct in your answers."""

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
        # Ignore system messages in the history because Gemini handles them 
        # more efficiently through the configuration (system_instruction).
        if msg.role not in (MessageRole.USER, MessageRole.MODEL):
            continue
            
        role = "user" if msg.role == MessageRole.USER else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg.content)]
            )
        )
            
    return contents

async def generate_response(user_message_content: str, history: list[DBMessage]) -> str:
    """
    Builds the historical context, appends the current message, and calls Gemini asynchronously.
    """
    # Build history
    contents = build_context(history)
    
    # Append the new user message at the end
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message_content)]
        )
    )
    
    # Call the model using the async SDK client
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )
    )
    
    # Always validate that it returns text; in an MVP we fail fast
    if not response.text:
        raise ValueError("The model did not return text. Check console/logs.")
        
    return response.text

async def generate_response_stream(user_message_content: str, history: list[DBMessage]):
    """
    Builds context and yields chunks of the response from Gemini asynchronously.
    """
    contents = build_context(history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message_content)]
        )
    )
    
    response_stream = await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )
    )
    
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text
