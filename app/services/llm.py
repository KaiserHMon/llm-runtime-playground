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

def build_context(db_messages: list[DBMessage], limit: int = 10) -> list[types.Content]:
    """
    Builds the conversational context for the Gemini SDK.
    Takes the last N messages and formats them as google.genai.types.Content.
    
    Why a limit? Because the context window is not infinite and tokens cost money.
    """
    # Take the last N messages to avoid overflowing the context
    recent_messages = db_messages[-limit:] if limit > 0 else db_messages
    
    contents = []
    for msg in recent_messages:
        # Map our DB roles to the roles understood by Gemini ("user" or "model")
        role = "user" if msg.role == MessageRole.USER else "model"
        
        # Ignore system messages in the history because Gemini handles them 
        # more efficiently through the configuration (system_instruction).
        if msg.role in (MessageRole.USER, MessageRole.MODEL):
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
