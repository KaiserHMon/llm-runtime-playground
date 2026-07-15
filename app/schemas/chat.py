from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from app.models.chat import MessageRole

# -----------------
# Message Schemas
# -----------------

class MessageCreate(BaseModel):
    """Payload sent by the user to the chat endpoint."""
    content: str = Field(..., description="The text content of the message")

class MessageResponse(BaseModel):
    """Response model returned by the API for a message."""
    id: str
    role: MessageRole
    content: str
    tokens: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# -----------------
# Conversation Schemas
# -----------------

class ConversationCreate(BaseModel):
    """Payload to start a new conversation."""
    title: str | None = Field(default=None, description="Optional title for the conversation")

class ConversationResponse(BaseModel):
    """Response model returned by the API for a conversation summary."""
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ConversationWithMessages(ConversationResponse):
    """Conversation details including the full message history."""
    messages: list[MessageResponse] = []
