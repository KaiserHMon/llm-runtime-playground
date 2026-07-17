from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_db
from app.schemas.chat import (
    ConversationCreate, 
    ConversationResponse, 
    MessageCreate, 
    MessageResponse, 
    ConversationWithMessages
)
from app.models.chat import Conversation
from app.services.chat_service import process_chat_message

router = APIRouter(prefix="/conversations", tags=["Chat"])

@router.post("", response_model=ConversationResponse)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    """Creates a new empty conversation."""
    db_conv = Conversation(title=payload.title)
    db.add(db_conv)
    db.commit()
    db.refresh(db_conv)
    return db_conv

@router.post("/{conversation_id}/messages", response_model=MessageResponse)
def send_message(conversation_id: str, payload: MessageCreate, db: Session = Depends(get_db)):
    """Sends a user message, calls Gemini, and returns the model's response."""
    try:
        model_message = process_chat_message(
            db=db, 
            conversation_id=conversation_id, 
            content=payload.content
        )
        return model_message
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Avoid leaking raw exceptions in production, but good for local debugging MVP
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@router.get("/{conversation_id}", response_model=ConversationWithMessages)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Fetches a conversation and its entire message history."""
    db_conv = db.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if not db_conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db_conv
