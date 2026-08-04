from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.schemas.chat import (
    ConversationCreate, 
    ConversationResponse, 
    MessageCreate, 
    MessageResponse, 
    ConversationWithMessages
)
from app.models.chat import Conversation
from app.services.chat_service import process_chat_message, stream_chat_message

router = APIRouter(prefix="/conversations", tags=["Chat"])

@router.post("", response_model=ConversationResponse)
async def create_conversation(payload: ConversationCreate, db: AsyncSession = Depends(get_db)):
    """Creates a new empty conversation."""
    db_conv = Conversation(title=payload.title)
    db.add(db_conv)
    await db.commit()
    await db.refresh(db_conv)
    return db_conv

@router.get("", response_model=list[ConversationResponse])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """Lists all conversations ordered by updated_at descending."""
    stmt = select(Conversation).order_by(Conversation.updated_at.desc())
    result = await db.scalars(stmt)
    return result.all()

@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(conversation_id: str, payload: MessageCreate, db: AsyncSession = Depends(get_db)):
    """Sends a user message, calls LLM, and returns the model's response."""
    try:
        model_message = await process_chat_message(
            db=db, 
            conversation_id=conversation_id, 
            content=payload.content,
            provider_name=payload.provider,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            enabled_tools=payload.enabled_tools
        )
        return model_message
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Avoid leaking raw exceptions in production, but good for local debugging MVP
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@router.post("/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    payload: MessageCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Sends a user message and streams the model's response back via SSE."""
    try:
        return StreamingResponse(
            stream_chat_message(
                db=db, 
                conversation_id=conversation_id, 
                content=payload.content,
                provider_name=payload.provider,
                system_prompt=payload.system_prompt,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                enabled_tools=payload.enabled_tools,
                background_tasks=background_tasks
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Fetches a conversation and its entire message history."""
    stmt = select(Conversation).options(selectinload(Conversation.messages)).where(Conversation.id == conversation_id)
    db_conv = await db.scalar(stmt)
    if not db_conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db_conv
