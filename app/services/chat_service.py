from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.chat import Conversation, Message, MessageRole
from app.services.llm import generate_response, generate_response_stream


async def process_chat_message(db: AsyncSession, conversation_id: str, content: str) -> Message:
    """
    Orchestrates the chat flow asynchronously:
    1. Validate conversation.
    2. Fetches history.
    3. Calls LLM asynchronously.
    4. Saves user and model messages atomically.
    """
    # 1. Validate that the conversation exists
    conversation = await db.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    # 2. Read-only: Fetch historical messages BEFORE the heavy network call
    history_result = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    history = history_result.all()

    # 3. Network: Heavy call to the Gemini API (outside the transaction)
    # If this crashes, the function explodes and NOTHING gets saved to the DB.
    model_response_text = await generate_response(content, list(history))

    # 4. Short transaction: Save BOTH messages atomically
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content
    )
    db.add(user_message)
    
    model_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.MODEL,
        content=model_response_text
    )
    db.add(model_message)
    
    await db.commit()
    await db.refresh(model_message)

    return model_message

async def stream_chat_message(db: AsyncSession, conversation_id: str, content: str):
    """
    Generator that orchestrates the streaming chat flow:
    1. Validates and fetches history.
    2. Streams chunks from LLM to the client.
    3. Accumulates the full response in memory.
    4. Persists everything atomically once the stream finishes.
    """
    # 1. Validate and fetch history
    conversation = await db.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    history_result = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    history = history_result.all()

    # 2. Start the LLM stream
    stream = generate_response_stream(content, list(history))
    
    full_response = ""
    
    # 3. Yield chunks as they arrive and accumulate them
    async for chunk in stream:
        full_response += chunk
        yield chunk
        
    # 4. Stream finished: Atomic persistence
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content
    )
    db.add(user_message)
    
    model_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.MODEL,
        content=full_response
    )
    db.add(model_message)
    
    await db.commit()
