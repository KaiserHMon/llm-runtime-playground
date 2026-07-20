import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.chat import Conversation, Message, MessageRole
from app.services.llm import generate_response, generate_response_stream, count_tokens

TOKEN_BUDGET = 4000
TOKEN_BUFFER_PER_MESSAGE = 20

async def ensure_message_tokens_and_slice(db: AsyncSession, history: list[Message]) -> list[Message]:
    """
    Ensures all messages in history have a valid `tokens` count (populating missing ones
    using Gemini API count_tokens or fallback len(content)//4), updates the database,
    and returns the sliced history that fits within the 4000 token budget (with 20 tokens safety buffer per message).
    """
    # 1. Fill in missing tokens
    missing_msgs = [msg for msg in history if msg.tokens is None]
    if missing_msgs:
        try:
            token_counts = await asyncio.gather(*(count_tokens(msg.content) for msg in missing_msgs))
            for msg, count in zip(missing_msgs, token_counts):
                msg.tokens = count
            await db.commit()
        except Exception:
            # Fallback to len // 4
            for msg in missing_msgs:
                msg.tokens = len(msg.content) // 4 if msg.content else 0
            try:
                await db.commit()
            except Exception:
                pass

    # 2. Slice history from the newest end to fit within 4000 tokens budget (including 20 tokens safety buffer per message)
    current_sum = 0
    sliced_history = []
    
    # We iterate from the newest (end of list) to oldest
    for msg in reversed(history):
        msg_tokens = msg.tokens if msg.tokens is not None else (len(msg.content) // 4 if msg.content else 0)
        cost = msg_tokens + TOKEN_BUFFER_PER_MESSAGE
        if current_sum + cost <= TOKEN_BUDGET:
            current_sum += cost
            sliced_history.append(msg)
        else:
            break
            
    # Reverse it back to chronological order (oldest to newest)
    sliced_history.reverse()
    return sliced_history

async def process_chat_message(db: AsyncSession, conversation_id: str, content: str) -> Message:
    """
    Orchestrates the chat flow asynchronously:
    1. Validate conversation.
    2. Fetches history, ensures all loaded messages have valid token counts, and slices context dynamically.
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
    history = list(history_result.all())

    # Ensure historical messages have tokens populated and slice them to fit the budget
    sliced_history = await ensure_message_tokens_and_slice(db, history)

    # 3. Network: Heavy call to the Gemini API (outside the transaction)
    # If this crashes, the function explodes and NOTHING gets saved to the DB.
    model_response_text = await generate_response(content, sliced_history)

    # Compute tokens for the new user and model messages
    user_tokens, model_tokens = await asyncio.gather(
        count_tokens(content),
        count_tokens(model_response_text)
    )

    # 4. Short transaction: Save BOTH messages atomically
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content,
        tokens=user_tokens
    )
    db.add(user_message)
    
    model_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.MODEL,
        content=model_response_text,
        tokens=model_tokens
    )
    db.add(model_message)
    
    await db.commit()
    await db.refresh(model_message)

    return model_message

async def stream_chat_message(db: AsyncSession, conversation_id: str, content: str):
    """
    Generator that orchestrates the streaming chat flow:
    1. Validates and fetches history, ensures tokens are populated, and slices context.
    2. Streams chunks from LLM to the client.
    3. Accumulates the full response in memory.
    4. Persists everything atomically once the stream finishes with token counts.
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
    history = list(history_result.all())

    # Ensure historical messages have tokens populated and slice them to fit the budget
    sliced_history = await ensure_message_tokens_and_slice(db, history)

    # 2. Start the LLM stream
    stream = generate_response_stream(content, sliced_history)
    
    full_response = ""
    
    # 3. Yield chunks as they arrive and accumulate them
    async for chunk in stream:
        full_response += chunk
        yield chunk
        
    # 4. Stream finished: Atomic persistence
    user_tokens, model_tokens = await asyncio.gather(
        count_tokens(content),
        count_tokens(full_response)
    )

    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content,
        tokens=user_tokens
    )
    db.add(user_message)
    
    model_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.MODEL,
        content=full_response,
        tokens=model_tokens
    )
    db.add(model_message)
    
    await db.commit()
