from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.chat import Conversation, Message, MessageRole
from app.services.llm import generate_response


def process_chat_message(db: Session, conversation_id: str, content: str) -> Message:
    """
    Orchestrates the chat flow:
    1. Saves the user message and commits.
    2. Fetches history.
    3. Calls LLM outside the transaction.
    4. Saves the LLM response and commits.
    """
    # 1. Validate that the conversation exists
    conversation = db.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    # 2. Short transaction: Save the user message
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content
    )
    db.add(user_message)
    db.commit()  # <- CRITICAL: Release the connection before network call
    db.refresh(user_message)

    # 3. Read-only: Fetch historical messages (excluding the current one since llm.py appends it)
    history = db.scalars(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.id != user_message.id
        )
        .order_by(Message.created_at.asc())
    ).all()

    # 4. Network: Heavy call to the Gemini API (outside the transaction)
    model_response_text = generate_response(content, list(history))

    # 5. Short transaction: Save the model response
    model_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.MODEL,
        content=model_response_text
    )
    db.add(model_message)
    db.commit()
    db.refresh(model_message)

    return model_message
