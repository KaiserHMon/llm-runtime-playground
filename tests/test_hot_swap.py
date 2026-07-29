import pytest
from app.models.chat import Conversation
from app.services.chat_service import process_chat_message

@pytest.mark.asyncio
async def test_mock_provider_plain_text(db_session):
    """Test plain text response using the Mock provider."""
    conversation = Conversation(title="Hot-Swap Test Conversation")
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    msg = await process_chat_message(
        db_session,
        conversation_id=conversation.id,
        content="Hola robot",
        provider_name="mock"
    )
    assert msg.content is not None
    assert "[Mock Provider]" in msg.content
    assert "Hola robot" in msg.content

@pytest.mark.asyncio
async def test_mock_provider_tool_calling(db_session):
    """Test tool calling simulation using the Mock provider (weather trigger)."""
    conversation = Conversation(title="Hot-Swap Test Conversation")
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    msg_tool = await process_chat_message(
        db_session,
        conversation_id=conversation.id,
        content="Tell me the weather please",
        provider_name="mock"
    )
    assert msg_tool.content is not None
    assert "[Mock Provider]" in msg_tool.content
    assert "weather" in msg_tool.content.lower() or "sunny" in msg_tool.content.lower()
