import pytest
from sqlalchemy import select
from app.models.chat import Conversation

@pytest.mark.asyncio
async def test_automatic_title_generation(api_client, db_session):
    """
    Verifies that sending a message to a new conversation automatically triggers 
    the background task and updates the conversation's title in the DB.
    """
    # 1. Create a new conversation with a null title
    conv_response = await api_client.post("/conversations", json={"title": None})
    assert conv_response.status_code == 200
    conv = conv_response.json()
    conv_id = conv["id"]
    assert conv["title"] is None

    # 2. Call the streaming endpoint to send the first message
    msg_payload = {
        "content": "This is a brand new conversation message",
        "provider": "mock"
    }
    
    async with api_client.stream("POST", f"/conversations/{conv_id}/messages/stream", json=msg_payload) as response:
        assert response.status_code == 200
        # Consume the stream fully to trigger database commits and background task lifecycle
        async for chunk in response.aiter_text():
            pass

    # 3. Query the database to verify the title was updated asynchronously
    stmt = select(Conversation).where(Conversation.id == conv_id)
    
    # Refresh to ensure we read from DB
    db_session.expire_all()
    result = await db_session.execute(stmt)
    updated_conv = result.scalar_one_or_none()
    
    assert updated_conv is not None
    assert updated_conv.title is not None
    assert "Mock Title" in updated_conv.title
    assert "This is a brand" in updated_conv.title
