import pytest
from app.models.chat import Message, MessageRole
from app.services.llm.router import route_message

@pytest.mark.asyncio
async def test_router_chat_route():
    # A generic message should be routed to CHAT
    decision = await route_message("Hola, ¿cómo estás?", provider_name="mock")
    assert decision == "CHAT"

@pytest.mark.asyncio
async def test_router_rag_route_by_content():
    # A message containing keywords should be routed to RAG
    decision = await route_message("¿Cuál es el código de ventilación?", provider_name="mock")
    assert decision == "RAG"

@pytest.mark.asyncio
async def test_router_rag_route_by_history():
    # A message itself might not contain a keyword, but history does
    history = [
        Message(
            conversation_id="temp",
            role=MessageRole.USER,
            content="La clave secreta de antigravity es banana-split"
        ),
        Message(
            conversation_id="temp",
            role=MessageRole.MODEL,
            content="Entendido, guardado."
        )
    ]
    decision = await route_message("¿Me la repetís?", history=history, provider_name="mock")
    assert decision == "RAG"
