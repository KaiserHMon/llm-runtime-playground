import pytest
from sqlalchemy import select
from app.models.chat import Conversation
from app.models.document import DocumentChunk
from app.services.rag_service import ingest_document, search_chunks
from app.services.chat_service import process_chat_message

@pytest.mark.asyncio
async def test_rag_database_operations(db_session):
    """Test RAG ingestion, semantic chunk search, chat integration, and cascade deletion."""
    # 1. Create a test conversation
    conversation = Conversation(title="RAG Test Conversation")
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)

    # 2. Ingest a document
    doc_name = "secret_project_info.txt"
    doc_content = (
        "Project antigravity is a highly confidential AI engineering initiative.\n"
        "The master security code for the quantum generator is: quantum-antigravity-9988.\n"
        "Only personnel with Level 5 clearance are permitted to access the chamber.\n"
        "The secondary code for the ventilation system is vent-5544."
    )
    document = await ingest_document(db_session, name=doc_name, content=doc_content, conversation_id=conversation.id)
    assert document.id is not None

    # Verify chunks exist in db
    chunks_res = await db_session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    chunks_list = list(chunks_res.all())
    assert len(chunks_list) > 0

    # 3. Search chunks semantic test
    query = "What is the master security code for the quantum generator?"
    found_chunks = await search_chunks(db_session, query=query, conversation_id=conversation.id, top_k=2)
    assert len(found_chunks) > 0
    assert any("quantum-antigravity-9988" in c.content for c in found_chunks)

    # 4. Process chat message (RAG integration with Mock/Gemini - we use default provider which resolves to Gemini,
    # but we can force 'mock' to make it fully local and key-independent!)
    # Let's specify provider_name="mock" to make the test deterministic and offline-capable.
    chat_query = "Please tell me what the master security code is. Be brief."
    model_msg = await process_chat_message(
        db_session, 
        conversation_id=conversation.id, 
        content=chat_query, 
        provider_name="mock"
    )
    model_content = model_msg.content or ""
    # Our MockProvider appends RAG context if present: "Contexto RAG recuperado:"
    assert "contexto rag recuperado" in model_content.lower()
    assert "9988" in model_content

    # 5. Test Cascading Delete
    await db_session.delete(document)
    await db_session.commit()

    # Check that the chunks are gone
    chunks_left = await db_session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    chunks_left_list = list(chunks_left.all())
    assert len(chunks_left_list) == 0


@pytest.mark.asyncio
async def test_rag_http_endpoints(api_client):
    """Test the document ingestion and chat endpoints over HTTP."""
    # 1. List existing documents (should be empty initially)
    response = await api_client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    
    # 2. Upload a new document via API
    upload_payload = {
        "name": "api_test_doc.txt",
        "content": "This is a document uploaded via the HTTP API. The key passphrase is banana-split-100.",
        "conversation_id": None
    }
    response = await api_client.post("/documents/upload", json=upload_payload)
    assert response.status_code == 200
    doc_resp = response.json()
    assert doc_resp["name"] == "api_test_doc.txt"

    # List documents again to verify inclusion
    response = await api_client.get("/documents")
    docs = response.json()
    assert len(docs) > 0
    assert any(d["name"] == "api_test_doc.txt" for d in docs)

    # 3. Create a conversation for HTTP chat
    conv_response = await api_client.post("/conversations", json={"title": "HTTP RAG Chat"})
    assert conv_response.status_code == 200
    conv = conv_response.json()
    conv_id = conv["id"]

    # Send message using "mock" provider to keep tests fast and API-key independent
    msg_payload = {
        "content": "What is the key passphrase from the HTTP API document?",
        "provider": "mock"
    }
    response = await api_client.post(f"/conversations/{conv_id}/messages", json=msg_payload)
    assert response.status_code == 200
    msg_resp = response.json()
    assert "banana-split" in msg_resp["content"].lower()

    # 4. Delete document
    response = await api_client.delete("/documents/api_test_doc.txt")
    assert response.status_code == 200

    # Verify it's gone
    response = await api_client.get("/documents")
    docs = response.json()
    assert not any(d["name"] == "api_test_doc.txt" for d in docs)
