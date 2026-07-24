import os
import asyncio
import sys
import httpx

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override database to use test_runtime.db for safety
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core import database as db_module

test_db_url = "sqlite+aiosqlite:///./test_runtime.db"
test_engine = create_async_engine(test_db_url, connect_args={"check_same_thread": False})
db_module.engine = test_engine
db_module.AsyncSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)

from app.core.database import Base  # noqa: E402
from app.models.chat import Conversation, DocumentChunk  # noqa: E402
from app.services.rag_service import ingest_document, search_chunks  # noqa: E402
from app.services.chat_service import process_chat_message  # noqa: E402
from main import app as fastapi_app  # noqa: E402

async def run_tests():
    print("=== Starting RAG Integration Tests ===")
    
    # 1. Initialize test database tables
    async with test_engine.begin() as conn:
        # Drop and recreate tables to start fresh
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
    print("[1] Test database initialized.")
    
    # Create session
    async with db_module.AsyncSessionLocal() as db:
        # 2. Create a test conversation
        conversation = Conversation(title="RAG Test Conversation")
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        print(f"[2] Conversation created with ID: {conversation.id}")
        
        # 3. Ingest a document
        doc_name = "secret_project_info.txt"
        doc_content = (
            "Project antigravity is a highly confidential AI engineering initiative.\n"
            "The master security code for the quantum generator is: quantum-antigravity-9988.\n"
            "Only personnel with Level 5 clearance are permitted to access the chamber.\n"
            "The secondary code for the ventilation system is vent-5544."
        )
        print(f"[3] Ingesting document '{doc_name}'...")
        document = await ingest_document(db, name=doc_name, content=doc_content, conversation_id=conversation.id)
        print(f"    Document ingested successfully. ID: {document.id}")
        
        # Verify chunks exist in db
        from sqlalchemy import select
        chunks_res = await db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        chunks_list = list(chunks_res.all())
        print(f"    Total chunks created in DB: {len(chunks_list)}")
        assert len(chunks_list) > 0, "No chunks created!"
        
        # 4. Search chunks semantic test
        query = "What is the master security code for the quantum generator?"
        print(f"[4] Testing semantic chunk search for query: '{query}'")
        found_chunks = await search_chunks(db, query=query, conversation_id=conversation.id, top_k=2)
        print(f"    Chunks found: {len(found_chunks)}")
        for idx, chunk in enumerate(found_chunks, 1):
            print(f"    Chunk {idx} (Source: {chunk.document.name if chunk.document else 'None'}): {chunk.content.strip()}")
            
        assert any("quantum-antigravity-9988" in c.content for c in found_chunks), "Target chunk was not retrieved!"
        print("    Semantic search verification passed.")
        
        # 5. Process chat message (RAG integration)
        chat_query = "Please tell me what the master security code is. Be brief."
        print(f"[5] Sending user query to chat service: '{chat_query}'")
        model_msg = await process_chat_message(db, conversation_id=conversation.id, content=chat_query)
        print(f"    Model response:\n{model_msg.content}")
        
        # We expect the model to mention the code and cite the source
        model_content = model_msg.content or ""
        assert "9988" in model_content, "Model did not retrieve the correct code!"
        assert "[" in model_content, "Model did not include a citation (e.g. [1])!"
        print("    Chat RAG context injection and citation verification passed.")
        
        # 6. Test Cascading Delete
        print(f"[6] Deleting document '{doc_name}' to test cascade delete...")
        await db.delete(document)
        await db.commit()
        
        # Check that the chunks are gone
        chunks_left = await db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
        chunks_left_list = list(chunks_left.all())
        print(f"    Chunks remaining for deleted document: {len(chunks_left_list)}")
        assert len(chunks_left_list) == 0, "Chunks were not cascade-deleted!"
        print("    Cascade delete verification passed.")

    # 7. Run HTTP Endpoint Tests
    print("\n=== Starting HTTP Endpoint Tests ===")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test") as client:
        # List existing documents (should be empty initially)
        response = await client.get("/documents")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        docs = response.json()
        print(f"    Initial documents list length: {len(docs)}")
        
        # Upload a new document via API
        upload_payload = {
            "name": "api_test_doc.txt",
            "content": "This is a document uploaded via the HTTP API. The key passphrase is banana-split-100.",
            "conversation_id": None
        }
        print("    Uploading document via POST /documents/upload...")
        response = await client.post("/documents/upload", json=upload_payload)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        doc_resp = response.json()
        assert doc_resp["name"] == "api_test_doc.txt"
        print(f"    Uploaded document ID: {doc_resp['id']}")
        
        # List documents again
        response = await client.get("/documents")
        docs = response.json()
        assert len(docs) > 0, "No documents listed after upload"
        assert any(d["name"] == "api_test_doc.txt" for d in docs)
        print(f"    List documents count: {len(docs)}")
        
        # Check if we can search and chat using the uploaded document context
        # We start a conversation first via POST /conversations
        conv_response = await client.post("/conversations", json={"title": "HTTP RAG Chat"})
        assert conv_response.status_code == 200
        conv = conv_response.json()
        conv_id = conv["id"]
        print(f"    Created conversation for HTTP chat: {conv_id}")
        
        # Send message
        msg_payload = {"content": "What is the key passphrase from the HTTP API document?"}
        print("    Sending chat message via POST /conversations/{id}/messages...")
        response = await client.post(f"/conversations/{conv_id}/messages", json=msg_payload)
        assert response.status_code == 200, f"Chat failed: {response.text}"
        msg_resp = response.json()
        print(f"    Model HTTP RAG response:\n{msg_resp['content']}")
        assert "banana-split" in msg_resp["content"].lower(), "Model did not recall information from the uploaded API document"
        
        # Delete document
        print("    Deleting document via DELETE /documents/{name}...")
        response = await client.delete("/documents/api_test_doc.txt")
        assert response.status_code == 200, f"Delete failed: {response.text}"
        print("    Document deleted.")
        
        # Verify it's gone
        response = await client.get("/documents")
        docs = response.json()
        assert not any(d["name"] == "api_test_doc.txt" for d in docs), "Document still exists after deletion"
        print("    HTTP API tests passed!")

    print("\n=== All Integration Tests Completed Successfully! ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
