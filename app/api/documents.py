from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.schemas.document import DocumentUpload, DocumentResponse
from app.models.document import Document
from app.services.rag_service import ingest_document, delete_document as delete_doc_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    payload: DocumentUpload,
    conversation_id: str | None = Query(None, description="Optional conversation context to bind the document chunks to"),
    db: AsyncSession = Depends(get_db)
):
    """
    Uploads and processes a document. Splitting, embedding generation,
    and storage are handled inside the transactional ingestion service.
    """
    cid = payload.conversation_id or conversation_id
    try:
        doc = await ingest_document(db, name=payload.name, content=payload.content, conversation_id=cid)
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    Lists all ingested documents.
    """
    result = await db.scalars(select(Document).order_by(Document.created_at.desc()))
    return list(result.all())

@router.delete("/{name}")
async def delete_document(name: str, db: AsyncSession = Depends(get_db)):
    """
    Deletes a document by its unique name, triggering cascading deletion of its chunks.
    """
    try:
        deleted = await delete_doc_service(db, name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{name}' not found")
        await db.commit()
        return {"status": "success", "message": f"Document '{name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
