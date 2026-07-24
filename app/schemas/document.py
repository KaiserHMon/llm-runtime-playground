from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class DocumentUpload(BaseModel):
    """Payload to upload a new document."""
    name: str = Field(..., description="Unique name/filename of the document")
    content: str = Field(..., description="Full text content of the document")
    conversation_id: str | None = Field(default=None, description="Optional conversation context to bind the document to")

class DocumentResponse(BaseModel):
    """Response model for document metadata."""
    id: str
    name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DocumentChunkResponse(BaseModel):
    """Response model for document chunks."""
    id: str
    document_id: str
    conversation_id: str | None
    chunk_index: int
    content: str

    model_config = ConfigDict(from_attributes=True)
