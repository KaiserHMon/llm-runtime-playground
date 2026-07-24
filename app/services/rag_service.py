import asyncio
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from sqlalchemy.orm import joinedload

from app.models.chat import Document, DocumentChunk
from app.services.llm import client

def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits text into chunks of maximum size chunk_size, with specified overlap,
    recursively trying a list of delimiters: ["\n\n", "\n", " ", ""].
    """
    if not text:
        return []
        
    delimiters = ["\n\n", "\n", " ", ""]
    
    def _split(text_to_split: str, separators: list[str]) -> list[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
        if not separators:
            # Force split by characters if no separators left
            return [text_to_split[i : i + chunk_size] for i in range(0, len(text_to_split), chunk_size - overlap)]
        
        separator = separators[0]
        next_separators = separators[1:]
        
        # Split text by separator
        if separator == "":
            splits = list(text_to_split)
        else:
            splits = text_to_split.split(separator)
            
        # Re-add separator where it was (except for the last split)
        final_splits = []
        for i, s in enumerate(splits):
            if separator != "" and i < len(splits) - 1:
                final_splits.append(s + separator)
            else:
                final_splits.append(s)
                
        chunks = []
        current_chunk = []
        current_len = 0
        
        for s in final_splits:
            if not s:
                continue
            if len(s) > chunk_size:
                # If a single split is larger than chunk_size, we split it recursively
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Recursively split the long block
                sub_chunks = _split(s, next_separators)
                chunks.extend(sub_chunks)
            else:
                if current_len + len(s) > chunk_size:
                    # Current chunk is full, emit it
                    chunks.append("".join(current_chunk))
                    
                    # Retain overlap: take last elements from current_chunk up to overlap limit
                    overlap_chunk = []
                    overlap_len = 0
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) <= overlap:
                            overlap_chunk.insert(0, item)
                            overlap_len += len(item)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len
                
                current_chunk.append(s)
                current_len += len(s)
                
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks

    return _split(text, delimiters)

async def get_embedding(text: str) -> list[float]:
    """
    Asynchronously retrieves the embedding of the input text using Gemini's embedding model.
    Defaults to gemini-embedding-2 and falls back to gemini-embedding-001 if needed.
    """
    if not text:
        return []
    
    try:
        response = await client.aio.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        if response.embeddings and response.embeddings[0].values is not None:
            return response.embeddings[0].values
    except Exception:
        # Fall back to gemini-embedding-001 if gemini-embedding-2 is not available
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        if response.embeddings and response.embeddings[0].values is not None:
            return response.embeddings[0].values
            
    raise ValueError("Failed to retrieve embedding values from Gemini API response.")

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Computes the cosine similarity between two float vectors in pure Python.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_product / (mag1 * mag2)

async def search_chunks(db: AsyncSession, query: str, conversation_id: str | None = None, top_k: int = 5) -> list[DocumentChunk]:
    """
    Searches document chunks matching the query.
    Filters chunks where conversation_id IS NULL OR conversation_id == current.
    Calculates cosine similarities in memory and returns the top-k chunks.
    """
    if not query:
        return []
        
    query_emb = await get_embedding(query)
    
    # Query matching chunks
    stmt = (
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(
            or_(
                DocumentChunk.conversation_id.is_(None),
                DocumentChunk.conversation_id == conversation_id
            )
        )
    )
    result = await db.scalars(stmt)
    chunks = list(result.all())
    
    # Compute similarity and sort
    scored_chunks = []
    for chunk in chunks:
        sim = cosine_similarity(query_emb, chunk.embedding)
        scored_chunks.append((chunk, sim))
        
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, sim in scored_chunks[:top_k]]

async def ingest_document(db: AsyncSession, name: str, content: str, conversation_id: str | None = None) -> Document:
    """
    Ingests a document. First removes any existing document with the same name (cascade deletes chunks),
    splits content, generates embeddings concurrently, creates database records, and commits.
    If any step fails, rolls back the transaction.
    """
    try:
        # Delete existing document with the same name
        existing_doc = await db.scalar(select(Document).where(Document.name == name))
        if existing_doc:
            await db.delete(existing_doc)
            await db.flush()
            
        # Split text into chunks
        chunks = split_text(content, chunk_size=500, overlap=100)
        
        # Generate embeddings in parallel
        embeddings = []
        if chunks:
            embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))
            
        # Create Document and DocumentChunk database entries
        db_doc = Document(name=name, content=content)
        db.add(db_doc)
        await db.flush()  # Populate db_doc.id
        
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            db_chunk = DocumentChunk(
                document_id=db_doc.id,
                conversation_id=conversation_id,
                chunk_index=idx,
                content=chunk_text,
                embedding=emb
            )
            db.add(db_chunk)
            
        await db.commit()
        await db.refresh(db_doc)
        return db_doc
        
    except Exception as e:
        await db.rollback()
        raise e
