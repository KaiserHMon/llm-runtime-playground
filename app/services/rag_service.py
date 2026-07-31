import asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.services.embedding import embedding_factory

from qdrant_client import AsyncQdrantClient, models

# Initialize Qdrant Client
if settings.QDRANT_URL:
    qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
elif settings.QDRANT_PATH == ":memory:":
    qdrant_client = AsyncQdrantClient(location=":memory:")
else:
    qdrant_client = AsyncQdrantClient(path=settings.QDRANT_PATH)

QDRANT_COLLECTION = "document_chunks"
VECTOR_SIZE = 768

async def init_qdrant():
    """Initializes Qdrant by ensuring the collection exists."""
    collections_response = await qdrant_client.get_collections()
    collection_names = [col.name for col in collections_response.collections]
    if QDRANT_COLLECTION not in collection_names:
        await qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )

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
            return [text_to_split[i : i + chunk_size] for i in range(0, len(text_to_split), chunk_size - overlap)]
        
        separator = separators[0]
        next_separators = separators[1:]
        
        if separator == "":
            splits = list(text_to_split)
        else:
            splits = text_to_split.split(separator)
            
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
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                sub_chunks = _split(s, next_separators)
                chunks.extend(sub_chunks)
            else:
                if current_len + len(s) > chunk_size:
                    chunks.append("".join(current_chunk))
                    
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

async def get_embedding(text: str, provider_name: str | None = None) -> list[float]:
    """
    Asynchronously retrieves the embedding of the input text using the configured embedding provider.
    """
    provider = embedding_factory.get_provider(provider_name)
    return await provider.get_embedding(text)

async def search_chunks(
    db: AsyncSession,
    query: str,
    conversation_id: str | None = None,
    top_k: int = 5,
    embedding_provider: str | None = None,
    score_threshold: float | None = None
) -> list[DocumentChunk]:
    """
    Searches document chunks matching the query.
    Filters chunks where conversation_id IS NULL OR conversation_id == current.
    Calculates cosine similarities in Qdrant and returns the top-k chunks from SQLite.
    """
    if not query:
        return []
        
    query_emb = await get_embedding(query, provider_name=embedding_provider)
    
    filter_condition = models.Filter(
        should=[
            models.FieldCondition(
                key="conversation_id",
                match=models.MatchValue(value=conversation_id)
            ) if conversation_id else models.IsNullCondition(is_null=models.PayloadField(key="conversation_id")),
            models.IsNullCondition(is_null=models.PayloadField(key="conversation_id"))
        ]
    )
    
    # Default score threshold for non-mock providers (e.g. Gemini) to filter out irrelevant context
    if score_threshold is None and embedding_provider != "mock":
        score_threshold = 0.70
        
    search_result = await qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_emb,
        query_filter=filter_condition,
        limit=top_k,
        score_threshold=score_threshold
    )
    
    if not search_result or not search_result.points:
        return []
        
    chunk_ids = [str(point.id) for point in search_result.points]
    
    stmt = (
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    )
    result = await db.scalars(stmt)
    chunks = {chunk.id: chunk for chunk in result.all()}
    
    # Return chunks in the order returned by Qdrant
    ordered_chunks = []
    for point in search_result.points:
        chunk_id = str(point.id)
        if chunk_id in chunks:
            ordered_chunks.append(chunks[chunk_id])
            
    return ordered_chunks

async def delete_document(db: AsyncSession, name: str) -> bool:
    """
    Deletes a document by its unique name, removing its chunks from Qdrant vector DB
    and cascading deletions in SQLite. Returns True if deleted, False if not found.
    """
    doc = await db.scalar(select(Document).where(Document.name == name))
    if not doc:
        return False
        
    await qdrant_client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=doc.id)
                )
            ]
        )
    )
    await db.delete(doc)
    return True

async def ingest_document(
    db: AsyncSession,
    name: str,
    content: str,
    conversation_id: str | None = None,
    embedding_provider: str | None = None
) -> Document:
    """
    Ingests a document. First removes any existing document with the same name (cascade deletes chunks),
    splits content, generates embeddings concurrently, creates database records, and commits.
    If any step fails, rolls back the transaction.
    """
    try:
        # Delete existing document with the same name
        await delete_document(db, name)
        await db.flush()
            
        # Split text into chunks
        chunks = split_text(content, chunk_size=500, overlap=100)
        
        # Generate embeddings in parallel
        embeddings = []
        if chunks:
            embeddings = await asyncio.gather(*(get_embedding(chunk, provider_name=embedding_provider) for chunk in chunks))
            
        # Create Document and DocumentChunk database entries
        db_doc = Document(name=name, content=content)
        db.add(db_doc)
        await db.flush()  # Populate db_doc.id
        
        points = []
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            db_chunk = DocumentChunk(
                id=chunk_id,
                document_id=db_doc.id,
                conversation_id=conversation_id,
                chunk_index=idx,
                content=chunk_text
            )
            db.add(db_chunk)
            
            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=emb,
                    payload={
                        "document_id": db_doc.id,
                        "document_name": db_doc.name,
                        "conversation_id": conversation_id,
                        "chunk_index": idx,
                        "content": chunk_text
                    }
                )
            )
            
        if points:
            await qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            
        await db.commit()
        await db.refresh(db_doc)
        return db_doc
        
    except Exception as e:
        await db.rollback()
        raise e
