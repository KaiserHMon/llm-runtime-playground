from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from contextlib import asynccontextmanager
from sqlalchemy import text
from app.core.database import Base, engine
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router

from app.services.rag_service import init_qdrant

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Qdrant collection
    await init_qdrant()
    
    # Initialize database tables asynchronously
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Check and migrate columns if database already exists
        def migrate_db(connection):
            cursor = connection.execute(text("PRAGMA table_info(conversations)"))
            columns = [row[1] for row in cursor.fetchall()]
            if "summary" not in columns:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT"))
            if "last_summarized_message_id" not in columns:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN last_summarized_message_id VARCHAR"))
                
            cursor = connection.execute(text("PRAGMA table_info(messages)"))
            columns = [row[1] for row in cursor.fetchall()]
            if "rag_route" not in columns:
                connection.execute(text("ALTER TABLE messages ADD COLUMN rag_route VARCHAR"))
            if "rag_sources" not in columns:
                connection.execute(text("ALTER TABLE messages ADD COLUMN rag_sources JSON"))
                
        await conn.run_sync(migrate_db)
    yield

app = FastAPI(
    title="LLM Runtime Playground",
    description="Clean Architecture implementation of an LLM backend",
    version="0.1.0",
    lifespan=lifespan
)

# Include the API routers
app.include_router(chat_router)
app.include_router(documents_router)

# Mount React build assets and serve index at root
app.mount("/assets", StaticFiles(directory="client/dist/assets"), name="assets")

@app.get("/")
def read_root():
    return FileResponse("client/dist/index.html")

