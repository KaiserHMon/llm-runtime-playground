from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.database import Base, engine
from app.api.chat import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables asynchronously
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Check and migrate columns if database already exists
        def migrate_db(connection):
            cursor = connection.execute("PRAGMA table_info(conversations)")
            columns = [row[1] for row in cursor.fetchall()]
            if "summary" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN summary TEXT")
            if "last_summarized_message_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN last_summarized_message_id VARCHAR")
                
        await conn.run_sync(migrate_db)
    yield

app = FastAPI(
    title="LLM Runtime Playground",
    description="Clean Architecture implementation of an LLM backend",
    version="0.1.0",
    lifespan=lifespan
)

# Include the API router
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Hello from llm-runtime-playground!"}
