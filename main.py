from fastapi import FastAPI
from app.core.database import Base, engine
from app.api.chat import router as chat_router

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="LLM Runtime Playground",
    description="Clean Architecture implementation of an LLM backend",
    version="0.1.0"
)

# Include the API router
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Hello from llm-runtime-playground!"}
