import os
os.environ["QDRANT_PATH"] = ":memory:"

import pytest_asyncio
import httpx

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core import database as db_module
from app.core.database import Base
from main import app as fastapi_app

test_db_url = "sqlite+aiosqlite:///./test_runtime.db"

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    # Override database engine and sessionmaker to use the test database
    test_engine = create_async_engine(test_db_url, connect_args={"check_same_thread": False})
    db_module.engine = test_engine
    db_module.AsyncSessionLocal = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # Initialize and clean database tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
    from app.services.rag_service import init_qdrant
    await init_qdrant()
    
    yield
    
    # Teardown database tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()

@pytest_asyncio.fixture
async def db_session():
    # Session fixture yielding a clean session for each test case
    async with db_module.AsyncSessionLocal() as session:
        yield session

@pytest_asyncio.fixture
async def api_client():
    # Client fixture yielding an async httpx client connected to the FastAPI application
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test") as client:
        yield client
