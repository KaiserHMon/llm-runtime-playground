from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Using local SQLite for the MVP with the async driver.
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./llm_runtime.db"

# check_same_thread=False is required for SQLite in FastAPI
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# SQLAlchemy 2.0: Modern way to define the base class
class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
