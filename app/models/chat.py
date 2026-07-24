import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Text, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)

class MessageRole(str, enum.Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"
    TOOL = "tool"

class Conversation(Base):
    __tablename__ = "conversations"

    # SQLAlchemy 2.0: Using Mapped and mapped_column for strict typing
    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_summarized_message_id: Mapped[str | None] = mapped_column(String, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="conversation", cascade="all, delete-orphan"
    )
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    
    role: Mapped[MessageRole] = mapped_column(String, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # JSON-serialized list of function calls requested by the model in this turn
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    # The name of the tool associated with the function response (only for TOOL role)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    # The unique call ID from the model (optional)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Full Turn payload structure containing all parts (text, function calls, and thought signatures).
    # Storing the raw parts list preserves the binary thought_signature needed by Gemini 2.0/2.5.
    parts: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    conversation: Mapped["Conversation | None"] = relationship("Conversation", back_populates="document_chunks")
