import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Text, Integer
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

class Conversation(Base):
    __tablename__ = "conversations"

    # SQLAlchemy 2.0: Using Mapped and mapped_column for strict typing
    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_uuid, index=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    
    role: Mapped[MessageRole] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
