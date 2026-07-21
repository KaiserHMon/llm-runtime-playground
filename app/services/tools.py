from typing import Callable, Dict, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.chat import Conversation

class ToolRegistry:
    """
    Registry for Python function tools that the LLM is allowed to invoke.
    Decouples raw Python functions from Gemini SDK declarations.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        
    def register(self, func: Callable) -> Callable:
        """Registers a Python callable tool with its name as the lookup key."""
        self._tools[func.__name__] = func
        return func
        
    def get_tool(self, name: str) -> Callable | None:
        """Retrieves a registered Python tool function by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Callable]:
        """Returns a list of all registered tool callables."""
        return list(self._tools.values())

registry = ToolRegistry()

def tool(func: Callable) -> Callable:
    """Decorator to mark and register a Python function as an executable tool."""
    return registry.register(func)

@tool
async def get_current_datetime() -> str:
    """Returns the current date and time in UTC format."""
    return datetime.now(timezone.utc).isoformat()

@tool
async def list_conversations(db: AsyncSession) -> list[dict]:
    """Lists all active conversations in the system.
    
    Returns a list of conversation details.
    """
    result = await db.scalars(select(Conversation).order_by(Conversation.created_at.desc()))
    conversations = result.all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in conversations]

@tool
async def get_weather(location: str) -> str:
    """Gets the current weather for a specific location."""
    return f"The weather in {location} is 72F and sunny."
