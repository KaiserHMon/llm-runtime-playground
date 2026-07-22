import asyncio
import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.chat import Conversation, Message, MessageRole
from app.services.llm import generate_response, count_tokens, summarize_messages
from app.services.tools import registry

TOKEN_BUDGET = 4000
TOKEN_BUFFER_PER_MESSAGE = 20
MAX_TOOL_LOOP_ITERATIONS = 5

async def ensure_message_tokens_and_slice(db: AsyncSession, history: list[Message]) -> list[Message]:
    """
    Ensures all messages in history have a valid `tokens` count (populating missing ones
    using Gemini API count_tokens or fallback len(content)//4), updates the database,
    and returns the sliced history that fits within the 4000 token budget (with 20 tokens safety buffer per message).
    """
    # 1. Fill in missing tokens
    missing_msgs = [msg for msg in history if msg.tokens is None]
    if missing_msgs:
        try:
            token_counts = await asyncio.gather(*(count_tokens(msg.content or "") for msg in missing_msgs))
            for msg, count in zip(missing_msgs, token_counts):
                msg.tokens = count
            await db.commit()
        except Exception:
            # Fallback to len // 4
            for msg in missing_msgs:
                msg.tokens = len(msg.content) // 4 if msg.content else 0
            try:
                await db.commit()
            except Exception:
                pass

    # 2. Slice history from the newest end to fit within 4000 tokens budget (including 20 tokens safety buffer per message)
    current_sum = 0
    sliced_history = []
    
    # We iterate from the newest (end of list) to oldest
    for msg in reversed(history):
        msg_tokens = msg.tokens if msg.tokens is not None else (len(msg.content) // 4 if msg.content else 0)
        cost = msg_tokens + TOKEN_BUFFER_PER_MESSAGE
        if current_sum + cost <= TOKEN_BUDGET:
            current_sum += cost
            sliced_history.append(msg)
        else:
            break
            
    # Reverse it back to chronological order (oldest to newest)
    sliced_history.reverse()
    return sliced_history

async def get_active_history(db: AsyncSession, conversation_id: str) -> tuple[list[Message], Conversation]:
    """
    Validates conversation existence, fetches chronological history,
    ensures token counts are populated, and slices it to fit the token budget.
    Also handles incremental summarization for messages excluded from the active history.
    """
    conversation = await db.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    history_result = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    history = list(history_result.all())

    sliced_history = await ensure_message_tokens_and_slice(db, history)

    # Incremental Summarization logic
    if sliced_history:
        oldest_sliced = sliced_history[0]
        try:
            oldest_sliced_idx = history.index(oldest_sliced)
        except ValueError:
            oldest_sliced_idx = len(history)

        messages_to_summarize = []
        if oldest_sliced_idx > 0:
            if conversation.last_summarized_message_id:
                last_sum_idx = -1
                for idx, msg in enumerate(history):
                    if msg.id == conversation.last_summarized_message_id:
                        last_sum_idx = idx
                        break
                if last_sum_idx != -1:
                    messages_to_summarize = history[last_sum_idx + 1 : oldest_sliced_idx]
                else:
                    messages_to_summarize = history[:oldest_sliced_idx]
            else:
                messages_to_summarize = history[:oldest_sliced_idx]

        if messages_to_summarize:
            new_summary = await summarize_messages(conversation.summary, messages_to_summarize)
            conversation.summary = new_summary
            conversation.last_summarized_message_id = messages_to_summarize[-1].id
            db.add(conversation)
            await db.commit()

    return sliced_history, conversation


async def execute_tool(db: AsyncSession, tool_name: str, args: dict) -> str:
    """
    Executes a registered Python tool function dynamically.
    
    If the function signature requests a 'db' parameter, the active AsyncSession 
    connection is dynamically injected. Errors are captured and returned as strings, 
    and output length is capped to protect context window token budget.
    """
    tool_func = registry.get_tool(tool_name)
    if not tool_func:
        return f"Error: Tool {tool_name} not found"
        
    try:
        sig = inspect.signature(tool_func)
        call_args = dict(args)
        # Dynamically inject active database session if expected by the tool signature
        if "db" in sig.parameters:
            call_args["db"] = db
        result = await tool_func(**call_args)
        
        result_str = str(result)
        # Cap tool output size to protect context token limits and avoid bloat
        if len(result_str) > 1000:
            result_str = result_str[:997] + "..."
        return result_str
    except Exception as e:
        # Gracefully catch tool exceptions and return error text so the model can handle it
        return f"Error executing tool {tool_name}: {str(e)}"



async def process_chat_message(db: AsyncSession, conversation_id: str, content: str) -> Message:
    """
    Orchestrates the chat turn flow with manual tool execution loops.
    
    This manages a multi-turn conversation turn sequence using Gemini API.
    If the model generates a function call turn (MODEL role), this backend intercepts it,
    executes the requested tool functions, and appends the result turn (TOOL role)
    to the history, continuing the cycle until the model generates a final text response.
    
    All messages (user prompt, intermediate tool requests, tool outputs, and final text)
    are saved to the database in a single atomic commit at the end.
    """
    sliced_history, conversation = await get_active_history(db, conversation_id)
    
    user_tokens = await count_tokens(content)
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=content,
        tokens=user_tokens,
        parts=[{"text": content}]
    )
    
    pending_messages = [user_message]
    current_history = sliced_history.copy()
    current_history.append(user_message)
    
    final_model_message = None
    
    for i in range(MAX_TOOL_LOOP_ITERATIONS):
        response = await generate_response(current_history, summary=conversation.summary)
        
        if response.function_calls:
            tool_calls_json = [{"name": fc.name, "args": fc.args} for fc in response.function_calls]
            parts_json = [p.model_dump(mode="json") for p in response.candidates[0].content.parts]
                
            model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=response.text,
                tool_calls=tool_calls_json,
                parts=parts_json,
                tokens=await count_tokens(response.text) if response.text else 0
            )
            pending_messages.append(model_message)
            current_history.append(model_message)
            
            for fc in response.function_calls:
                result_str = await execute_tool(db, fc.name, fc.args)
                
                from google.genai import types
                part_dict = types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_str}
                ).model_dump(mode="json")
                
                tool_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.TOOL,
                    content=result_str,
                    tool_name=fc.name,
                    parts=[part_dict],
                    tokens=await count_tokens(result_str)
                )
                pending_messages.append(tool_msg)
                current_history.append(tool_msg)
        else:
            parts_json = [p.model_dump(mode="json") for p in response.candidates[0].content.parts]
            final_model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=response.text,
                parts=parts_json,
                tokens=await count_tokens(response.text) if response.text else 0
            )
            pending_messages.append(final_model_message)
            break
            
    if not final_model_message:
        final_model_message = pending_messages[-1]
        
    for msg in pending_messages:
        db.add(msg)
    await db.commit()
    await db.refresh(final_model_message)
    return final_model_message

async def stream_chat_message(db: AsyncSession, conversation_id: str, content: str):
    # Streaming with tools is complex because we may need to execute tools mid-stream.
    # For this MVP phase, we run the same execution loop without chunking, 
    # but yield the final text response chunks to satisfy the stream interface.
    final_message = await process_chat_message(db, conversation_id, content)
    if final_message.content:
        # yield artificial chunks to satisfy the API
        chunk_size = 20
        for i in range(0, len(final_message.content), chunk_size):
            yield final_message.content[i:i+chunk_size]
