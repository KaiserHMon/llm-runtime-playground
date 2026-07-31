import asyncio
import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.chat import Conversation, Message, MessageRole
from app.services.llm import factory
from app.services.llm.base import LLMProvider
from app.services.tools import registry
from app.services.rag_service import search_chunks
from app.services.llm.router import route_message
from app.services.guardrail_service import (
    verify_prompt_safety,
    anonymize_pii,
    deanonymize_pii,
    deanonymize_stream,
)


def estimate_message_tokens(msg: Message) -> int:
    """
    Estimates the number of tokens in a message as a fallback.
    Inspects `content` first, then falls back to serializing `parts` if content is empty/null.
    """
    if msg.content:
        return len(msg.content) // 4
    if msg.parts:
        try:
            import json
            return len(json.dumps(msg.parts)) // 4
        except Exception:
            return len(str(msg.parts)) // 4
    return 0

async def ensure_message_tokens_and_slice(
    db: AsyncSession, history: list[Message], provider: LLMProvider
) -> list[Message]:
    """
    Ensures all messages in history have a valid `tokens` count (populating missing ones
    using the provider's count_tokens or fallback estimation), updates the database,
    and returns the sliced history that fits within the configured token budget.
    """
    # 1. Fill in missing tokens
    missing_msgs = [msg for msg in history if msg.tokens is None]
    if missing_msgs:
        try:
            token_counts = await asyncio.gather(*(provider.count_tokens(msg.content or "") for msg in missing_msgs))
            for msg, count in zip(missing_msgs, token_counts):
                msg.tokens = count
            await db.commit()
        except Exception:
            # Fallback to robust token estimation logic
            for msg in missing_msgs:
                msg.tokens = estimate_message_tokens(msg)
            try:
                await db.commit()
            except Exception:
                pass

    # 2. Slice history from the newest end to fit within the configured token budget
    current_sum = 0
    sliced_history = []
    
    # We iterate from the newest (end of list) to oldest
    for msg in reversed(history):
        msg_tokens = msg.tokens if msg.tokens is not None else estimate_message_tokens(msg)
        cost = msg_tokens + settings.TOKEN_BUFFER_PER_MESSAGE
        if current_sum + cost <= settings.TOKEN_BUDGET:
            current_sum += cost
            sliced_history.append(msg)
        else:
            break
            
    # Reverse it back to chronological order (oldest to newest)
    sliced_history.reverse()
    return sliced_history

async def get_active_history(
    db: AsyncSession, conversation_id: str, provider: LLMProvider
) -> tuple[list[Message], Conversation]:
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

    sliced_history = await ensure_message_tokens_and_slice(db, history, provider)

    # Incremental Summarization logic
    if history:
        if sliced_history:
            oldest_sliced = sliced_history[0]
            try:
                oldest_sliced_idx = history.index(oldest_sliced)
            except ValueError:
                oldest_sliced_idx = len(history)
        else:
            # If sliced_history is empty, all messages in the history have been evicted
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
            new_summary = await provider.summarize_messages(conversation.summary, messages_to_summarize)
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


async def process_chat_message(
    db: AsyncSession, conversation_id: str, content: str, provider_name: str | None = None
) -> Message:
    """
    Orchestrates the chat turn flow with manual tool execution loops.
    
    This manages a multi-turn conversation turn sequence using the selected LLM provider.
    If the model generates a function call turn (MODEL role), this backend intercepts it,
    executes the requested tool functions, and appends the result turn (TOOL role)
    to the history, continuing the cycle until the model generates a final text response.
    
    All messages (user prompt, intermediate tool requests, tool outputs, and final text)
    are saved to the database in a single atomic commit at the end.
    """
    # 1. Input safety check on raw query content
    await verify_prompt_safety(content)
    
    # 2. PII anonymization on query content
    anon_content, pii_mapping = anonymize_pii(content)
    content = anon_content

    provider = factory.get_provider(provider_name)
    sliced_history, conversation = await get_active_history(db, conversation_id, provider)
    
    # Determine the route and conditionally retrieve RAG context
    route = await route_message(content, history=sliced_history, provider_name=provider_name)
    rag_context = ""
    if route == "RAG":
        emb_provider = "mock" if provider_name == "mock" else None
        chunks = await search_chunks(db, query=content, conversation_id=conversation_id, top_k=5, embedding_provider=emb_provider)
        if chunks:
            formatted_chunks = []
            for idx, chunk in enumerate(chunks, 1):
                source_name = chunk.document.name if chunk.document else "Unknown"
                formatted_chunks.append(f"[{idx}] (Source: {source_name}): {chunk.content}")
            rag_context = "\n\n".join(formatted_chunks)

    
    user_tokens = await provider.count_tokens(content)
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
    
    for i in range(settings.MAX_TOOL_LOOP_ITERATIONS):
        response = await provider.generate_response(
            current_history,
            summary=conversation.summary,
            rag_context=rag_context or None
        )
        
        if response.tool_calls:
            tool_calls_json = [{"name": tc.name, "args": tc.args} for tc in response.tool_calls]
            parts_json = response.parts
                
            model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=response.content,
                tool_calls=tool_calls_json,
                parts=parts_json,
                tokens=await provider.count_tokens(response.content) if response.content else 0
            )
            pending_messages.append(model_message)
            current_history.append(model_message)
            
            for tc in response.tool_calls:
                result_str = await execute_tool(db, tc.name, tc.args)
                
                parts_list = provider.format_tool_response(tc.name, result_str)
                
                tool_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.TOOL,
                    content=result_str,
                    tool_name=tc.name,
                    parts=parts_list,
                    tokens=await provider.count_tokens(result_str)
                )
                pending_messages.append(tool_msg)
                current_history.append(tool_msg)
        else:
            parts_json = response.parts
            final_model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=response.content,
                parts=parts_json,
                tokens=await provider.count_tokens(response.content) if response.content else 0
            )
            pending_messages.append(final_model_message)
            break
            
    if not final_model_message:
        final_model_message = pending_messages[-1]
        
    # Deanonymize final response content using the generated mapping
    if final_model_message and final_model_message.content:
        final_model_message.content = deanonymize_pii(final_model_message.content, pii_mapping)
        if final_model_message.parts:
            for part in final_model_message.parts:
                if isinstance(part, dict) and "text" in part and part["text"]:
                    part["text"] = deanonymize_pii(part["text"], pii_mapping)
        
    for msg in pending_messages:
        db.add(msg)
    await db.commit()
    await db.refresh(final_model_message)
    return final_model_message

async def stream_chat_message(db: AsyncSession, conversation_id: str, content: str, provider_name: str | None = None):
    """
    Orchestrates the chat turn flow with manual tool execution loops,
    but streams the final model response turn to the client in real-time.
    
    Like process_chat_message, intermediate tool requests and responses are executed
    synchronously. Once the final turn is reached (no more tool calls), this yields 
    response chunks as they are received from the LLM provider, saving the complete
    sequence in the database at the end.
    """
    # 1. Input safety check on raw query content
    await verify_prompt_safety(content)
    
    # 2. PII anonymization on query content
    anon_content, pii_mapping = anonymize_pii(content)
    content = anon_content

    provider = factory.get_provider(provider_name)
    sliced_history, conversation = await get_active_history(db, conversation_id, provider)
    
    # Determine the route and conditionally retrieve RAG context
    route = await route_message(content, history=sliced_history, provider_name=provider_name)
    rag_context = ""
    if route == "RAG":
        emb_provider = "mock" if provider_name == "mock" else None
        chunks = await search_chunks(db, query=content, conversation_id=conversation_id, top_k=5, embedding_provider=emb_provider)
        if chunks:
            formatted_chunks = []
            for idx, chunk in enumerate(chunks, 1):
                source_name = chunk.document.name if chunk.document else "Unknown"
                formatted_chunks.append(f"[{idx}] (Source: {source_name}): {chunk.content}")
            rag_context = "\n\n".join(formatted_chunks)

    
    user_tokens = await provider.count_tokens(content)
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
    
    for i in range(settings.MAX_TOOL_LOOP_ITERATIONS):
        response = await provider.generate_response(
            current_history,
            summary=conversation.summary,
            rag_context=rag_context or None
        )
        
        if response.tool_calls:
            tool_calls_json = [{"name": tc.name, "args": tc.args} for tc in response.tool_calls]
            parts_json = response.parts
                
            model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=response.content,
                tool_calls=tool_calls_json,
                parts=parts_json,
                tokens=await provider.count_tokens(response.content) if response.content else 0
            )
            pending_messages.append(model_message)
            current_history.append(model_message)
            
            for tc in response.tool_calls:
                result_str = await execute_tool(db, tc.name, tc.args)
                
                parts_list = provider.format_tool_response(tc.name, result_str)
                
                tool_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.TOOL,
                    content=result_str,
                    tool_name=tc.name,
                    parts=parts_list,
                    tokens=await provider.count_tokens(result_str)
                )
                pending_messages.append(tool_msg)
                current_history.append(tool_msg)
        else:
            # Final text turn: Stream the response content
            accumulated_content = []
            
            async def raw_stream():
                async for chunk in provider.generate_response_stream(
                    current_history,
                    summary=conversation.summary,
                    rag_context=rag_context or None
                ):
                    if chunk:
                        yield chunk

            async for chunk in deanonymize_stream(raw_stream(), pii_mapping):
                accumulated_content.append(chunk)
                yield chunk
            
            final_text = "".join(accumulated_content)
            final_model_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.MODEL,
                content=final_text,
                parts=[{"text": final_text}],
                tokens=await provider.count_tokens(final_text) if final_text else 0
            )
            pending_messages.append(final_model_message)
            break
            
    if not final_model_message:
        final_model_message = pending_messages[-1]
        if final_model_message and final_model_message.content:
            final_model_message.content = deanonymize_pii(final_model_message.content, pii_mapping)
            if final_model_message.parts:
                for part in final_model_message.parts:
                    if isinstance(part, dict) and "text" in part and part["text"]:
                        part["text"] = deanonymize_pii(part["text"], pii_mapping)
        
    for msg in pending_messages:
        db.add(msg)
    await db.commit()
