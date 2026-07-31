# ruff: noqa: E402, F401, F541
import os
import asyncio
import json
import logging
import sys

# 1. Set environment variables to run isolated
os.environ["QDRANT_PATH"] = ":memory:"

# 2. Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_evals")

# 3. Setup temporary database path and override app session maker/engine before importing models/services
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core import database as db_module
from app.core.database import Base

eval_db_url = "sqlite+aiosqlite:///./eval_runtime.db"
engine = create_async_engine(eval_db_url, connect_args={"check_same_thread": False})
db_module.engine = engine
db_module.AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# Import models and services now that database module is overridden
from app.models.chat import Conversation
from app.services.rag_service import ingest_document, search_chunks, init_qdrant
from app.services.chat_service import process_chat_message
from app.core.config import settings
from tests.evals.evaluator import evaluate_response

async def call_with_retry(async_func, *args, max_retries=5, base_delay=5, **kwargs):
    """
    Executes an async function with exponential backoff on 429 Rate Limit/Resource Exhausted errors.
    """
    for attempt in range(max_retries):
        try:
            return await async_func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit (429) hit during {async_func.__name__}. "
                    f"Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                raise e
    # Final attempt
    return await async_func(*args, **kwargs)

async def seed_data(db: AsyncSession, embedding_provider: str | None) -> None:
    logger.info("Seeding test documents...")
    doc1_content = (
        "Project Antigravity is a secret project aiming to develop gravity-defying systems using quantum stabilizers. "
        "The core component is the Antigravity Quantum Stabilizer (AQS) which requires a sustained power input of 500MW. "
        "The project is led by Dr. Sarah Chen, and the laboratory is located in Area 51, Sector G. "
        "All blueprints are encrypted with the key AQS-SECURE-2026."
    )
    doc2_content = (
        "FastAPI applications should be deployed using Uvicorn or Gunicorn with Uvicorn workers. "
        "The recommended production command is: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4`. "
        "To enable autostart, use a systemd service configured at `/etc/systemd/system/fastapi.service`. "
        "SSL termination should be handled by an Nginx reverse proxy running on port 443."
    )
    
    await ingest_document(
        db,
        name="Antigravity Project Specifications",
        content=doc1_content,
        conversation_id=None,
        embedding_provider=embedding_provider
    )
    await ingest_document(
        db,
        name="FastAPI deployment details",
        content=doc2_content,
        conversation_id=None,
        embedding_provider=embedding_provider
    )
    logger.info("Successfully seeded 2 test documents.")

async def main():
    # Determine providers based on key presence
    has_api_key = bool(settings.GEMINI_API_KEY)
    provider_name = "gemini" if has_api_key else "mock"
    embedding_provider = None if has_api_key else "mock"
    
    logger.info(f"Running evals with LLM Provider: {provider_name}, Embedding Provider: {embedding_provider or 'default'}")
    
    # Re-initialize the database
    logger.info("Initializing evaluation database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
    await init_qdrant()
    
    # Load goldens
    goldens_path = os.path.join(os.path.dirname(__file__), "goldens.json")
    with open(goldens_path, "r", encoding="utf-8") as f:
        goldens = json.load(f)
        
    results = []
    
    # Process queries using a database session
    async with db_module.AsyncSessionLocal() as db:
        await seed_data(db, embedding_provider)
        
        for case in goldens:
            case_id = case["id"]
            query = case["query"]
            category = case["category"]
            expected_keywords = case.get("expected_keywords", [])
            
            logger.info(f"[{case_id}] Processing {category} query: '{query}'")
            
            # Create a new conversation for isolation
            conversation = Conversation(title=f"Eval Conversation {case_id}")
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            # Execute chat query with retry wrapper
            model_msg = await call_with_retry(
                process_chat_message,
                db,
                conversation_id=conversation.id,
                content=query,
                provider_name=provider_name
            )
            response_content = model_msg.content or ""
            
            # Retrieve context manually for RAG
            context = ""
            if category == "RAG":
                chunks = await search_chunks(
                    db,
                    query=query,
                    conversation_id=conversation.id,
                    top_k=5,
                    embedding_provider=embedding_provider
                )
                if chunks:
                    formatted_chunks = []
                    for idx, chunk in enumerate(chunks, 1):
                        source_name = chunk.document.name if chunk.document else "Unknown"
                        formatted_chunks.append(f"[{idx}] (Source: {source_name}): {chunk.content}")
                    context = "\n\n".join(formatted_chunks)
            
            # Call the LLM-as-a-judge evaluator with retry wrapper
            logger.info(f"[{case_id}] Evaluating response...")
            eval_res = await call_with_retry(
                evaluate_response,
                query,
                response_content,
                context
            )
            
            # Check expected keywords in response
            matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_content.lower()]
            
            case_result = {
                "id": case_id,
                "category": category,
                "query": query,
                "response": response_content,
                "context": context,
                "context_len": len(context),
                "faithfulness_score": eval_res.faithfulness_score,
                "faithfulness_reason": eval_res.faithfulness_reason,
                "relevance_score": eval_res.relevance_score,
                "relevance_reason": eval_res.relevance_reason,
                "expected_keywords": expected_keywords,
                "matched_keywords": matched_keywords
            }
            results.append(case_result)
            
            # Real-time console output
            print(f"\n==================================================")
            print(f"Test Case: {case_id} ({category})")
            print(f"Query: {query}")
            print(f"Response: {response_content}")
            print(f"Context Length: {len(context)} characters")
            print(f"Faithfulness Score: {eval_res.faithfulness_score}/5 - Reason: {eval_res.faithfulness_reason}")
            print(f"Relevance Score: {eval_res.relevance_score}/5 - Reason: {eval_res.relevance_reason}")
            print(f"Keywords Matched: {len(matched_keywords)}/{len(expected_keywords)} ({matched_keywords})")
            print(f"==================================================\n")
            
            # Sleep 2 seconds between cases to proactively prevent rate limit issues
            await asyncio.sleep(2)
            
    # Calculate global metrics
    total_faith = sum(r["faithfulness_score"] for r in results)
    total_rel = sum(r["relevance_score"] for r in results)
    avg_faith = total_faith / len(results) if results else 0
    avg_rel = total_rel / len(results) if results else 0
    
    # Generate report.md content
    summary_rows = []
    for r in results:
        summary_rows.append(
            f"| {r['id']} | {r['category']} | {r['query']} | {r['context_len']} | {r['faithfulness_score']}/5 | {r['relevance_score']}/5 |"
        )
    summary_table_rows = "\n".join(summary_rows)
    
    detailed_sections = []
    for r in results:
        ctx_display = r['context'] if r['context'] else "*No context retrieved.*"
        kw_display = f"{len(r['matched_keywords'])}/{len(r['expected_keywords'])} matched: " + ", ".join([f"`{kw}`" for kw in r['matched_keywords']])
        detailed_sections.append(
            f"### Test Case: {r['id']} ({r['category']})\n\n"
            f"- **Query**: {r['query']}\n"
            f"- **Retrieved Context (Length: {r['context_len']} chars)**:\n"
            f"  ```\n"
            f"  {ctx_display}\n"
            f"  ```\n"
            f"- **Generated Response**:\n"
            f"  ```\n"
            f"  {r['response']}\n"
            f"  ```\n"
            f"- **Expected Keywords**: {', '.join([f'`{kw}`' for kw in r['expected_keywords']])}\n"
            f"- **Keywords Match**: {kw_display}\n"
            f"- **Faithfulness Score**: {r['faithfulness_score']}/5\n"
            f"  - **Reason**: {r['faithfulness_reason']}\n"
            f"- **Relevance Score**: {r['relevance_score']}/5\n"
            f"  - **Reason**: {r['relevance_reason']}\n"
        )
    detailed_reports = "\n---\n\n".join(detailed_sections)
    
    report_content = f"""# AI Evaluation Suite Report

This report summarizes the evaluation of the LLM Runtime Playground using LLM-as-a-judge (model: `gemini-flash-lite-latest`).

## Global Metrics

| Metric | Average Score |
| :--- | :--- |
| **Faithfulness** | {avg_faith:.2f} / 5.0 |
| **Relevance** | {avg_rel:.2f} / 5.0 |

## Test Case Results

| ID | Category | Query | Retrieved Context Length | Faithfulness | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- |
{summary_table_rows}

---

## Detailed Test Case Evaluations

{detailed_reports}
"""

    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Markdown report generated at {report_path}")
    
    # 6. Clean up the database
    logger.info("Cleaning up database engine and database file...")
    await engine.dispose()
    
    for filename in ["eval_runtime.db", "eval_runtime.db-journal", "eval_runtime.db-shm", "eval_runtime.db-wal"]:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Removed temporary file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary database file {filename}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
