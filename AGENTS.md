# LLM Runtime Playground

## Core Rules
- DO NOT use LangChain, LlamaIndex, or any AI abstraction frameworks.
- Learn by implementing fundamental components.
- Clean architecture without over-engineering.
- ALWAYS write documentation, code comments, identifiers, and text artifacts in English.
- Socratic learning: Before implementing any new feature, explain the architecture/concepts or ask questions to help the user reason about the design so they learn the fundamentals.

## Phase 1: MVP
### Functional Requirements
1. Basic chat endpoint (FastAPI).
2. Manual static context construction (Fixed System Prompt + last N messages).
3. Direct connection with Gemini SDK.
4. Conversation persistence (SQLAlchemy + SQLite).

### Architecture (3 Layers)
1. **API Layer (Presentation):** FastAPI routers and Pydantic validation.
2. **Service Layer (Business Logic):** Context Engineering logic (prompt building) and LLM calls.
3. **Data Layer (Repositories):** SQLAlchemy models and SQLite queries.

### Data Model
1. **conversations:**
   - `id`: UUID (PK)
   - `title`: String
   - `created_at`: DateTime
   - `updated_at`: DateTime
2. **messages:**
   - `id`: UUID (PK)
   - `conversation_id`: UUID (FK)
   - `role`: String ('user', 'model', 'system')
   - `content`: Text
   - `tokens`: Integer (Nullable)
   - `created_at`: DateTime

## Future Phases
- Tool Calling (Native via SDK)
- Memory (History summaries with LLM)
- Dynamic Token Budget

## Current Progress
- **Completed:** Project setup (`uv`), Data Layer (SQLAlchemy 2.0 modern syntax), Schemas (Pydantic V2), Core Config, LLM Service (`llm.py`), Orchestration Service (`chat_service.py`), and FastAPI API Endpoints (with streaming support).
- **Next Step:** Upload the repository to GitHub.
