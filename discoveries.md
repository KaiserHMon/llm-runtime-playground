# Technical Discoveries & Best Practices

This document tracks key architectural decisions and framework-specific nuances discovered during development. These notes are specifically formatted to be converted into AI agent skills/instructions later.

## SQLAlchemy 2.0 vs Static Type Checkers (Pyright)
* **Context**: Using SQLAlchemy's classical declarative mapping (`Column(String)`) causes type-inference failures in strict static type checkers like `pyright`.
* **Discovery**: Always use the modern **Annotated Declarative Mapping** pattern.
* **Pattern**: 
  ```python
  from sqlalchemy.orm import Mapped, mapped_column
  from sqlalchemy import String

  # Correct
  name: Mapped[str] = mapped_column(String, nullable=False)
  ```
* **Why**: It separates the Python runtime type (`Mapped[str]`) from the SQL schema definition (`mapped_column(...)`), making the code 100% type-safe.

## Pydantic V2 Strict Typing and Optionals
* **Context**: Pydantic V2 is significantly stricter than V1.
* **Discovery**: Declaring a type without `| None` but assigning a default of `None` will result in a runtime validation crash.
* **Pattern**:
  ```python
  # Incorrect (Pydantic V2 will error)
  title: str = Field(default=None)
  
  # Correct (Modern Python 3.10+)
  title: str | None = Field(default=None)
  ```
* **Why**: The type hint defines the allowed types. `str` strictly forbids nulls. You must explicitly allow `None` in the type hint union.

## Pydantic V2 ORM Integration
* **Context**: Returning SQLAlchemy objects directly from FastAPI endpoints requires Pydantic to read object attributes instead of dictionary keys.
* **Discovery**: `orm_mode = True` is deprecated.
* **Pattern**:
  ```python
  from pydantic import BaseModel, ConfigDict

  class MySchema(BaseModel):
      model_config = ConfigDict(from_attributes=True)
  ```

## Model Separation (Create vs Response)
* **Context**: Using a single schema for both API input and output is a security risk (Mass Assignment vulnerabilities).
* **Discovery**: Always segregate schemas. E.g., `MessageCreate` (only `content`) vs `MessageResponse` (includes `id`, `role`, `created_at`).

## Environment Configuration
* **Context**: Relying on `os.getenv()` without validation leads to silent failures.
* **Discovery**: Use `pydantic-settings` to enforce a **Fail-Fast** startup if critical secrets (like API keys) are missing.

## LLM Temperature Guide
- **0.0 (Deterministic):** No creativity. Use for JSON parsing, strict data extraction, deterministic code, or RAG systems.
- **0.2 - 0.4 (Conservative):** Fluid but fact-based. Use for technical translations, summaries, and strict reporting.
- **0.7 - 0.8 (Conversational):** Default for chatbots, virtual tutors, and email drafting. Balances human-like tone and factuality.
- **1.0+ (Creative):** Maximum freedom. Use for fiction, brainstorming, or marketing copy.

## LLM Chat Memory & Context Management
* **Context**: Managing chat history in production LLM backends without exploding latency or costs.
* **Discovery**: Storing message token counts in the database avoids redundant token-counting network calls. Pruning context dynamically (e.g. 4000 tokens) with a fixed safety buffer (e.g. 20 tokens per message) effectively handles API role delimiters and templates.
* **Pattern (Hybrid Memory Model)**:
  - **Sliding Window**: Keeps recent chat turns (e.g. 3000 tokens) to maintain conversational flow and formatting.
  - **Summarization**: Uses background tasks to condense older messages falling out of the window into a persistent summary.
  - **RAG (Semantic Memory)**: Stores messages in a Vector DB and retrieves relevant old turns on-demand.
  - **Prompt Layout**: `[System Instruction] + [RAG Semantic Context] + [Summarized History] + [Sliding Window]`

## Pydantic V2 and Binary Serialization in LLM Tool Calling (bytes)
* **Context**: Modern Gemini models (like Gemini 2.0/2.5) return thought process data (`thought_signature`) as binary `bytes` fields.
* **Discovery**: Using standard python `json.dumps` to serialize conversation history results in a `TypeError: Object of type bytes is not JSON serializable` crash.
* **Pattern**: Always use Pydantic's `model.model_dump(mode="json")` (introduced in Pydantic V2) instead of standard `model_dump()`.
* **Why**: The `json` mode automatically base64-encodes all binary (`bytes`) data to strings, allowing them to be safely saved in database JSON columns (e.g. SQLite JSON or PostgreSQL jsonb) and later deserialized seamlessly back to python `bytes` during validation with `types.Part.model_validate()`.

## Decoupling Dynamic Backend Parameters in LLM Tool Registration
* **Context**: When using Gemini's automatic tool generation via python callables (e.g. passing tools directly to the SDK config), Pydantic attempts to build parameter schemas for all arguments. If a function expects a backend object (like `db: AsyncSession`), it will crash because database session types are not JSON-serializable schema types.
* **Discovery**: Filter out dynamic parameters by manually constructing `types.Tool` declarations and mapping python arguments to schemas, while excluding internal parameters.
* **Pattern**:
  ```python
  import inspect
  from google.genai import types

  def get_tools_for_gemini(registry) -> list[types.Tool]:
      declarations = []
      for name, func in registry._tools.items():
          sig = inspect.signature(func)
          properties = {}
          for param_name, param in sig.parameters.items():
              if param_name == "db": # Exclude dynamic DB parameter
                  continue
              # Map type annotations ...
              properties[param_name] = types.Schema(type=types.Type.STRING)
          
          declarations.append(
              types.FunctionDeclaration(
                  name=name,
                  description=func.__doc__.strip().split("\n")[0] if func.__doc__ else "",
                  parameters=types.Schema(type=types.Type.OBJECT, properties=properties)
              )
          )
      return [types.Tool(function_declarations=declarations)]
  ```
* **Why**: The LLM only sees the parameters it needs to supply (like `location`), while the backend retains the ability to dynamically inject database sessions during the execution loop.

## Incremental Chat Summarization Memory (Pointer Pattern)
* **Context**: When designing chat summarization memory, we need to decide how to track which messages have already been compressed into the summary to avoid double-processing.
* **Discovery**: Using an individual state/flag (e.g., `is_summarized` boolean) on each message requires updating $N$ message rows in the database upon every eviction. Instead, storing a pointer (`last_summarized_message_id`) in the parent `Conversation` table is much more efficient.
* **Pattern**:
  - The `Conversation` model stores a `last_summarized_message_id` (string) referencing the last message included in the cumulative `summary`.
  - When history exceeds the token budget, we fetch messages between the current pointer and the new eviction boundary.
  - We merge their content with the existing `summary` using the LLM, and update `conversation.summary` and `conversation.last_summarized_message_id` in a single database update.
* **Why**: It reduces database I/O from $N$ message row updates to a single conversation row update per eviction event, while maintaining strict chronological consistency.

## RAG Text Chunking: Recursive Character Splitter with Overlap
* **Context**: Splitting documents into smaller, semantically coherent segments (chunks) for vector database storage.
* **Discovery**: Fixed-size splitting or simple line-based splitting shears sentences and destroys context. Standard best practice for high-precision retrieval is using a **Recursive Character Text Splitter** configured with a custom **Chunk Overlap**.
* **Pattern**:
  - **Chunk Size**: Target length of each fragment (e.g. 500-1000 characters).
  - **Chunk Overlap**: Sequence of repeating characters between adjacent chunks (e.g. 100-200 characters) to ensure no context is lost at the boundaries.
  - **Recursive Split Delimiters**: Split hierarchically by checking paragraphs (`\n\n`), then lines (`\n`), then words (` `), and finally individual characters.
* **Why**: It ensures document chunks maintain logical readability and syntactic cohesion, avoiding half-broken sentences in the vector search index, while overlap guarantees that transitions between blocks are not lost.

## Multi-Provider LLM Abstraction (Factory Pattern)
* **Context**: Coupling the core conversation services directly to a specific LLM vendor SDK (such as `google-genai`) makes it difficult to run unit tests without internet/API keys, migrate to other model APIs, or run cheap local simulation scenarios.
* **Discovery**: Define a strict abstract provider interface (`LLMProvider`) that encapsulates all operations involving LLM interaction (response generation, streaming, token counting, cumulative summarization, and vendor-specific message part formatting). Then, resolve the active provider at runtime via a centralized Registry/Factory (`LLMFactory`).
* **Pattern**:
  - Define the base class and response schema in a shared contract module:
    ```python
    class LLMResponse(BaseModel):
        content: str | None = None
        tool_calls: list[ToolCall] = []
        parts: list[dict] | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

    class LLMProvider(ABC):
        @abstractmethod
        async def generate_response(self, history: list[Message], summary: str | None, rag_context: str | None) -> LLMResponse: ...
    ```
  - Implement vendor concrete providers (e.g. `GeminiProvider`, `MockProvider`) and register them to a global factory instance.
  - In `chat_service.py`, fetch the provider via the factory on each request payload field:
    ```python
    provider = factory.get_provider(provider_name)
    response = await provider.generate_response(current_history, ...)
    ```
* **Why**: This enforces Clean Architecture (dependency inversion), makes core business logic 100% unit-testable using a non-networked mock provider, and allows swapping or using multiple models concurrently based on payload preferences.

## Pytest-Asyncio Testing with Database Isolation
* **Context**: Running integration tests on asynchronous FastAPI services with database side effects can pollute the database state between tests, making them flaky and non-deterministic.
* **Discovery**: Use a unified `tests/conftest.py` file to declare database and client fixtures. By creating a temporary database (`test_runtime.db`), and using an async fixture with session scope (`setup_test_db`), the tables can be dropped and recreated automatically once per session. Individual test cases can use a fresh `db_session` fixture to manage atomic transaction states, and an `api_client` using `httpx.ASGITransport` to run FastAPI completely in-memory.
* **Pattern**:
  - Define fixtures in `conftest.py`:
    ```python
    @pytest_asyncio.fixture(scope="session", autouse=True)
    async def setup_test_db():
        test_engine = create_async_engine(test_db_url)
        db_module.engine = test_engine
        # ... create all tables
        yield
        # ... teardown tables
        
    @pytest_asyncio.fixture
    async def db_session():
        async with db_module.AsyncSessionLocal() as session:
            yield session
    ```
  - Inject them into test cases:
    ```python
    @pytest.mark.asyncio
    async def test_feature(db_session):
        # test code ...
    ```
* **Why**: It ensures strict state isolation, prevents flaky tests, avoids setting up real network sockets (speeding up runtimes), and makes testing completely local.

## Pytest path configuration (ModuleNotFoundError)
* **Context**: Running `pytest` from the root of a Python project often throws `ModuleNotFoundError: No module named 'app'` because the execution path does not automatically include the current directory in `sys.path`.
* **Discovery**: Configure the python path directly in the `pyproject.toml` configuration under `[tool.pytest.ini_options]`.
* **Pattern**:
  ```toml
  [tool.pytest.ini_options]
  asyncio_mode = "auto"
  pythonpath = ["."]
  ```
* **Why**: This standardizes running the test suite via a simple `uv run pytest` or `pytest` without needing manual `sys.path` hacks (like `sys.path.append(...)`) inside the test files.

## Qdrant Semantic Retrieval & Score Thresholds
* **Context**: When building RAG applications, querying vector databases with a fixed `top_k` (e.g. `top_k=5`) without a minimum similarity score threshold can lead to severe context pollution, especially in small-scale test suites or development databases.
* **Discovery**: Vector search queries (like Qdrant's `query_points`) will *always* return up to `top_k` matches sorted by cosine similarity, even if their actual similarity score is near 0. For real embeddings (like Gemini's `text-embedding-004`), relevant documents average `~0.85` similarity, while irrelevant ones average `~0.50`. Adding a `score_threshold = 0.70` cleanly filters out irrelevant context.
* **Heuristic/Mock Caveat**: Mock embedding providers that generate pseudo-random vectors (e.g., using deterministic text hashes) will yield orthogonal vectors where the cosine similarity between any two different documents is `~0.0`. Thus, applying a high `score_threshold` for mock providers will result in zero matches and break local/offline test suites.
* **Pattern**:
  - Dynamically bypass or adjust the threshold based on the provider:
    ```python
    # Default score threshold for non-mock providers (e.g. Gemini) to filter out irrelevant context
    if score_threshold is None and embedding_provider != "mock":
        score_threshold = 0.70
        
    search_result = await qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_emb,
        query_filter=filter_condition,
        limit=top_k,
        score_threshold=score_threshold
    )
    ```
* **Why**: This prevents unrelated document chunks from polluting the LLM's system prompt (mitigating model distractions and token overhead), while preserving the offline testability of mock provider configurations.

## Streaming-aware PII Masking & Hybrid Safety Guardrails
* **Context**: When deploying safety filters in high-performance conversational backends, we face two critical challenges:
  1. Preserving user privacy (PII masking) during real-time Server-Sent Events (SSE) streaming without breaking placeholders split across chunk borders.
  2. Defending against adversarial prompt injections (jailbreaks) without introducing massive API latency and cost overhead to every message.
* **Discovery**: 
  - For PII: Replace sensitive data with tags (e.g. `[EMAIL_1]`) prior to executing RAG/LLM pipelines. For streaming, use a **bracket-buffering character generator** that yields characters normally but buffers them as soon as `[` is detected, only mapping/releasing once the matching `]` is found or the buffer exceeds maximum placeholder length.
  - For Jailbreaks: Enforce a **two-phase hybrid guardrail**. Run a fast, local Regex check on inputs using compiled high-risk override patterns. Trigger secondary LLM-based safety checks with structured outputs (`SafetyDecision` schema at temperature `0.0`) only when the regex matches.
* **Pattern**:
  - The streaming generator:
    ```python
    async def deanonymize_stream(stream, mapping: dict[str, str]):
        buffer = ""
        in_brackets = False
        async for chunk in stream:
            for char in chunk:
                if char == '[':
                    if in_brackets:
                        yield buffer
                        buffer = ""
                    in_brackets = True
                    buffer += char
                elif char == ']':
                    if in_brackets:
                        buffer += char
                        yield mapping.get(buffer, buffer)
                        buffer = ""
                        in_brackets = False
                    else:
                        yield char
                else:
                    if in_brackets:
                        buffer += char
                        if len(buffer) > 40:
                            yield buffer
                            buffer = ""
                            in_brackets = False
                    else:
                        yield char
    ```
* **Why**: This guarantees 100% PII protection (sensitive data never touches the LLM or DB), ensures clean real-time streaming output, and keeps latency at absolute zero for 99% of normal queries, only incurring API costs when a prompt pattern is genuinely suspicious.

## Semantic Routing via Gemini Structured Outputs
* **Context**: In dynamic RAG workflows, routing queries to RAG vs CHAT dynamically is necessary to prevent irrelevant vector lookups and prompt noise.
* **Discovery**: Gemini's support for structured JSON outputs (with `response_schema` and `response_mime_type="application/json"`) can be utilized with Pydantic schemas (e.g., `RoutingDecision` containing `route` and `justification`) at `temperature=0.0` for deterministic and highly accurate routing decisions.
* **Pattern**:
  - Formulate a strict schema (e.g., `RoutingDecision` Pydantic model) describing routing rules for RAG (specific knowledge, codes, passphrases, project-specific data) vs CHAT (conversational, general programming, broad concepts).
  - Format the last 10 messages of conversation history to maintain conversational context.
  - Call the model passing `RoutingDecision` as the schema and parsing the JSON response. Fall back to standard CHAT routing if errors occur.
* **Why**: Eliminates manual regex/heuristic routing hacks and ensures consistent routing decisions by letting the model self-classify its query requirements deterministically, avoiding unnecessary embedding generations and vector queries for 99% of regular conversations.

## React SPA Real-time SSE & Tool Timeline Parsing
* **Context**: Displaying dynamic tool calls, intermediate LLM responses, and real-time generation chunks during multi-turn orchestration loops.
* **Discovery**: Traditional REST endpoints make it hard to show streaming progress alongside structured database updates (e.g., intermediate tool calls, RAG sources). Combining SSE (Server-Sent Events) for the final response generation stream with a post-stream reload of the conversation details solves this.
* **Pattern**:
  - **SSE Streaming**: POST JSON containing message parameters to the streaming endpoint `/conversations/{id}/messages/stream`. Read the response body stream using a reader (`ReadableStreamDefaultReader`) and decode it with `TextDecoder` to update a `streamedContent` state.
  - **Post-Stream Sync**: Once streaming is complete, perform a separate fetch to the GET `/conversations/{id}` endpoint. This reloads the database state, replacing optimistic client messages with real backend entries containing full metadata (e.g., actual database IDs, tokens consumed, tool execution responses, and RAG sources).
  - **Turn Timeline Grouping**: Write a client-side utility (`getGroupedTurns`) that aggregates chronological messages by grouping intermediate turns (`model` messages requesting tools and matching `tool` response messages) under their initiating `user` message, rendering an execution timeline alongside the final answers.
* **Why**: Keeps the UI highly responsive by rendering text chunks as they arrive, and seamlessly visualizes the internal tool loops, routing decisions, and RAG grounding scores once the generation turn completes.

## Automated LLM-as-a-Judge Evaluation (Faithfulness & Relevance)
* **Context**: Programmatically measuring response quality in production or CI/CD pipelines without human-in-the-loop dependencies or subjective/flaky evaluations.
* **Discovery**: Using a lightweight LLM (e.g., `gemini-flash-lite-latest`) configured with a strict system prompt grading rubric, structured JSON output matching a Pydantic model (`EvaluationResponse`), and `temperature=0.0` provides a highly cost-effective, reproducible, and deterministic rating suite for *Faithfulness* (grounding) and *Relevance*.
* **Pattern**:
  - Define the evaluation schema:
    ```python
    class EvaluationResponse(BaseModel):
        faithfulness_score: int = Field(description="Faithfulness score from 1 to 5")
        faithfulness_reason: str = Field(description="Detailed explanation for the faithfulness score")
        relevance_score: int = Field(description="Relevance score from 1 to 5")
        relevance_reason: str = Field(description="Detailed explanation for the relevance score")
    ```
  - Formulate a strict system prompt containing structured grading criteria for both metrics.
  - Request structured output:
    ```python
    response = await client.aio.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=f"User Query: {query}\n\nGenerated Response: {response_content}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=EvaluationResponse,
            temperature=0.0
        )
    )
    ```
* **Why**: Enforcing structured schemas on the judge LLM eliminates parsing/regex headaches, guarantees well-formed JSON evaluations, and providing strict rubric criteria inside the system instruction grounds the judge's scoring behavior, reducing variance.

## Asynchronous Background Title Generation and Session Isolation
* **Context**: When a conversation is started, generating a user-friendly title based on the first message asynchronously. In a streaming SSE endpoint, we must trigger this without stalling the response stream.
* **Discovery**:
  1. **FastAPI BackgroundTasks & Database Session Lifecycle**: Since `BackgroundTasks` run after the HTTP response is sent and the request's connection is closed, passing the request-scoped database session (`db`) directly can lead to `Cannot operate on a closed session` errors. We must use a fallback database strategy: try using the passed session, but if it is closed/expired, create a fresh database connection (`AsyncSessionLocal()`) within the task to commit the title safely.
  2. **Preventing State Pollution on Singletons**: LLM provider classes resolved via `LLMFactory` are singletons. Modifying attributes like `system_prompt` temporarily inside async tasks causes race conditions with concurrent requests. Always instantiate a clean, local instance of `GeminiProvider` inside the task to encapsulate task-specific system instructions.
* **Pattern**:
  - The API endpoint:
    ```python
    @router.post("/{conversation_id}/messages/stream")
    async def send_message_stream(
        conversation_id: str,
        payload: MessageCreate,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
    ):
        return StreamingResponse(
            stream_chat_message(..., background_tasks=background_tasks),
            media_type="text/event-stream"
        )
    ```
  - The streaming service:
    ```python
    async def stream_chat_message(..., background_tasks=None):
        # Persist user message first
        db.add(user_message)
        await db.commit()
        
        # Check message count
        count = await db.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
        )
        if count == 1 and background_tasks:
            background_tasks.add_task(generate_conversation_title, conversation_id, content, db)
    ```
  - The background task:
    ```python
    async def generate_conversation_title(conversation_id, content, db):
        provider = GeminiProvider(model_id="gemini-flash-lite-latest")
        provider.system_prompt = "You are a conversation titler..."
        response = await provider.generate_response([Message(role="user", content=content)], temperature=0.0)
        title = response.content.strip().strip('"\'`').replace(".", "")
        try:
            # Try request-scoped DB
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            conv = await db.scalar(stmt)
            if conv:
                conv.title = title
                await db.commit()
        except Exception:
            # Fallback to new DB session
            async with AsyncSessionLocal() as new_db:
                stmt = select(Conversation).where(Conversation.id == conversation_id)
                conv = await new_db.scalar(stmt)
                if conv:
                    conv.title = title
                    await new_db.commit()
    ```
* **Why**: Ensures zero latency impact on the user's chat streaming experience, eliminates state pollution and race conditions across requests, and guarantees database write safety even after HTTP connections are terminated.