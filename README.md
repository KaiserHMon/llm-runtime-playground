# LLM Runtime Playground

A robust, minimal-abstraction Python backend built from scratch to explore the fundamentals of AI Engineering, Context Engineering, and Large Language Model (LLM) integrations.

Instead of relying on heavy abstraction frameworks (like LangChain or LlamaIndex), this repository implements the core components of a conversational AI system from the ground up using clean architecture principles, FastAPI, SQLAlchemy 2.0, and the official `google-genai` SDK.

---

## Quick Path

### 1. Set Up Environment
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY="your-api-key-here"
```

### 2. Start the Development Server
Install dependencies and run the server using `uv`:
```bash
uv run uvicorn main:app --reload
```

### 3. Verify
Open your browser and navigate to the interactive OpenAPI documentation:
* **Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **API Status**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## API Reference

| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| `/conversations` | `POST` | `{"title": "Optional Title"}` | Creates a new empty conversation thread. |
| `/conversations/{id}` | `GET` | *None* | Retrieves a conversation including its full sorted message history. |
| `/conversations/{id}/messages` | `POST` | `{"content": "Your message"}` | Sends a message, calls Gemini, and returns the full response synchronously. |
| `/conversations/{id}/messages/stream` | `POST` | `{"content": "Your message"}` | Streams the Gemini response chunks in real-time via Server-Sent Events (SSE). |

---

## Core Architecture & Rationale

| Layer | Technologies | Rationale & Design Decision |
| :--- | :--- | :--- |
| **API Layer** | FastAPI & Pydantic V2 | Async-first routing with strict contract validations. Keeps input schemas (`MessageCreate`) separate from output schemas (`MessageResponse`) to prevent mass-assignment security vulnerabilities. |
| **Service Layer** | `google-genai` SDK | Handles **Context Engineering**. Implements custom prompt building, token optimization (truncating chat history context window to a maximum limit), and orchestrates LLM API calls. |
| **Data Layer** | SQLAlchemy 2.0 & SQLite | Local SQLite persistence using `aiosqlite` for asynchronous connection handling. Adheres to modern type-safe declarative mapping to support static analyzers like Pyright/Ruff. |

### Transactional Atomicity
To avoid corrupted or mismatched chat histories due to network or provider issues, the orchestration layer in [app/services/chat_service.py](file:///C:/Proyectos/ai-engineering/llm-runtime-playground/app/services/chat_service.py) enforces atomic writes:
1. Fetch history first (read-only query).
2. Execute the heavy network request to the Gemini API outside the database transaction.
3. If the network call completes successfully, open a short transaction to insert both the user message and the model's response atomically.

---

## Testing the API (Examples)

### Create a Conversation
```bash
curl -X POST http://127.0.0.1:8000/conversations \
     -H "Content-Type: application/json" \
     -d '{"title": "AI Engineering Chat"}'
```
Response:
```json
{
  "id": "e6f47700-1122-3344-5566-778899aabbcc",
  "title": "AI Engineering Chat",
  "created_at": "2026-07-20T17:00:00Z",
  "updated_at": "2026-07-20T17:00:00Z"
}
```

### Send Message (Synchronous)
```bash
curl -X POST http://127.0.0.1:8000/conversations/e6f47700-1122-3344-5566-778899aabbcc/messages \
     -H "Content-Type: application/json" \
     -d '{"content": "Explain clean architecture in 1 sentence."}'
```

### Send Message (Streaming SSE)
```bash
curl -X POST http://127.0.0.1:8000/conversations/e6f47700-1122-3344-5566-778899aabbcc/messages/stream \
     -H "Content-Type: application/json" \
     -d '{"content": "Write a short poem about coding."}'
```

---

## Future Enhancements
* [ ] **Streaming UI**: Build a minimal frontend to consume the SSE endpoint.
* [x] **Dynamic Token Budget**: Count and track tokens per message using `google-genai` capabilities to actively prune history before hitting context limits.
* [x] **Native Tool Calling**: Register python helper functions as native tools/functions in the Gemini API config.
* [ ] **Semantic Memory**: Add a vector storage layer for retrieval-augmented generation (RAG).
