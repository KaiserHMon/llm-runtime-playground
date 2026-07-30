# Refactoring and Code Quality Roadmap

This roadmap outlines best practices and refactoring opportunities to improve robustness, testability, and configuration management for the LLM Runtime Playground.

## Priority Tasks

- [x] **Decouple Test Suite from live LLM APIs**: Mock Gemini's embedding models in RAG tests to make the test suite fully offline-capable.
- [x] **Make LLM API Keys Optional at Startup**: Set up fallback dummy keys during local/mock tests to prevent initialization crashes when API keys are not configured.
- [x] **Add Explicit Message Ordering in Database**: Ensure the `Conversation.messages` relationship uses an explicit `order_by` clause to prevent scrambled histories.
- [x] **Extract Constants to Core Configuration**: Move all hardcoded variables (like `TOKEN_BUDGET` and `MAX_TOOL_LOOP_ITERATIONS`) to [config.py](file:///C:/Proyectos/ai-engineering/llm-runtime-playground/app/core/config.py).
- [x] **Improve Token Count Fallback Logic**: Adapt token estimation fallback to inspect message `parts` when `content` is null, preventing massive underestimations for tool turns.

---

## Architectural Backlog

| Topic | Description | Impact |
|---|---|---|
| **API Key Check** | `settings.py` fails fast on missing `GEMINI_API_KEY`. | Prevents booting without credentials, but blocks test runners utilizing `MockProvider`. |
| **Token Fallback** | Fallback uses `len(msg.content) // 4`. | Ignores tool calls/responses payloads (which reside in `parts`), risking context overflows. |
| **Message Ordering** | `selectinload(Conversation.messages)` relies on default DB order. | Can lead to scrambled message histories on database engines that do not guarantee write order. |
| **Constants Isolation** | Hardcoded loop limit (5) and budget (4000) in `chat_service.py`. | Restricts flexibility of configuring parameters on a per-model basis. |
| **API Test isolation** | RAG tests make real calls to Gemini `embed_content`. | Tests are slow, require network, and consume user token quotas. |

## Verification Checklist

Reviewers can verify implementation of these items:
- [x] Pytest passes completely without active internet connection (after test mocks are introduced).
- [x] No hardcoded model/token settings are left in `chat_service.py`.
- [x] Message history loads with consistent chronological sorting in SQLite.
