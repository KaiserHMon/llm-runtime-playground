# LLM Runtime Playground

A robust, minimal-abstraction Python backend built from scratch to explore the fundamentals of AI Engineering, Context Engineering, and Large Language Model (LLM) integrations.

## Purpose
This project is an educational and architectural playground. Instead of relying on heavy abstraction frameworks like LangChain or LlamaIndex, this repository builds the core components of a conversational AI system from the ground up using clean architecture principles. 

The goal is to deeply understand token budgets, context window management, dynamic prompt construction, and native tool calling using official provider SDKs.

## Architecture
Built on a standard 3-layer Clean Architecture approach:

1. **API Layer (`app/api` & `app/schemas`)**
   - **FastAPI** for high-performance async routing.
   - **Pydantic V2** for strict input/output validation and contract definition.
   - Segregation between `Create` (Input) and `Response` (Output) schemas.

2. **Service Layer (`app/services`)**
   - The core brain of the application.
   - Implements **Context Engineering**: retrieving history, assembling the prompt array, managing system prompts, and calling the Gemini SDK natively.

3. **Data Layer (`app/models` & `app/core`)**
   - **SQLAlchemy 2.0** utilizing modern `Mapped` and `mapped_column` syntax for 100% type safety with static checkers.
   - **SQLite** database (for MVP simplicity) managing conversational state and relationships.

## Technology Stack
* **Language**: Python 3.12+
* **Dependency Management**: `uv`
* **API Framework**: FastAPI & Uvicorn
* **Database / ORM**: SQLAlchemy 2.0 & SQLite
* **Configuration**: Pydantic Settings (Fail-Fast pattern)
* **Type Checking & Linting**: Pyright & Ruff

## Project Structure
```text
/
├── app/
│   ├── api/         # FastAPI endpoints and routers
│   ├── core/        # DB engine, Config, and dependencies injection
│   ├── models/      # SQLAlchemy data models (Conversations, Messages)
│   ├── schemas/     # Pydantic validation contracts
│   └── services/    # Business logic (Context Builder, LLM integration)
├── .env             # Environment variables (Ignored in Git)
├── AGENTS.md        # Project specification and phase tracking
├── discoveries.md   # Architectural decisions and technical lessons
└── pyproject.toml   # Project metadata and uv dependencies
```

## Core Concepts Explored
* **Context Construction**: Building the array of `[System, History, New Message]`.
* **State Persistence**: Using bidirectional SQLAlchemy relationships to store chat history with UTC-aware timestamps.
* **Type Safety**: Enforcing `MessageRole` Enums to guarantee valid AI interaction states (`user`, `model`, `system`).

## Getting Started
*(Instructions pending API layer implementation)*
