from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM runtime playground"
    GEMINI_API_KEY: str | None = None
    
    QDRANT_PATH: str = "qdrant_storage"
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # Orchestration & Token Budget Settings
    TOKEN_BUDGET: int = 4000
    TOKEN_BUFFER_PER_MESSAGE: int = 20
    MAX_TOOL_LOOP_ITERATIONS: int = 5

    # Pydantic V2 config to read from .env file
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

# Instantiating the settings. 
# If GEMINI_API_KEY is missing in the environment, the app will crash here (Fail Fast).
settings = Settings()  # type: ignore
