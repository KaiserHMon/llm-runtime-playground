from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM runtime playground"
    GEMINI_API_KEY: str

    # Pydantic V2 config to read from .env file
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

# Instantiating the settings. 
# If GEMINI_API_KEY is missing in the environment, the app will crash here (Fail Fast).
settings = Settings()
