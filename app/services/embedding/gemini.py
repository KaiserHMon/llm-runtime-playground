from google import genai
from app.core.config import settings
from app.services.embedding.base import EmbeddingProvider

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured. Cannot call Gemini Embedding API.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def get_embedding(self, text: str) -> list[float]:
        """
        Asynchronously retrieves the embedding of the input text using Gemini's embedding model.
        Defaults to gemini-embedding-2 and falls back to gemini-embedding-001 if needed.
        """
        if not text:
            return []

        try:
            response = await self.client.aio.models.embed_content(
                model="gemini-embedding-2",
                contents=text
            )
            if response.embeddings and response.embeddings[0].values is not None:
                return response.embeddings[0].values[:768]
        except Exception:
            try:
                # Fall back to gemini-embedding-001 if gemini-embedding-2 is not available
                response = await self.client.aio.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text
                )
                if response.embeddings and response.embeddings[0].values is not None:
                    return response.embeddings[0].values[:768]
            except Exception:
                pass

        raise ValueError("Failed to retrieve embedding values from Gemini API response.")
