from typing import Dict
from app.services.embedding.base import EmbeddingProvider
from app.services.embedding.gemini import GeminiEmbeddingProvider
from app.services.embedding.mock import MockEmbeddingProvider

class EmbeddingFactory:
    def __init__(self):
        self._providers: Dict[str, EmbeddingProvider] = {}

    def register_provider(self, name: str, provider: EmbeddingProvider):
        """Registers an embedding provider under a name."""
        self._providers[name.lower()] = provider

    def get_provider(self, name: str | None = None) -> EmbeddingProvider:
        """
        Retrieves the requested Embedding provider.
        Defaults to 'gemini' if no provider name is specified.
        """
        provider_name = (name or "gemini").lower()
        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Embedding Provider '{provider_name}' is not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )
        return provider

# Global embedding provider factory instance
embedding_factory = EmbeddingFactory()

# Register our current providers
embedding_factory.register_provider("gemini", GeminiEmbeddingProvider())
embedding_factory.register_provider("mock", MockEmbeddingProvider())
