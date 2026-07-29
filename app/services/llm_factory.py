from typing import Dict
from app.services.llm_base import LLMProvider
from app.services.llm_gemini import GeminiProvider
from app.services.llm_mock import MockProvider

class LLMFactory:
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        
    def register_provider(self, name: str, provider: LLMProvider):
        """Registers a provider class under a name."""
        self._providers[name.lower()] = provider
        
    def get_provider(self, name: str | None = None) -> LLMProvider:
        """
        Retrieves the requested LLM provider.
        Defaults to 'gemini' if no provider name is specified.
        """
        provider_name = (name or "gemini").lower()
        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"LLM Provider '{provider_name}' is not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )
        return provider

# Global LLM provider factory instance
factory = LLMFactory()

# Register our current providers
factory.register_provider("gemini", GeminiProvider())
factory.register_provider("mock", MockProvider())
