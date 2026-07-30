from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """
        Calcula y devuelve el vector de embeddings para el texto provisto.
        Debe retornar una lista de números flotantes (normalmente de dimensión 768 para Gemini).
        """
        pass
