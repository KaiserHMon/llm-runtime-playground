import hashlib
from app.services.embedding.base import EmbeddingProvider

class MockEmbeddingProvider(EmbeddingProvider):
    async def get_embedding(self, text: str) -> list[float]:
        """
        Generates a deterministic mock embedding vector of size 768
        based on the SHA-256 hash of the input text.
        """
        if not text:
            return [0.0] * 768

        # Create a deterministic mock vector using sha256 of the text
        hasher = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hasher.digest()

        # Seed a simple deterministic Linear Congruential Generator (LCG)
        # using the first 8 bytes of the hash to produce a deterministic vector of floats
        seed = int.from_bytes(hash_bytes[:8], byteorder="big")

        vector = []
        state = seed
        for _ in range(768):
            # LCG parameters (Numerical Recipes constants)
            state = (state * 1664525 + 1013904223) % 2**32
            # Normalize to range [-1.0, 1.0]
            val = (state / (2**32 - 1)) * 2.0 - 1.0
            vector.append(val)

        return vector
