"""
Abstract interface for embedding providers.
Defines the contract for text embedding operations.
"""

from abc import ABC, abstractmethod

from ..models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers"""

    @abstractmethod
    async def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Embed a single text.

        Args:
            request: Embedding request with text and model

        Returns:
            Embedding response with vector and metadata
        """

    @abstractmethod
    async def embed_batch(
        self, request: BatchEmbeddingRequest
    ) -> BatchEmbeddingResponse:
        """
        Embed multiple texts in batch.

        Args:
            request: Batch embedding request with list of texts

        Returns:
            Batch embedding response with list of vectors
        """

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported embedding models.

        Returns:
            List of supported model names
        """

    @abstractmethod
    def get_model_dimensions(self, model: str) -> int:
        """
        Get the dimension of a specific model.

        Args:
            model: Model name

        Returns:
            Vector dimension for the model
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the embedding provider is healthy.

        Returns:
            True if healthy, False otherwise
        """
