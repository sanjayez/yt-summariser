"""
Abstract interface for vector store providers.
Defines the contract for vector storage and similarity search operations.
"""

from abc import ABC, abstractmethod

from ..models import (
    IndexConfig,
    IndexStats,
    ProcessingJob,
    VectorDocument,
    VectorQuery,
    VectorSearchResponse,
)


class VectorStoreProvider(ABC):
    """Abstract interface for vector store providers"""

    @abstractmethod
    async def upsert_documents(self, documents: list[VectorDocument]) -> ProcessingJob:
        """
        Upsert documents into the vector store.

        Args:
            documents: List of documents to upsert

        Returns:
            Processing job for tracking the operation
        """

    @abstractmethod
    async def search_similar(self, query: VectorQuery) -> VectorSearchResponse:
        """
        Search for similar documents.

        Args:
            query: Search query with text and parameters

        Returns:
            Search response with results and metadata
        """

    @abstractmethod
    async def delete_documents(self, document_ids: list[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def get_document(self, document_id: str) -> VectorDocument | None:
        """
        Get a specific document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document if found, None otherwise
        """

    @abstractmethod
    async def create_index(self, config: IndexConfig) -> bool:
        """
        Create a new vector index.

        Args:
            config: Index configuration

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def get_index_stats(self, index_name: str) -> IndexStats | None:
        """
        Get statistics for a vector index.

        Args:
            index_name: Name of the index

        Returns:
            Index statistics if found, None otherwise
        """

    @abstractmethod
    async def list_indices(self) -> list[str]:
        """
        List all available indices.

        Returns:
            List of index names
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
