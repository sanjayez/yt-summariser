"""
Abstract interface for vector store providers.
Defines the contract for vector storage and similarity search operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models import (
    VectorDocument, VectorQuery, VectorSearchResponse, 
    IndexConfig, IndexStats, ProcessingJob
)

class VectorStoreProvider(ABC):
    """Abstract interface for vector store providers"""
    
    @abstractmethod
    async def upsert_documents(self, documents: List[VectorDocument]) -> ProcessingJob:
        """
        Upsert documents into the vector store.
        
        Args:
            documents: List of documents to upsert
            
        Returns:
            Processing job for tracking the operation
        """
        pass
    
    @abstractmethod
    async def search_similar(self, query: VectorQuery) -> VectorSearchResponse:
        """
        Search for similar documents.
        
        Args:
            query: Search query with text and parameters
            
        Returns:
            Search response with results and metadata
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def create_index(self, config: IndexConfig) -> bool:
        """
        Create a new vector index.
        
        Args:
            config: Index configuration
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Optional[IndexStats]:
        """
        Get statistics for a vector index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index statistics if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List of index names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass 