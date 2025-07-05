"""
Abstract interface for LLM providers.
Defines the contract for RAG and text generation operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models import RAGQuery, RAGResponse, VectorSearchResult

class LLMProvider(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    async def generate_rag_response(
        self, 
        query: RAGQuery, 
        context_documents: List[VectorSearchResult]
    ) -> RAGResponse:
        """
        Generate a RAG response using context documents.
        
        Args:
            query: RAG query with parameters
            context_documents: Relevant documents for context
            
        Returns:
            RAG response with generated answer and sources
        """
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported LLM models.
        
        Returns:
            List of supported model names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass 