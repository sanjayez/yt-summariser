"""
AI Utils Package
Provides modular AI utilities for embeddings, vector storage, and language models
"""

__version__ = "0.1.0"

from .config import get_config, AIConfig
from .providers import OpenAIEmbeddingProvider, PineconeVectorStoreProvider, OpenAILLMProvider
from .services import EmbeddingService, VectorService, LLMService
from .models import (
    VectorDocument, VectorQuery, VectorSearchResult, VectorSearchResponse,
    EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse,
    RAGQuery, RAGResponse, ProcessingJob, ProcessingStatus,
    ChatMessage, ChatRequest, ChatResponse, ChatRole,
    TextGenerationRequest, TextGenerationResponse
)

__all__ = [
    # Configuration
    "get_config", "AIConfig",
    
    # Providers
    "OpenAIEmbeddingProvider", "PineconeVectorStoreProvider", "OpenAILLMProvider",
    
    # Services
    "EmbeddingService", "VectorService", "LLMService",
    
    # Models
    "VectorDocument", "VectorQuery", "VectorSearchResult", "VectorSearchResponse",
    "EmbeddingRequest", "EmbeddingResponse", "BatchEmbeddingRequest", "BatchEmbeddingResponse",
    "RAGQuery", "RAGResponse", "ProcessingJob", "ProcessingStatus",
    "ChatMessage", "ChatRequest", "ChatResponse", "ChatRole",
    "TextGenerationRequest", "TextGenerationResponse"
] 