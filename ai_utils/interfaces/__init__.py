"""
Abstract interfaces for AI utilities.
These interfaces define contracts for different AI operations.
"""

from .embeddings import EmbeddingProvider
from .vector_store import VectorStoreProvider
from .llm import LLMProvider

__all__ = [
    "EmbeddingProvider",
    "VectorStoreProvider", 
    "LLMProvider"
] 