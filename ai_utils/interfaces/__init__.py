"""
Abstract interfaces for AI utilities.
These interfaces define contracts for different AI operations.
"""

from .embeddings import EmbeddingProvider
from .llm import LLMProvider
from .vector_store import VectorStoreProvider

__all__ = ["EmbeddingProvider", "VectorStoreProvider", "LLMProvider"]
