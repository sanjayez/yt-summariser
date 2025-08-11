"""
Shared AI service registry

Provides per-process singletons for LLM services to avoid cold starts and
unnecessary client re-instantiation across tasks.
"""

from typing import Optional

from ..config import get_config
from ..providers.gemini_llm import GeminiLLMProvider
from ..providers.weaviate_store import WeaviateVectorStoreProvider
from .llm_service import LLMService
from .vector_service import VectorService

_gemini_llm_service: Optional[LLMService] = None
_vector_service: Optional[VectorService] = None


def get_gemini_llm_service() -> LLMService:
    """Return a shared LLMService backed by GeminiLLMProvider (singleton)."""
    global _gemini_llm_service
    if _gemini_llm_service is None:
        config = get_config()
        provider = GeminiLLMProvider(config=config)
        _gemini_llm_service = LLMService(provider=provider)
    return _gemini_llm_service


async def warmup_gemini_llm() -> bool:
    """Warm up Gemini by performing a minimal health check."""
    service = get_gemini_llm_service()
    try:
        result = await service.health_check()
        return result.get("status") == "healthy"
    except Exception:
        return False


def get_vector_service() -> VectorService:
    """Return a shared VectorService backed by Weaviate provider (singleton)."""
    global _vector_service
    if _vector_service is None:
        config = get_config()
        provider = WeaviateVectorStoreProvider(config=config)
        _vector_service = VectorService(provider=provider)
    return _vector_service


async def warmup_vector_store() -> bool:
    """Warm up vector store: health check and a tiny search to prime native vectorizer."""
    try:
        service = get_vector_service()
        health = await service.health_check()
        # Perform a minimal search to load vectorizer/collection paths
        from ..models import VectorQuery
        try:
            await service.search_similar(VectorQuery(query="hello", embedding=None, top_k=1, filters=None))
        except Exception:
            # Search may fail if empty collection; that's fine for warmup
            pass
        return health.get("status") == "healthy"
    except Exception:
        return False

