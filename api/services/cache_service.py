"""
Cache Service for API layer caching operations.
Handles embedding caching and other API-specific caching needs.
"""
from typing import Optional, List
from django.core.cache import cache
from telemetry import get_logger


class CacheService:
    """
    Handles caching operations for the API layer.
    
    This service provides centralized caching for:
    - Embedding vectors for questions
    - Frequently accessed responses
    - Service instances (when appropriate)
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_cached_embedding(self, question: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for a question.
        
        Args:
            question: The question text to get cached embedding for
            
        Returns:
            List of floats representing the embedding, or None if not cached
        """
        cache_key = self._generate_embedding_cache_key(question)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            self.logger.debug(f"Cache hit for embedding: {question[:50]}...")
        else:
            self.logger.debug(f"Cache miss for embedding: {question[:50]}...")
            
        return cached_result
    
    def cache_embedding(self, question: str, embedding: List[float], ttl: int = 3600) -> None:
        """
        Cache an embedding for future use.
        
        Args:
            question: The question text
            embedding: The embedding vector to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        cache_key = self._generate_embedding_cache_key(question)
        cache.set(cache_key, embedding, ttl)
        
        self.logger.debug(f"Cached embedding for: {question[:50]}... (TTL: {ttl}s)")
    
    def get_cached_response(self, cache_key: str) -> Optional[dict]:
        """
        Get a cached response by key.
        
        Args:
            cache_key: The cache key to retrieve
            
        Returns:
            Cached response dict or None if not found
        """
        return cache.get(cache_key)
    
    def cache_response(self, cache_key: str, response: dict, ttl: int = 1800) -> None:
        """
        Cache a response for future use.
        
        Args:
            cache_key: The cache key to store under
            response: The response data to cache
            ttl: Time to live in seconds (default: 30 minutes)
        """
        cache.set(cache_key, response, ttl)
        self.logger.debug(f"Cached response with key: {cache_key} (TTL: {ttl}s)")
    
    def invalidate_cache(self, pattern: str = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Pattern to match for cache invalidation (if supported by backend)
        """
        if pattern:
            # Note: Pattern-based invalidation depends on cache backend
            # For Redis, we could use SCAN, but for default cache, we clear all
            self.logger.warning("Pattern-based cache invalidation not implemented for current backend")
        
        cache.clear()
        self.logger.info("Cache cleared")
    
    def _generate_embedding_cache_key(self, question: str) -> str:
        """
        Generate a consistent cache key for embeddings.
        
        Args:
            question: The question text
            
        Returns:
            Cache key string
        """
        # Use hash to create consistent key from normalized question
        normalized_question = question.lower().strip()
        question_hash = hash(normalized_question)
        return f"embedding_{question_hash}"
    
    def generate_response_cache_key(self, question: str, video_id: str) -> str:
        """
        Generate a cache key for search responses.
        
        Args:
            question: The question text
            video_id: The video ID
            
        Returns:
            Cache key string
        """
        question_hash = hash(question.lower().strip())
        return f"search_response_{video_id}_{question_hash}"
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics if available.
        
        Returns:
            Dictionary with cache statistics
        """
        # This is basic - would need Redis or Memcached for detailed stats
        return {
            "backend": cache.__class__.__name__,
            "status": "active"
        }