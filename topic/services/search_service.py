"""
Search Service Implementation
Provides abstract interface for search providers and YouTube search service
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from .providers.scrapetube_provider import ScrapeTubeProvider, SearchResult

logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    """Request model for search operations with validation"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query string")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return")
    include_metadata: bool = Field(default=False, description="Whether to include metadata in results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional search filters")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v):
        if v < 1:
            raise ValueError('max_results must be at least 1')
        if v > 50:
            raise ValueError('max_results cannot exceed 50')
        return v

class SearchResponse(BaseModel):
    """Response model for search operations with validation"""
    results: List[str] = Field(..., description="List of content URLs")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    search_time_ms: float = Field(..., ge=0, description="Search execution time in milliseconds")
    metadata: Optional[List[SearchResult]] = Field(default=None, description="Optional metadata for results")
    
    @model_validator(mode='after')
    def validate_results_consistency(self):
        # Ensure total_results matches actual results count
        if len(self.results) != self.total_results:
            self.total_results = len(self.results)
        
        # Ensure metadata count matches results count if provided
        if self.metadata is not None and len(self.metadata) != len(self.results):
            raise ValueError('Metadata count must match results count')
        
        return self
    
    @field_validator('results')
    @classmethod
    def validate_results_urls(cls, v):
        for url in v:
            if not url or not isinstance(url, str):
                raise ValueError('All results must be non-empty strings')
            # Basic URL validation for YouTube
            if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']):
                logger.warning(f"Non-YouTube URL found in results: {url}")
        return v

class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    @abstractmethod
    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        """
        Search for content and return URLs
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of content URLs
            
        Raises:
            ValueError: If query is invalid
            Exception: If search fails
        """
        pass
    
    @abstractmethod
    def search_with_metadata(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Search for content and return detailed metadata
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with metadata
            
        Raises:
            ValueError: If query is invalid
            Exception: If search fails
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_query(self, query: str) -> bool:
        """
        Validate search query
        
        Args:
            query: Search query to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information
        
        Returns:
            Dictionary with provider details
        """
        pass

class YouTubeSearchService:
    """YouTube search service wrapper using dependency injection"""
    
    def __init__(self, provider: Optional[SearchProvider] = None):
        """
        Initialize YouTube search service
        
        Args:
            provider: Search provider instance (defaults to ScrapeTubeProvider)
        """
        self.provider = provider or ScrapeTubeProvider()
        
        # Validate provider implements SearchProvider interface
        if not isinstance(self.provider, SearchProvider):
            # Wrap ScrapeTubeProvider to implement SearchProvider interface
            self.provider = self._wrap_scrapetube_provider(self.provider)
        
        logger.info(f"YouTube search service initialized with provider: {self.provider.__class__.__name__}")
    
    def _wrap_scrapetube_provider(self, scrapetube_provider: ScrapeTubeProvider) -> SearchProvider:
        """
        Wrap ScrapeTubeProvider to implement SearchProvider interface
        
        Args:
            scrapetube_provider: ScrapeTubeProvider instance
            
        Returns:
            SearchProvider compatible wrapper
        """
        class ScrapeTubeSearchProvider(SearchProvider):
            def __init__(self, provider: ScrapeTubeProvider):
                self.provider = provider
            
            def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
                return self.provider.search(query, max_results)
            
            def search_with_metadata(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
                return self.provider.search_with_metadata(query, max_results)
            
            def health_check(self) -> bool:
                return self.provider.health_check()
            
            def validate_query(self, query: str) -> bool:
                return self.provider.validate_query(query)
            
            def get_provider_info(self) -> Dict[str, Any]:
                return self.provider.get_provider_info()
        
        return ScrapeTubeSearchProvider(scrapetube_provider)
    
    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Search YouTube for videos
        
        Args:
            request: Search request parameters
            
        Returns:
            SearchResponse with results
            
        Raises:
            ValueError: If request is invalid
            Exception: If search fails
        """
        # Validate request
        if not request.query or not request.query.strip():
            raise ValueError("Search query cannot be empty")
        
        if not self.provider.validate_query(request.query):
            raise ValueError("Invalid search query")
        
        logger.info(f"Searching YouTube for: '{request.query}' (max_results: {request.max_results})")
        
        try:
            import time
            start_time = time.time()
            
            if request.include_metadata:
                # Get results with metadata
                metadata_results = self.provider.search_with_metadata(
                    request.query, 
                    request.max_results
                )
                
                # Extract URLs from metadata results
                urls = [result.url for result in metadata_results]
                
                search_time_ms = (time.time() - start_time) * 1000
                
                return SearchResponse(
                    results=urls,
                    query=request.query,
                    total_results=len(urls),
                    search_time_ms=search_time_ms,
                    metadata=metadata_results
                )
            else:
                # Get URLs only
                urls = self.provider.search(request.query, request.max_results)
                
                search_time_ms = (time.time() - start_time) * 1000
                
                return SearchResponse(
                    results=urls,
                    query=request.query,
                    total_results=len(urls),
                    search_time_ms=search_time_ms
                )
                
        except Exception as e:
            logger.error(f"YouTube search failed for query '{request.query}': {str(e)}")
            raise
    
    def search_simple(self, query: str, max_results: int = 5) -> List[str]:
        """
        Simple search interface that returns URLs
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of YouTube video URLs
            
        Raises:
            ValueError: If query is invalid
            Exception: If search fails
        """
        request = SearchRequest(
            query=query,
            max_results=max_results,
            include_metadata=False
        )
        
        response = self.search(request)
        return response.results
    
    def search_with_metadata(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search with metadata interface
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with metadata
            
        Raises:
            ValueError: If query is invalid
            Exception: If search fails
        """
        request = SearchRequest(
            query=query,
            max_results=max_results,
            include_metadata=True
        )
        
        response = self.search(request)
        return response.metadata or []
    
    def health_check(self) -> bool:
        """
        Check if the search service is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            return self.provider.health_check()
        except Exception as e:
            logger.error(f"Search service health check failed: {str(e)}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider
        
        Returns:
            Dictionary with provider details
        """
        try:
            return self.provider.get_provider_info()
        except Exception as e:
            logger.error(f"Failed to get provider info: {str(e)}")
            return {
                "name": "Unknown",
                "version": "Unknown",
                "description": "Provider information unavailable",
                "error": str(e)
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the search service
        
        Returns:
            Dictionary with service details
        """
        provider_info = self.get_provider_info()
        
        return {
            "service_name": "YouTubeSearchService",
            "service_version": "1.0.0",
            "description": "YouTube search service with configurable providers",
            "provider": provider_info,
            "supported_operations": [
                "search",
                "search_simple",
                "search_with_metadata",
                "health_check"
            ]
        }
    
    def configure_provider(self, provider: SearchProvider) -> None:
        """
        Configure a new search provider
        
        Args:
            provider: New search provider instance
        """
        if not isinstance(provider, SearchProvider):
            raise ValueError("Provider must implement SearchProvider interface")
        
        old_provider = self.provider.__class__.__name__
        self.provider = provider
        
        logger.info(f"Search provider changed from {old_provider} to {provider.__class__.__name__}")
    
    def get_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID if found, None otherwise
        """
        try:
            # Check if provider has this method
            if hasattr(self.provider, 'get_video_id_from_url'):
                return self.provider.get_video_id_from_url(url)
            
            # Fallback implementation
            if "youtube.com/watch?v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting video ID from URL {url}: {str(e)}")
            return None

# Factory function for easy instantiation
def create_youtube_search_service(provider: Optional[SearchProvider] = None) -> YouTubeSearchService:
    """
    Factory function to create YouTube search service
    
    Args:
        provider: Optional search provider (defaults to ScrapeTubeProvider)
        
    Returns:
        YouTubeSearchService instance
    """
    return YouTubeSearchService(provider)

# Default instance for convenience
default_search_service = create_youtube_search_service()