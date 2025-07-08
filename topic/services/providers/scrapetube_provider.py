"""
ScrapeTube Provider Implementation
Handles YouTube search functionality using the scrapetube library
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import scrapetube

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search result data"""
    url: str
    title: str
    video_id: str
    duration: Optional[str] = None
    view_count: Optional[str] = None
    upload_date: Optional[str] = None
    channel_name: Optional[str] = None

class ScrapeTubeProvider:
    """YouTube search provider using scrapetube library"""
    
    def __init__(self, max_results: int = 5, timeout: int = 30):
        """
        Initialize the ScrapeTube provider
        
        Args:
            max_results: Maximum number of results to return (default: 5)
            timeout: Request timeout in seconds (default: 30)
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://www.youtube.com/watch?v="
        
    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        """
        Search YouTube for videos and return video URLs
        
        Args:
            query: Search query string
            max_results: Override default max results
            
        Returns:
            List of YouTube video URLs (exactly 5 by default)
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search fails or times out
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        query = query.strip()
        results_limit = max_results or self.max_results
        
        logger.info(f"Searching YouTube for: '{query}' (limit: {results_limit})")
        
        try:
            start_time = time.time()
            
            # Use scrapetube to search YouTube
            videos = scrapetube.get_search(
                query=query,
                limit=results_limit,
                sleep=1  # Add delay between requests to be respectful
            )
            
            video_urls = []
            
            # Process the results
            for video in videos:
                try:
                    video_id = video.get('videoId')
                    if video_id:
                        url = f"{self.base_url}{video_id}"
                        video_urls.append(url)
                        
                        # Log some basic info for debugging
                        title = video.get('title', {}).get('runs', [{}])[0].get('text', 'Unknown')
                        logger.debug(f"Found video: {title} - {url}")
                        
                        # Stop if we have enough results
                        if len(video_urls) >= results_limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing video result: {str(e)}")
                    continue
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s, found {len(video_urls)} results")
            
            if not video_urls:
                logger.warning(f"No videos found for query: '{query}'")
                return []
            
            return video_urls
            
        except Exception as e:
            logger.error(f"YouTube search failed for query '{query}': {str(e)}")
            raise Exception(f"YouTube search failed: {str(e)}")
    
    def search_with_metadata(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Search YouTube for videos and return detailed metadata
        
        Args:
            query: Search query string
            max_results: Override default max results
            
        Returns:
            List of SearchResult objects with metadata
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search fails or times out
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        query = query.strip()
        results_limit = max_results or self.max_results
        
        logger.info(f"Searching YouTube with metadata for: '{query}' (limit: {results_limit})")
        
        try:
            start_time = time.time()
            
            # Use scrapetube to search YouTube
            videos = scrapetube.get_search(
                query=query,
                limit=results_limit,
                sleep=1  # Add delay between requests to be respectful
            )
            
            search_results = []
            
            # Process the results with metadata
            for video in videos:
                try:
                    video_id = video.get('videoId')
                    if not video_id:
                        continue
                    
                    # Extract title
                    title = "Unknown"
                    title_runs = video.get('title', {}).get('runs', [])
                    if title_runs:
                        title = title_runs[0].get('text', 'Unknown')
                    
                    # Extract duration
                    duration = None
                    length_text = video.get('lengthText', {})
                    if length_text:
                        duration = length_text.get('simpleText')
                    
                    # Extract view count
                    view_count = None
                    view_count_text = video.get('viewCountText', {})
                    if view_count_text:
                        view_count = view_count_text.get('simpleText')
                    
                    # Extract channel name
                    channel_name = None
                    long_byline = video.get('longBylineText', {})
                    if long_byline:
                        runs = long_byline.get('runs', [])
                        if runs:
                            channel_name = runs[0].get('text')
                    
                    # Extract upload date
                    upload_date = None
                    published_time = video.get('publishedTimeText', {})
                    if published_time:
                        upload_date = published_time.get('simpleText')
                    
                    # Create search result
                    result = SearchResult(
                        url=f"{self.base_url}{video_id}",
                        title=title,
                        video_id=video_id,
                        duration=duration,
                        view_count=view_count,
                        upload_date=upload_date,
                        channel_name=channel_name
                    )
                    
                    search_results.append(result)
                    logger.debug(f"Found video with metadata: {title} - {result.url}")
                    
                    # Stop if we have enough results
                    if len(search_results) >= results_limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing video result with metadata: {str(e)}")
                    continue
            
            search_time = time.time() - start_time
            logger.info(f"Search with metadata completed in {search_time:.2f}s, found {len(search_results)} results")
            
            if not search_results:
                logger.warning(f"No videos found for query: '{query}'")
                return []
            
            return search_results
            
        except Exception as e:
            logger.error(f"YouTube search with metadata failed for query '{query}': {str(e)}")
            raise Exception(f"YouTube search with metadata failed: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if the provider is healthy by performing a simple search
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test search
            results = self.search("test", max_results=1)
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID if found, None otherwise
        """
        try:
            if "youtube.com/watch?v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting video ID from URL {url}: {str(e)}")
            return None
    
    def validate_query(self, query: str) -> bool:
        """
        Validate search query
        
        Args:
            query: Search query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        
        # Check minimum length
        if len(query) < 2:
            return False
        
        # Check maximum length (YouTube search limit)
        if len(query) > 100:
            return False
        
        return True
    
    def get_supported_search_types(self) -> List[str]:
        """
        Get list of supported search types
        
        Returns:
            List of supported search types
        """
        return [
            "general",
            "video",
            "channel",
            "playlist"
        ]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information
        
        Returns:
            Dictionary with provider details
        """
        return {
            "name": "ScrapeTubeProvider",
            "version": "1.0.0",
            "description": "YouTube search provider using scrapetube library",
            "max_results": self.max_results,
            "timeout": self.timeout,
            "supported_search_types": self.get_supported_search_types()
        }