"""
ScrapeTube Provider Implementation
Handles YouTube search functionality using the scrapetube library
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import scrapetube
import re

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
    
    def __init__(self, 
                 max_results: int = 5, 
                 timeout: int = 30,
                 filter_shorts: bool = True,
                 english_only: bool = True,
                 min_duration_seconds: int = 60,
                 max_duration_seconds: Optional[int] = None):
        """
        Initialize the ScrapeTube provider
        
        Args:
            max_results: Maximum number of results to return (default: 5)
            timeout: Request timeout in seconds (default: 30)
            filter_shorts: Whether to filter out YouTube shorts (default: True)
            english_only: Whether to filter for English content only (default: True)
            min_duration_seconds: Minimum video duration in seconds to exclude shorts (default: 60)
            max_duration_seconds: Maximum video duration in seconds to cap long videos (default: None for no limit)
        """
        self.max_results = max_results
        self.timeout = timeout
        self.filter_shorts = filter_shorts
        self.english_only = english_only
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.base_url = "https://www.youtube.com/watch?v="
        
        # Common non-English indicators (basic detection)
        self.non_english_patterns = [
            # Chinese characters
            r'[\u4e00-\u9fff]',
            # Japanese Hiragana and Katakana
            r'[\u3040-\u309f\u30a0-\u30ff]',
            # Korean
            r'[\uac00-\ud7af]',
            # Arabic
            r'[\u0600-\u06ff]',
            # Russian/Cyrillic
            r'[\u0400-\u04ff]',
            # Hindi/Devanagari
            r'[\u0900-\u097f]',
            # Thai
            r'[\u0e00-\u0e7f]',
        ]
    
    def _parse_duration_to_seconds(self, duration_str: str) -> Optional[int]:
        """
        Parse YouTube duration string to seconds
        
        Args:
            duration_str: Duration string like "1:23" or "12:34" or "1:23:45"
            
        Returns:
            Duration in seconds or None if parsing fails
        """
        if not duration_str:
            return None
            
        try:
            parts = duration_str.split(':')
            if len(parts) == 1:  # Just seconds
                return int(parts[0])
            elif len(parts) == 2:  # Minutes:seconds
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # Hours:minutes:seconds
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    def _is_likely_english(self, text: str) -> bool:
        """
        Check if text is likely English using basic heuristics
        
        Args:
            text: Text to analyze
            
        Returns:
            True if likely English, False otherwise
        """
        if not text:
            return True  # Default to True for empty text
        
        # Check for non-English character patterns
        for pattern in self.non_english_patterns:
            if re.search(pattern, text):
                return False
        
        # Check if majority of characters are ASCII (basic English check)
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        total_chars = len(text)
        
        if total_chars == 0:
            return True
        
        ascii_ratio = ascii_chars / total_chars
        return ascii_ratio > 0.8  # 80% ASCII characters
    
    def _should_include_video(self, video_data: dict, title: str = None, duration: str = None) -> bool:
        """
        Check if video should be included based on filters
        
        Args:
            video_data: Raw video data from scrapetube
            title: Video title
            duration: Video duration string
            
        Returns:
            True if video should be included, False otherwise
        """
        # Filter out shorts by duration
        if self.filter_shorts and duration:
            duration_seconds = self._parse_duration_to_seconds(duration)
            if duration_seconds and duration_seconds < self.min_duration_seconds:
                logger.debug(f"Filtering out short video: {duration} ({duration_seconds}s)")
                return False
        
        # Filter out long videos by max duration
        if self.max_duration_seconds and duration:
            duration_seconds = self._parse_duration_to_seconds(duration)
            if duration_seconds and duration_seconds > self.max_duration_seconds:
                logger.debug(f"Filtering out long video: {duration} ({duration_seconds}s > {self.max_duration_seconds}s)")
                return False
        
        # Filter for English content
        if self.english_only and title:
            if not self._is_likely_english(title):
                logger.debug(f"Filtering out non-English video: {title}")
                return False
        
        return True
        
    def search(self, query: str, max_results: Optional[int] = None) -> List[str]:
        """
        Search YouTube for videos and return video URLs
        
        Args:
            query: Search query string
            max_results: Override default max results
            
        Returns:
            List of YouTube video URLs (filtered according to settings)
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search fails or times out
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        query = query.strip()
        results_limit = max_results or self.max_results
        
        logger.info(f"Searching YouTube for: '{query}' (target: {results_limit} filtered results)")
        
        try:
            start_time = time.time()
            video_urls = []
            total_fetched = 0
            batch_size = 20  # Fetch in batches
            max_total_fetch = 100  # Don't fetch more than 100 total to avoid infinite loops
            
            # Use scrapetube to search YouTube - it returns a generator
            videos = scrapetube.get_search(
                query=query,
                limit=max_total_fetch,  # Set high limit, we'll break when we have enough
                sleep=1,  # Add delay between requests to be respectful
                sort_by='relevance',
                results_type='video'  # Only videos, not channels/playlists
            )
            
            # Process the results iteratively
            for video in videos:
                try:
                    video_id = video.get('videoId')
                    if not video_id:
                        continue
                    
                    total_fetched += 1
                    
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
                    
                    # Apply filters
                    if not self._should_include_video(video, title, duration):
                        logger.debug(f"Filtered out: {title} ({duration})")
                        continue
                    
                    url = f"{self.base_url}{video_id}"
                    video_urls.append(url)
                    
                    # Log some basic info for debugging
                    logger.debug(f"Accepted video {len(video_urls)}/{results_limit}: {title} ({duration}) - {url}")
                    
                    # Stop if we have enough results
                    if len(video_urls) >= results_limit:
                        logger.info(f"Found target {results_limit} filtered results after checking {total_fetched} videos")
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing video result: {str(e)}")
                    continue
                
                # Safety check - don't fetch indefinitely
                if total_fetched >= max_total_fetch:
                    logger.warning(f"Reached maximum fetch limit ({max_total_fetch}) - stopping search")
                    break
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s, found {len(video_urls)} filtered results from {total_fetched} total videos")
            
            if not video_urls:
                logger.warning(f"No videos found for query: '{query}' after filtering {total_fetched} videos")
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
            List of SearchResult objects with metadata (filtered according to settings)
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search fails or times out
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        query = query.strip()
        results_limit = max_results or self.max_results
        
        logger.info(f"Searching YouTube with metadata for: '{query}' (target: {results_limit} filtered results)")
        
        try:
            start_time = time.time()
            search_results = []
            total_fetched = 0
            max_total_fetch = 100  # Don't fetch more than 100 total to avoid infinite loops
            
            # Use scrapetube to search YouTube - it returns a generator
            videos = scrapetube.get_search(
                query=query,
                limit=max_total_fetch,  # Set high limit, we'll break when we have enough
                sleep=1,  # Add delay between requests to be respectful
                sort_by='relevance',
                results_type='video'  # Only videos, not channels/playlists
            )
            
            # Process the results with metadata iteratively
            for video in videos:
                try:
                    video_id = video.get('videoId')
                    if not video_id:
                        continue
                    
                    total_fetched += 1
                    
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
                    
                    # Apply filters
                    if not self._should_include_video(video, title, duration):
                        logger.debug(f"Filtered out: {title} ({duration})")
                        continue
                    
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
                    logger.debug(f"Accepted video {len(search_results)}/{results_limit}: {title} ({duration}) - {result.url}")
                    
                    # Stop if we have enough results
                    if len(search_results) >= results_limit:
                        logger.info(f"Found target {results_limit} filtered results after checking {total_fetched} videos")
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing video result with metadata: {str(e)}")
                    continue
                
                # Safety check - don't fetch indefinitely
                if total_fetched >= max_total_fetch:
                    logger.warning(f"Reached maximum fetch limit ({max_total_fetch}) - stopping search")
                    break
            
            search_time = time.time() - start_time
            logger.info(f"Search with metadata completed in {search_time:.2f}s, found {len(search_results)} filtered results from {total_fetched} total videos")
            
            if not search_results:
                logger.warning(f"No videos found for query: '{query}' after filtering {total_fetched} videos")
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
            "filter_shorts": self.filter_shorts,
            "english_only": self.english_only,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "supported_search_types": self.get_supported_search_types()
        }