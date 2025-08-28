"""
Content Validators for Query Processor
Handles validation for different request types (video/playlist/topic)
"""

import re
import bleach
from django.core.exceptions import ValidationError
from video_processor.validators import validate_youtube_url


def validate_request_content(content: str, request_type: str) -> dict:
    """
    Validate and prepare content based on request type.
    
    Args:
        content: User input (URL or search query)
        request_type: Type of request ('video', 'playlist', 'topic')
        
    Returns:
        dict: Validation result with processed content and appropriate fields
        
    Raises:
        ValidationError: If content is invalid for the request type
    """
    
    if request_type in ['video', 'playlist']:
        return _validate_video_content(content, request_type)
    elif request_type == 'topic':
        return _validate_topic_content(content)
    else:
        raise ValidationError(f"Invalid request type: {request_type}")


def _validate_video_content(content: str, request_type: str) -> dict:
    """
    Validate video/playlist URL content.
    
    Args:
        content: YouTube URL
        request_type: 'video' or 'playlist'
        
    Returns:
        dict: Validated content for video/playlist requests
              - total_videos: 1 for videos, 0 for playlists (unknown until processed)
        
    Raises:
        ValidationError: If URL is invalid
    """
    if not content or not content.strip():
        raise ValidationError("URL cannot be empty")
    
    content = content.strip()
    
    try:
        # Use existing YouTube URL validation from video_processor
        validate_youtube_url(content)
    except Exception as e:
        raise ValidationError(f"Invalid YouTube {request_type} URL: {str(e)}")
    
    # Validate specific URL type
    if request_type == 'video':
        if not _is_video_url(content):
            raise ValidationError("URL must be a YouTube video URL, not a playlist or channel")
    elif request_type == 'playlist':
        if not _is_playlist_url(content):
            raise ValidationError("URL must be a YouTube playlist URL, not a video or channel")
    
    result = {
        'original_content': content,
        'video_urls': [content],  # Single item array for consistency
        'concepts': [],  # Empty for video/playlist requests
        'enhanced_queries': [],  # Empty for video/playlist requests
        'intent_type': None,  # Not applicable for video/playlist requests
        'total_videos': 1 if request_type == 'video' else 0  # 0 for playlists (unknown until processed)
    }
    return result


def _validate_topic_content(content: str) -> dict:
    """
    Validate topic search query content.
    
    Args:
        content: Search query string
        
    Returns:
        dict: Validated content for topic requests
        
    Raises:
        ValidationError: If query is invalid
    """
    if not content or not content.strip():
        raise ValidationError("Search query cannot be empty")
    
    content = content.strip()
    
    # Length validation
    if len(content) < 4:
        raise ValidationError("Search query must be at least 4 characters long")
    
    if len(content) > 500:
        raise ValidationError("Search query cannot exceed 500 characters")
    
    # Content validation - sanitize and check for suspicious patterns
    if _contains_suspicious_patterns(content):
        raise ValidationError("Search query contains potentially unsafe content")
    
    # Clean the content using bleach for additional safety
    content = bleach.clean(content, tags=[], attributes={}, strip=True).strip()
    
    return {
        'original_content': content,
        'video_urls': [],  # Will be populated after query processing
        'concepts': [],  # Will be populated by LLM
        'enhanced_queries': [],  # Will be populated by LLM
        'intent_type': None,  # Will be populated by LLM
        'total_videos': 0  # Will be updated after processing
    }


def _is_video_url(url: str) -> bool:
    """
    Check if URL is a YouTube video URL (not playlist).
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if it's a video URL
    """
    video_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&(?!list=)[\w=&-]*)*$',
        r'https?://(?:www\.)?youtu\.be/[\w-]+(?:\?(?!list=)[\w=&-]*)*$',
    ]
    
    return any(re.match(pattern, url.strip()) for pattern in video_patterns)


def _is_playlist_url(url: str) -> bool:
    """
    Check if URL is a YouTube playlist URL.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if it's a playlist URL
    """
    playlist_patterns = [
        r'https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/watch\?.*list=[\w-]+',
    ]
    
    return any(re.search(pattern, url.strip()) for pattern in playlist_patterns)


def _contains_suspicious_patterns(query: str) -> bool:
    """
    Check for suspicious patterns in search queries using a simplified approach.
    
    Args:
        query: Search query to check
        
    Returns:
        bool: True if query contains suspicious patterns
    """
    # Sanitize the query and compare with original
    # If bleach removes content, it was potentially malicious
    sanitized = bleach.clean(query, tags=[], attributes={}, strip=True)
    
    # Check if sanitization removed significant content (potential HTML/script injection)
    if len(sanitized) < len(query) * 0.8:  # More than 20% removed
        return True
    
    # Check for excessive special characters (but more lenient)
    if len(query) > 0:
        special_char_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace() and c not in "'-.,!?:;()") / len(query)
        if special_char_ratio > 0.5:  # More than 50% special characters
            return True
    
    # Check for obviously malicious patterns (very targeted)
    malicious_patterns = [
        r'(?i)\b(union|select)\s+(select|from|where)\b',  # Clear SQL injection
        r'(?i);\s*(drop|delete|truncate)\s+table\b',      # Destructive SQL
        r'<script[^>]*>.*?</script>',                      # Script tags
        r'javascript:\s*\w+',                             # JavaScript protocol
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, query):
            return True
    
    return False


def is_valid_youtube_url(url: str) -> bool:
    """
    Check if URL is a valid YouTube URL (video or playlist).
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid YouTube URL
    """
    if not url or not url.strip():
        return False
    
    try:
        validate_youtube_url(url.strip())
        return True
    except (ValueError, ValidationError):
        return False


def get_url_type(url: str) -> str:
    """
    Determine the type of YouTube URL.
    
    Args:
        url: YouTube URL to analyze
        
    Returns:
        str: 'video', 'playlist', or 'unknown'
    """
    if not is_valid_youtube_url(url):
        return 'unknown'
    
    if _is_playlist_url(url):
        return 'playlist'
    elif _is_video_url(url):
        return 'video'
    else:
        return 'unknown'


def validate_and_classify_content(content: str) -> dict:
    """
    Automatically classify and validate content.
    
    Args:
        content: User input to classify and validate
        
    Returns:
        dict: Classification result with request_type and validation data
        
    Raises:
        ValidationError: If content cannot be classified or is invalid
    """
    if not content or not content.strip():
        raise ValidationError("Content cannot be empty")
    
    content = content.strip()
    
    # Try to classify as URL first
    if is_valid_youtube_url(content):
        url_type = get_url_type(content)
        if url_type in ['video', 'playlist']:
            validation_result = validate_request_content(content, url_type)
            return {
                'request_type': url_type,
                **validation_result
            }
    
    # If not a valid URL, treat as topic search
    validation_result = validate_request_content(content, 'topic')
    return {
        'request_type': 'topic',
        **validation_result
    }
