"""
Content Validators for Query Processor
"""

import re
import bleach
from django.core.exceptions import ValidationError
from video_processor.validators import validate_youtube_url


def validate_request_content(content: str, request_type: str) -> dict:
    if request_type in ['video', 'playlist']:
        return _validate_video_content(content, request_type)
    elif request_type == 'topic':
        return _validate_topic_content(content)
    else:
        raise ValidationError(f"Invalid request type: {request_type}")


def _validate_video_content(content: str, request_type: str) -> dict:
    if not content or not content.strip():
        raise ValidationError("URL cannot be empty")
    
    content = content.strip()
    
    try:
        # Use existing YouTube URL validation from video_processor
        validate_youtube_url(content)
    except (ValidationError, ValueError) as e:
        raise ValidationError(f"Invalid YouTube {request_type} URL: {e}") from e
    
    # Validate specific URL type
    if request_type == 'video' and not _is_video_url(content):
        raise ValidationError("URL must be a YouTube video URL, not a playlist or channel")
    elif request_type == 'playlist' and not _is_playlist_url(content):
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
    video_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&(?!list=)[\w=&-]*)*$',
        r'https?://(?:www\.)?youtu\.be/[\w-]+(?:\?(?!list=)[\w=&-]*)*$',
    ]
    
    return any(re.match(pattern, url.strip()) for pattern in video_patterns)


def _is_playlist_url(url: str) -> bool:
    playlist_patterns = [
        r'https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/watch\?.*list=[\w-]+',
    ]
    
    return any(re.search(pattern, url.strip()) for pattern in playlist_patterns)


def _contains_suspicious_patterns(query: str) -> bool:
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
        r'(?i)\b(union|select)\s+(select|from|where)\b',      # SQL injection
        r'(?i);\s*(drop|delete|truncate)\s+table\b',          # Destructive SQL
        r'(?is)<script[^>]*>[\s\S]*?</script>',               # Multi-line <script>
        r'(?i)\bjavascript\s*:\s*\S+',                        # JS protocol
    ]

    return any(re.search(p, query) for p in malicious_patterns)