"""
Content Validators for API Layer
Centralized validation logic for all request types
"""

import re

import bleach
from django.core.exceptions import ValidationError

# YouTube URL validation regex
YOUTUBE_URL_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
    r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
)


# TODO
# Need to modify this helper function
# we can pass type and check if type and is_video_url or is_playlist_url
# Simplifying this logic
def validate_youtube_url(url):
    """Validate YouTube URL format for both videos and playlists"""
    if not url:
        raise ValidationError("URL cannot be empty")

    # Check if it's a valid YouTube domain
    if not ("youtube.com" in url or "youtu.be" in url):
        raise ValidationError("Must be a valid YouTube URL")

    # Check if it's a valid YouTube URL pattern (video or playlist)
    video_patterns = [
        r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]{11}",  # Video URLs
        r"https?://(?:www\.)?youtu\.be/[\w-]{11}",  # Short video URLs
    ]

    playlist_patterns = [
        r"https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+",  # Playlist URLs
        r"https?://(?:www\.)?youtube\.com/watch\?.*list=[\w-]+",  # Video with playlist
    ]

    is_valid = any(re.search(pattern, url) for pattern in video_patterns) or any(
        re.search(pattern, url) for pattern in playlist_patterns
    )

    if not is_valid:
        raise ValidationError("Invalid YouTube URL format")

    return url


def validate_request_content(content: str, request_type: str) -> None:
    """Validate request content based on type (validation only, no data construction)"""
    if request_type in ["video", "playlist"]:
        _validate_video_content(content, request_type)
    elif request_type == "topic":
        _validate_topic_content(content)
    else:
        raise ValidationError(f"Invalid request type: {request_type}")


def _validate_video_content(content: str, request_type: str) -> None:
    """Validate video/playlist URL content"""
    if not content or not content.strip():
        raise ValidationError("URL cannot be empty")

    content = content.strip()

    # TODO
    # Got three branches doing similar things -- checking youtube URL
    # Need to better handle this by cleaning up the below logic
    try:
        # Use YouTube URL validation
        validate_youtube_url(content)
    except (ValidationError, ValueError) as e:
        raise ValidationError(f"Invalid YouTube {request_type} URL: {e}") from e

    # Validate specific URL type
    if request_type == "video" and not _is_video_url(content):
        raise ValidationError(
            "URL must be a YouTube video URL, not a playlist or channel"
        )
    elif request_type == "playlist" and not _is_playlist_url(content):
        raise ValidationError(
            "URL must be a YouTube playlist URL, not a video or channel"
        )


def _validate_topic_content(content: str) -> None:
    """Validate topic search query content"""
    if not content or not content.strip():
        raise ValidationError("Search query cannot be empty")

    content = content.strip()

    # Length validation
    if len(content) < 4:
        raise ValidationError("Search query must be at least 4 characters long")

    if len(content) > 500:
        raise ValidationError("Search query cannot exceed 500 characters")

    # URL detection - reject URLs for topic searches
    url_indicators = [
        r"https?://",  # Protocol
        r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # www.domain.com
        r"[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|io|co|uk|de|fr|jp|cn|be)\b",  # domain.tld
    ]

    if any(re.search(pattern, content, re.IGNORECASE) for pattern in url_indicators):
        raise ValidationError(
            "Topic searches should be text queries, not URLs. Use 'video' or 'playlist' type for URLs."
        )

    # Content validation - sanitize and check for suspicious patterns
    if _contains_suspicious_patterns(content):
        raise ValidationError("Search query contains potentially unsafe content")


def _is_video_url(url: str) -> bool:
    video_patterns = [
        r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&(?!list=)[\w=&-]*)*$",
        r"https?://(?:www\.)?youtu\.be/[\w-]+(?:\?(?!list=)[\w=&-]*)*$",
    ]

    return any(re.match(pattern, url.strip()) for pattern in video_patterns)


def _is_playlist_url(url: str) -> bool:
    playlist_patterns = [
        r"https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+",
        r"https?://(?:www\.)?youtube\.com/watch\?.*list=[\w-]+",
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
        special_char_ratio = sum(
            1
            for c in query
            if not c.isalnum() and not c.isspace() and c not in "'-.,!?:;()"
        ) / len(query)
        if special_char_ratio > 0.5:  # More than 50% special characters
            return True

    # Check for obviously malicious patterns (very targeted)
    malicious_patterns = [
        r"(?i)\b(union|select)\s+(select|from|where)\b",  # SQL injection
        r"(?i);\s*(drop|delete|truncate)\s+table\b",  # Destructive SQL
        r"(?is)<script[^>]*>[\s\S]*?</script>",  # Multi-line <script>
        r"(?i)\bjavascript\s*:\s*\S+",  # JS protocol
    ]

    return any(re.search(p, query) for p in malicious_patterns)
