"""
Error Message Utilities
Converts technical validation errors to user-friendly messages
"""

from pydantic import ValidationError


def get_friendly_error_message(validation_error: ValidationError) -> str:
    """Convert technical Pydantic errors to user-friendly messages"""
    error_str = str(validation_error)

    # Field required errors
    if "Field required" in error_str:
        if "type\n" in error_str:
            return "Please specify request type (video, playlist, or topic)"
        elif "content\n" in error_str:
            return "Please provide content (URL or search query)"
        else:
            return "Missing required field"

    # Invalid type errors
    if "Input should be 'video', 'playlist' or 'topic'" in error_str:
        return "Request type must be 'video', 'playlist', or 'topic'"

    # Content validation errors
    if "Content cannot be empty" in error_str:
        return "Content cannot be empty"
    elif "Invalid YouTube video URL" in error_str:
        return "Please provide a valid YouTube URL"
    elif "URL must be a YouTube video URL" in error_str:
        return "URL must be a YouTube video, not a playlist or channel"
    elif "URL must be a YouTube playlist URL" in error_str:
        return "URL must be a YouTube playlist, not a video or channel"
    elif "Search query must be at least 4 characters" in error_str:
        return "Search query must be at least 4 characters long"
    elif "Search query cannot exceed 500 characters" in error_str:
        return "Search query cannot exceed 500 characters"
    elif "potentially unsafe content" in error_str:
        return "Search query contains potentially unsafe content"

    # Fallback for unknown errors
    return "Invalid request format"
