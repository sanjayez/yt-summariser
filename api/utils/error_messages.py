"""
Error Message Utilities
Converts technical validation errors to user-friendly messages
"""

from typing import Any


# TODO
# Need to improve the error message lookup, current lookup for the entire string seems inefficient
def get_friendly_error_message(validation_error: Any) -> str:
    """Convert technical validation errors (Django/Pydantic/other) to user-friendly messages"""
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
    if (
        "Input should be 'video', 'playlist' or 'topic'" in error_str
        or "Invalid request type" in error_str
    ):
        return "Request type must be 'video', 'playlist', or 'topic'"

    # Content validation errors
    if "Content cannot be empty" in error_str:
        return "Content cannot be empty"
    elif "URL cannot be empty" in error_str:
        return "URL cannot be empty"
    elif "Search query cannot be empty" in error_str:
        return "Search query cannot be empty"
    elif "Invalid YouTube video URL" in error_str or (
        "Invalid YouTube URL format" in error_str
        or "Must be a valid YouTube URL" in error_str
    ):
        return "Please provide a valid YouTube URL"
    elif "URL must be a YouTube video URL" in error_str:
        return "URL must be a YouTube video, not a playlist or channel"
    elif "URL must be a YouTube playlist URL" in error_str:
        return "URL must be a YouTube playlist, not a video or channel"
    elif "Search query must be at least 4 characters" in error_str:
        return "Search query must be at least 4 characters long"
    elif "Search query cannot exceed 500 characters" in error_str:
        return "Search query cannot exceed 500 characters"
    elif "Topic searches should be text queries, not URLs" in error_str:
        return "Invalid request. For URLs, please use video / playlist type instead."
    elif "potentially unsafe content" in error_str:
        return "Search query contains potentially unsafe content"

    # Fallback for unknown errors
    return "Invalid request format"
