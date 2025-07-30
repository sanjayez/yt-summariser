"""
Custom exception classes for the YT Summariser application.

This module defines all custom exception classes used throughout the application,
providing a hierarchy of exceptions with consistent error handling and context.
"""

from typing import Any, Dict, Optional


class BaseYTSummarizerError(Exception):
    """Base exception class for all YT Summarizer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception.
        
        Args:
            message: Error message
            details: Additional error details as a dictionary
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error message with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class VideoProcessingError(BaseYTSummarizerError):
    """Exception raised during video processing operations."""
    
    def __init__(self, message: str, video_id: Optional[str] = None, 
                 step: Optional[str] = None, **kwargs: Any):
        """
        Initialize video processing error.
        
        Args:
            message: Error message
            video_id: YouTube video ID
            step: Processing step where error occurred
            **kwargs: Additional error details
        """
        details = {"video_id": video_id, "step": step}
        details.update(kwargs)
        super().__init__(message, details)


class ExternalServiceError(BaseYTSummarizerError):
    """Exception raised when external services fail (YouTube, OpenAI, etc.)."""
    
    def __init__(self, service: str, message: str, status_code: Optional[int] = None,
                 response_body: Optional[str] = None, **kwargs: Any):
        """
        Initialize external service error.
        
        Args:
            service: Name of the external service
            message: Error message
            status_code: HTTP status code if applicable
            response_body: Response body from the service
            **kwargs: Additional error details
        """
        details = {
            "service": service,
            "status_code": status_code,
            "response_body": response_body
        }
        details.update(kwargs)
        super().__init__(f"{service} error: {message}", details)


class ValidationError(BaseYTSummarizerError):
    """Exception raised for validation failures."""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, **kwargs: Any):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: The invalid value
            **kwargs: Additional error details
        """
        details = {"field": field, "value": value}
        details.update(kwargs)
        super().__init__(message, details)


# Export all exception classes
__all__ = [
    'BaseYTSummarizerError',
    'VideoProcessingError',
    'ExternalServiceError',
    'ValidationError'
]