"""
Custom API Exceptions with telemetry integration.
Provides specific exception types for different API error scenarios.
"""

from typing import Any


class APIException(Exception):
    """
    Base exception for API-specific errors.
    Integrates with the telemetry exception handling system.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ValidationError(APIException):
    """Exception raised when request validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)


class VideoNotFoundError(APIException):
    """Exception raised when a video processing request is not found."""

    def __init__(self, request_id: str, message: str | None = None):
        default_message = f"No video processing request found with ID: {request_id}"
        super().__init__(
            message or default_message,
            status_code=404,
            details={"request_id": request_id},
        )


class VideoProcessingIncompleteError(APIException):
    """Exception raised when video processing is not yet complete."""

    def __init__(
        self, request_id: str, current_status: str, message: str | None = None
    ):
        default_message = f"Video processing is not complete (status: {current_status})"
        super().__init__(
            message or default_message,
            status_code=202,
            details={"request_id": request_id, "current_status": current_status},
        )


class VideoProcessingFailedError(APIException):
    """Exception raised when video processing has failed."""

    def __init__(
        self,
        request_id: str,
        failure_reason: str | None = None,
        message: str | None = None,
    ):
        default_message = "Video processing failed and cannot be completed"
        super().__init__(
            message or default_message,
            status_code=500,
            details={"request_id": request_id, "failure_reason": failure_reason},
        )


class TranscriptNotAvailableError(APIException):
    """Exception raised when video transcript is not available."""

    def __init__(self, request_id: str, message: str | None = None):
        default_message = "Video transcript is not available for question answering"
        super().__init__(
            message or default_message,
            status_code=404,
            details={"request_id": request_id},
        )


class SearchServiceError(APIException):
    """Exception raised when search service operations fail."""

    def __init__(
        self, message: str, search_method: str, details: dict[str, Any] | None = None
    ):
        enhanced_details = {"search_method": search_method}
        if details:
            enhanced_details.update(details)
        super().__init__(message, status_code=500, details=enhanced_details)


class ServiceInitializationError(APIException):
    """Exception raised when service initialization fails."""

    def __init__(self, service_name: str, error: str):
        message = f"Failed to initialize {service_name}: {error}"
        super().__init__(
            message,
            status_code=500,
            details={"service_name": service_name, "error": error},
        )
