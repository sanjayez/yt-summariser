"""
Unified gateway views for session management with rate limiting
"""

from pydantic import ValidationError
from rest_framework import status
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from rest_framework.views import APIView

from api.schemas import UnifiedProcessRequest, UnifiedProcessResponse
from api.services.session_service import SessionService
from api.utils import get_friendly_error_message
from telemetry.logging import get_logger

logger = get_logger(__name__)


class UnifiedGatewayView(APIView):
    """
    Unified gateway for all processing requests (video, playlist, topic).
    Handles session management, rate limiting, and request validation.
    """

    # TODO: Add OpenAPI documentation when drf_spectacular is available
    def post(self, request):
        """
        This endpoint:
        1. Validates the request data
        2. Creates or retrieves a session
        3. Checks rate limits (3 requests per day per session)
        4. Returns session info and processing status
        """
        try:
            # Use Pydantic schema for validation
            try:
                validated_request = UnifiedProcessRequest(**request.data)
                content = validated_request.content
                request_type = validated_request.type
            except ValidationError as e:
                logger.warning(f"Request validation failed: {e}")
                friendly_message = get_friendly_error_message(e)
                unified_response = UnifiedProcessResponse(
                    status="error",
                    message=friendly_message,
                )
                return Response(
                    unified_response.model_dump(exclude_none=True),
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except (ParseError, ValueError) as e:
                logger.warning(f"JSON parse error: {e}")
                unified_response = UnifiedProcessResponse(
                    status="error",
                    message="Invalid JSON format",
                )
                return Response(
                    unified_response.model_dump(exclude_none=True),
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")

            # Get or create session
            session_result = SessionService.get_or_create_session(request, session_id)

            # Handle invalid session case
            if len(session_result) == 3 and session_result[2] == "invalid_session":
                unified_response = UnifiedProcessResponse(
                    status="error",
                    message="Invalid session ID provided",
                )
                return Response(
                    unified_response.model_dump(exclude_none=True),
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Extract session and is_new flag
            session, is_new, _ = session_result

            # Check rate limit and update counters
            allowed, response_data = SessionService.check_rate_limit(
                session, request_type
            )

            if not allowed:
                # Map service status to HTTP status
                status_str = response_data.get("status", "error")

                if status_str == "rate_limited":
                    unified_response = UnifiedProcessResponse(
                        status="rate_limited",
                        message="Daily limit reached. Try again tomorrow",
                    )
                    return Response(
                        unified_response.model_dump(exclude_none=True),
                        status=status.HTTP_429_TOO_MANY_REQUESTS,
                    )
                else:
                    unified_response = UnifiedProcessResponse(
                        status="error",
                        message="Rate limit check failed",
                    )
                    return Response(
                        unified_response.model_dump(exclude_none=True),
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )

            # Log successful request
            logger.info(
                f"Unified request accepted - Session: {str(session.session_id)[:8]}, "
                f"Type: {request_type}, Content: {content[:50]}{'...' if len(content) > 50 else ''}, "
                f"New Session: {is_new}"
            )

            # Build success response with dynamic fields
            response_fields = {
                "status": "processing",
                "message": f"{request_type.title()} request accepted for processing",
                "remaining_limit": response_data["remaining_limit"],
            }

            # Include session_id only for new sessions
            if is_new:
                response_fields["session_id"] = str(session.session_id)

            unified_response = UnifiedProcessResponse(**response_fields)
            return Response(
                unified_response.model_dump(exclude_none=True),
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Unexpected error in UnifiedGatewayView: {e}", exc_info=True)
            unified_response = UnifiedProcessResponse(
                status="error",
                message="Internal server error occurred",
            )
            return Response(
                unified_response.model_dump(exclude_none=True),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
