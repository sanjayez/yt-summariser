"""
Unified gateway views for session management and request routing.
Provides a single entry point for all processing requests with rate limiting.
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ParseError
from api.schemas import UnifiedProcessRequest, UnifiedProcessResponse
from api.services.session_service import SessionService
from telemetry.logging import get_logger
from pydantic import ValidationError

logger = get_logger(__name__)


class UnifiedGatewayView(APIView):
    """
    Unified gateway for all processing requests (video, playlist, topic).
    Handles session management, rate limiting, and request validation.
    """
    
    # TODO: Add OpenAPI documentation when drf_spectacular is available
    def post(self, request):
        """
        Handle unified processing requests with session management.
        
        This endpoint:
        1. Validates the request data
        2. Creates or retrieves a session
        3. Checks rate limits (3 requests per day per session)
        4. Returns session info and processing status
        
        Future versions will route to actual processors.
        """
        try:
            # Use Pydantic schema for validation
            try:
                validated_request = UnifiedProcessRequest(**request.data)
                content = validated_request.content
                request_type = validated_request.type
            except ValidationError as e:
                logger.warning(f"Request validation failed: {e}")
                return Response({
                    'error': 'Invalid request data',
                    'message': 'Please check your request format'
                }, status=status.HTTP_400_BAD_REQUEST)
            except (ParseError, ValueError) as e:
                logger.warning(f"JSON parse error: {e}")
                return Response({
                    'error': 'Invalid JSON data',
                    'message': str(e)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get session ID from headers
            session_id = request.headers.get('X-Session-ID')
            
            # Get or create session
            session_result = SessionService.get_or_create_session(request, session_id)
            
            # Handle invalid session case
            if len(session_result) == 3 and session_result[2] == "invalid_session":
                return Response({
                    'error': 'Invalid session',
                    'message': 'Invalid session.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract session and is_new flag
            session, is_new = session_result[0], session_result[1]
            
            # Check rate limit and update counters
            allowed, response_data = SessionService.check_rate_limit(session, request_type)
            
            if not allowed:
                # Map service status to HTTP status
                status_str = response_data.get('status', 'error')
                http_status = (
                    status.HTTP_429_TOO_MANY_REQUESTS
                    if status_str == 'rate_limited'
                    else status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                # Add user-friendly message for rate limiting
                if status_str == 'rate_limited':
                    return Response({
                        'status': 'rate_limited',
                        'remaining_limit': response_data['remaining_limit'],
                        'message': 'Daily limit reached. Try again tomorrow'
                    }, status=http_status)
                else:
                    unified_response = UnifiedProcessResponse(
                        status=status_str,
                        remaining_limit=response_data['remaining_limit']
                    )
                    return Response(unified_response.model_dump(), status=http_status)
            
            # Log successful request
            logger.info(
                f"Unified request accepted - Session: {str(session.session_id)[:8]}, "
                f"Type: {request_type}, Content: {content[:50]}{'...' if len(content) > 50 else ''}, "
                f"New Session: {is_new}"
            )
            
            # Return clean, structured response using Pydantic schema
            # Include session_id only for new sessions
            unified_response = UnifiedProcessResponse(
                status=response_data['status'],
                remaining_limit=response_data['remaining_limit'],
                session_id=str(session.session_id) if is_new else None
            )
            
            return Response(unified_response.model_dump(exclude_none=True), status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error in UnifiedGatewayView: {e}", exc_info=True)
            return Response({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred while processing your request'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

