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
                    'details': str(e)
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
            session, is_new = SessionService.get_or_create_session(request, session_id)
            
            # Check rate limit and update counters
            allowed, response_data = SessionService.check_rate_limit(session, request_type)
            
            if not allowed:
                # Rate limit exceeded
                logger.warning(f"Rate limit exceeded for session {str(session.session_id)[:8]}")
                # Use UnifiedProcessResponse for rate-limited response as well
                unified_response = UnifiedProcessResponse(
                    session_id=response_data['session_id'],
                    status=response_data['status'],
                    remaining_limit=response_data['remaining_limit']
                )
                return Response(unified_response.model_dump(), status=status.HTTP_429_TOO_MANY_REQUESTS)
            
            # Log successful request
            logger.info(
                f"Unified request accepted - Session: {str(session.session_id)[:8]}, "
                f"Type: {request_type}, Content: {content[:50]}{'...' if len(content) > 50 else ''}, "
                f"New Session: {is_new}"
            )
            
            # Return clean, structured response using Pydantic schema
            unified_response = UnifiedProcessResponse(
                session_id=response_data['session_id'],
                status=response_data['status'],
                remaining_limit=response_data['remaining_limit']
            )
            
            return Response(unified_response.model_dump(), status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error in UnifiedGatewayView: {e}", exc_info=True)
            return Response({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred while processing your request'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionInfoView(APIView):
    """
    Get detailed information about a session for debugging/monitoring.
    """
    
    # TODO: Add OpenAPI documentation when drf_spectacular is available
    def get(self, request):
        """Get session information from X-Session-ID header"""
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id:
            return Response({
                'error': 'X-Session-ID header is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Validate UUID format first
            from uuid import UUID
            try:
                UUID(session_id)
            except ValueError:
                logger.error(f"Error retrieving session info: {session_id} is not a valid UUID")
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            from api.models import UnifiedSession
            session = UnifiedSession.objects.filter(session_id=session_id).first()
            
            if not session:
                return Response({
                    'error': 'Session not found'
                }, status=status.HTTP_404_NOT_FOUND)
            
            session_info = SessionService.get_session_info(session)
            return Response(session_info, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error retrieving session info: Service error")
            return Response({
                'error': 'Internal server error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
