"""
Session management service for unified request handling and rate limiting.
Handles session creation, validation, and rate limiting across all request types.
"""
from typing import Tuple, Dict, Any, Optional
from django.http import HttpRequest
from django.utils import timezone
from api.models import UnifiedSession
from api.utils.get_client_ip import get_client_ip
from telemetry.logging import get_logger

logger = get_logger(__name__)


class SessionService:
    """Service for managing unified sessions across all request types"""
    
    # Alpha settings
    DAILY_REQUEST_LIMIT = 3  # 3 requests per day across all types
    STALE_SESSION_DAYS = 1   # Sessions expire after 1 day
    
    @staticmethod
    def get_or_create_session(request: HttpRequest, session_id: Optional[str] = None) -> Tuple[UnifiedSession, bool, Optional[str]]:
        """
        Get existing session or create new one with IP-first validation and stale session handling.
        
        Alpha-optimized approach:
        1. Try session_id lookup (if provided) with IP and date validation
        2. Fall back to IP-based lookup for today
        3. Create new session if none exists for this IP today
        
        Args:
            request: Django HTTP request object
            session_id: Optional session ID from client
            
        Returns:
            Tuple of (session_object, is_new_session, error_type)
            error_type is None for success, "invalid_session" for invalid session_id
        """
        user_ip = get_client_ip(request)
        today = timezone.now().date()
        
        # Step 1: Try to use provided session_id (UX optimization)
        if session_id:
            try:
                # Validate UUID format first
                import uuid
                uuid.UUID(session_id)
                
                session = UnifiedSession.objects.get(session_id=session_id)
                
                # Check if session is fresh and IP matches
                if (session.user_ip == user_ip and 
                    session.created_at.date() == today):
                    logger.info(f"Retrieved existing session {str(session.session_id)[:8]} for IP {user_ip}")
                    return session, False, None
                
                # If we're here, either IP mismatch or stale session - log the specific reason
                if session.user_ip != user_ip:
                    logger.warning(f"Session {session_id[:8]} IP mismatch: expected {session.user_ip}, got {user_ip}")
                else:
                    # IP matches but session failed freshness check - must be stale
                    logger.info(f"Session {session_id[:8]} is stale (created: {session.created_at.date()}, today: {today})")
                
            except (ValueError, UnifiedSession.DoesNotExist):
                # Both malformed UUID and non-existent session get same treatment
                logger.info(f"Invalid session provided: {session_id[:8]}...")
                return None, False, "invalid_session"
            except Exception as e:
                logger.error(f"Error retrieving session {session_id}: {e}")
                return None, False, "invalid_session"
        
        # Step 2 & 3: Application-level race condition prevention
        # Check for existing session today, create if none exists
        try:
            existing_session = UnifiedSession.objects.get(
                user_ip=user_ip,
                created_at__date=today
            )
            logger.info(f"Found existing session {str(existing_session.session_id)[:8]} for IP {user_ip} today")
            return existing_session, False, None
        except UnifiedSession.DoesNotExist:
            # No session exists for this IP today, create new one
            new_session = UnifiedSession.objects.create(user_ip=user_ip)
            logger.info(f"Created new session {str(new_session.session_id)[:8]} for IP {user_ip}")
            return new_session, True, None
    
    @staticmethod
    def check_rate_limit(session: UnifiedSession, request_type: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limits and update counters.
        
        Args:
            session: UnifiedSession object
            request_type: Type of request ('video', 'playlist', 'topic')
            
        Returns:
            Tuple of (is_allowed, response_data)
        """
        # Validate request type
        if request_type not in ['video', 'playlist', 'topic']:
            return False, {
                'session_id': str(session.session_id),
                'status': 'error',
                'remaining_limit': session.get_remaining_requests(),
                'message': f'Invalid request type: {request_type}'
            }
        
        # Check if user has exceeded daily limit
        if not session.can_make_request():
            logger.warning(
                f"Rate limit exceeded for session {str(session.session_id)[:8]} "
                f"(IP: {session.user_ip}). Total requests: {session.total_requests}"
            )
            return False, {
                'session_id': str(session.session_id),
                'status': 'rate_limited',
                'remaining_limit': 0,
                'message': (
                    f'Daily limit reached ({SessionService.DAILY_REQUEST_LIMIT} requests per day). '
                    f'You have used {session.total_requests} requests today. '
                    f'Video: {session.video_requests}, Playlist: {session.playlist_requests}, '
                    f'Topic: {session.topic_requests}. Try again tomorrow!'
                )
            }
        
        # Increment counter for the request type
        try:
            session.increment_request_count(request_type)
            # Note: increment_request_count already refreshes the session object
            remaining = session.get_remaining_requests()
            
            logger.info(
                f"Request accepted for session {str(session.session_id)[:8]} "
                f"(type: {request_type}, remaining: {remaining})"
            )
            
            return True, {
                'session_id': str(session.session_id),
                'status': 'processing',
                'remaining_limit': remaining,
                'message': (
                    f'Request accepted. {remaining} requests remaining today. '
                    f'Current usage - Video: {session.video_requests}, '
                    f'Playlist: {session.playlist_requests}, Topic: {session.topic_requests}'
                )
            }
        except Exception as e:
            logger.error(f"Error updating session counters: {e}")
            return False, {
                'session_id': str(session.session_id),
                'status': 'error',
                'remaining_limit': session.get_remaining_requests(),
                'message': 'Internal error updating session'
            }
    
    @staticmethod
    def get_session_info(session: UnifiedSession) -> Dict[str, Any]:
        """
        Get detailed session information for debugging/monitoring.
        
        Args:
            session: UnifiedSession object
            
        Returns:
            Dictionary with session details
        """
        return {
            'session_id': str(session.session_id),
            'user_ip': session.user_ip,
            'total_requests': session.total_requests,
            'video_requests': session.video_requests,
            'playlist_requests': session.playlist_requests,
            'topic_requests': session.topic_requests,
            'remaining_limit': session.get_remaining_requests(),
            'created_at': session.created_at.isoformat(),
            'last_request_at': session.last_request_at.isoformat(),
            'has_account': session.user_account is not None
        }
