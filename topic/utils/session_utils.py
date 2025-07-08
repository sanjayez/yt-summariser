import logging
from django.db import transaction
from django.core.exceptions import ValidationError
from api.utils.get_client_ip import get_client_ip
from topic.models import SearchSession

logger = logging.getLogger(__name__)


def get_or_create_session(request):
    """
    Get or create a SearchSession for the current request.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        SearchSession: The session instance for the client
        
    Raises:
        ValidationError: If session creation fails due to validation errors
        Exception: For other unexpected errors during session creation
    """
    try:
        # Extract client IP address using existing utility
        ip_address = get_client_ip(request)
        
        if not ip_address:
            logger.warning("Could not determine client IP address from request")
            ip_address = '127.0.0.1'  # Fallback for development
        
        logger.debug(f"Processing session request for IP: {ip_address}")
        
        # Use atomic transaction to ensure data consistency
        with transaction.atomic():
            # Try to get existing active session for this IP
            # Look for the most recent session that isn't failed
            session = SearchSession.objects.filter(
                user_ip=ip_address
            ).exclude(
                status='failed'
            ).order_by('-created_at').first()
            
            if session:
                logger.debug(f"Found existing session {session.session_id} for IP {ip_address}")
                return session
            
            # Create new session if none exists or all previous sessions failed
            session = SearchSession.objects.create(
                user_ip=ip_address,
                status='processing'
            )
            
            logger.info(f"Created new session {session.session_id} for IP {ip_address}")
            return session
            
    except ValidationError as e:
        logger.error(f"Validation error creating session for IP {ip_address}: {e}")
        raise ValidationError(f"Session creation failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error creating session for IP {ip_address}: {e}")
        raise Exception(f"Session creation failed: {e}")


def update_session_status(session, status, save=True):
    """
    Update the status of a SearchSession.
    
    Args:
        session: SearchSession instance
        status: New status ('processing', 'failed', 'success')
        save: Whether to save the session immediately (default: True)
        
    Returns:
        SearchSession: The updated session instance
        
    Raises:
        ValueError: If status is not valid
    """
    valid_statuses = ['processing', 'failed', 'success']
    
    if status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_statuses}")
    
    try:
        session.status = status
        if save:
            session.save(update_fields=['status'])
            logger.debug(f"Updated session {session.session_id} status to '{status}'")
        
        return session
        
    except Exception as e:
        logger.error(f"Error updating session {session.session_id} status: {e}")
        raise