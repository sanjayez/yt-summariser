from celery import shared_task
from django.db import transaction
import logging

from api.models import URLRequestTable
from ..models import update_url_request_status
from ..config import YOUTUBE_CONFIG, TASK_STATES
from ..utils import (
    timeout, idempotent_task, handle_dead_letter_task, 
    update_task_progress
)

logger = logging.getLogger(__name__)

# Task 3: Update Overall Status
@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'],
             retry_kwargs=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update'])
@idempotent_task
def update_overall_status(self, transcript_result, url_request_id):
    """
    Atomic task: Update overall status with timeout protection.
    """
    try:
        update_task_progress(self, TASK_STATES['UPDATING_STATUS'], 50)
        
        # Optimized query with select_related
        url_request = URLRequestTable.objects.select_related('video_metadata__video_transcript').get(id=url_request_id)
        
        logger.info(f"Updating overall status for request {url_request_id}")
        
        # Update status with timeout protection
        with timeout(YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_update_timeout'], "Status update"):
            with transaction.atomic():
                update_url_request_status(url_request)
        
        # Refresh to get updated status
        url_request.refresh_from_db()
        
        update_task_progress(self, TASK_STATES['COMPLETED'], 100, {
            'final_status': url_request.status,
            'has_metadata': hasattr(url_request, 'video_metadata'),
            'has_transcript': hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript'),
        })
        
        logger.info(f"Final status for request {url_request_id}: {url_request.status}")
        
        result = {
            'status': url_request.status,
            'metadata_status': getattr(url_request.video_metadata, 'status', None) if hasattr(url_request, 'video_metadata') else None,
            'transcript_status': getattr(url_request.video_metadata.video_transcript, 'status', None) if hasattr(url_request, 'video_metadata') and hasattr(url_request.video_metadata, 'video_transcript') else None,
            'transcript_segments': transcript_result.get('transcript_segments', 0) if isinstance(transcript_result, dict) else 0
        }
        
        return f"Processing complete - {result}"
        
    except Exception as e:
        logger.error(f"Failed to update overall status for {url_request_id}: {e}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('update_overall_status', self.request.id, [url_request_id], {}, e)
        
        raise 