from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
import logging
from celery_progress.backend import ProgressRecorder

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
             name='video_processor.update_overall_status',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['status_hard_limit'],
             max_retries=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['max_retries'],
             default_retry_delay=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['countdown'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['status_update']['jitter'])
@idempotent_task
def update_overall_status(self, embedding_result, url_request_id):
    """
    Update overall processing status after workflow completion.
    
    Args:
        embedding_result (dict): Result from previous embedding task
        url_request_id (int): ID of the URLRequestTable to process
        
    Returns:
        str: Status update result message
        
    Raises:
        Exception: If status update fails after retries
    """
    url_request = None
    
    # Check if video was excluded in previous stage
    if embedding_result and embedding_result.get('excluded'):
        logger.info(f"Video was excluded - ensuring final status is correctly set: {embedding_result.get('exclusion_reason')}")
        # Get URLRequest and ensure it's marked as excluded (not processing)
        url_request = URLRequestTable.objects.get(id=url_request_id)
        if url_request.status == 'processing':
            url_request.status = 'failed'
            url_request.failure_reason = 'excluded'
            url_request.save()
            logger.info(f"Updated excluded video status from processing to failed/excluded")
        
        return f"Video excluded in previous stage: {embedding_result.get('exclusion_reason')}"
    
    try:
        # Set initial progress using celery_progress
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(0, 100, "Updating status")
        
        update_task_progress(self, TASK_STATES['UPDATING_STATUS'], 50)
        
        # Optimized query with select_related to avoid N+1 queries
        url_request = URLRequestTable.objects.select_related(
            'video_metadata',
            'video_metadata__video_transcript'
        ).get(id=url_request_id)
        
        logger.info(f"Updating overall status for request {url_request_id}")
        
        # Check for transcript extraction failures and provide descriptive error messages
        transcript_failed = False
        failure_reason = None
        
        # Check if transcript exists and failed
        if hasattr(url_request, 'video_metadata') and url_request.video_metadata:
            try:
                video_transcript = url_request.video_metadata.video_transcript
                if video_transcript and video_transcript.status == 'failed':
                    transcript_failed = True
                    failure_reason = "Transcript extraction failed"
                    # Check embedding result for specific error details
                    if isinstance(embedding_result, dict) and embedding_result.get('error'):
                        error_msg = embedding_result.get('error', '')
                        if '429' in error_msg or 'Too Many Requests' in error_msg:
                            failure_reason = "Rate limited by YouTube - transcript temporarily unavailable"
                        elif 'No transcript' in error_msg:
                            failure_reason = "No transcript available for this video"
                        else:
                            failure_reason = "Transcript extraction failed"
            except:
                # If no transcript object exists, check embedding result
                if isinstance(embedding_result, dict) and embedding_result.get('transcript_chunks_embedded', 0) == 0:
                    if embedding_result.get('error'):
                        error_msg = embedding_result.get('error', '')
                        if '429' in error_msg or 'Too Many Requests' in error_msg:
                            transcript_failed = True
                            failure_reason = "Rate limited by YouTube - transcript temporarily unavailable"
        
        # Update status (this should be very fast)
        with transaction.atomic():
            # If transcript failed, provide clear error message to video metadata
            if transcript_failed and hasattr(url_request, 'video_metadata'):
                video_metadata = url_request.video_metadata
                if video_metadata.status == 'processing':
                    video_metadata.status = 'failed'
                    video_metadata.error_message = failure_reason or "Processing failed"
                    video_metadata.save()
                    logger.info(f"Marked video metadata as failed: {failure_reason}")
                
                # Set failure reason on URLRequest if transcript failed
                if not url_request.failure_reason:
                    url_request.failure_reason = 'no_transcript'
                    url_request.save()
            
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
            'transcript_segments': embedding_result.get('transcript_chunks_embedded', 0) if isinstance(embedding_result, dict) else 0,
            'failure_reason': failure_reason if transcript_failed else None,
            'error_message': getattr(url_request.video_metadata, 'error_message', None) if hasattr(url_request, 'video_metadata') else None
        }
        
        # Set final progress using celery_progress
        progress_recorder.set_progress(100, 100, "Status updated")
        
        return f"Processing complete - {result}"
        
    except SoftTimeLimitExceeded:
        # Status update task is approaching timeout - this should never happen as it should be very fast
        logger.error(f"Status update soft timeout reached for request {url_request_id} - this indicates a serious database issue")
        
        try:
            # Last-ditch effort to mark as failed
            if url_request:
                url_request.status = 'failed'
                url_request.save()
                logger.error(f"Emergency status update to failed for {url_request_id}")
                
        except Exception as cleanup_error:
            logger.critical(f"Failed to emergency update status during timeout - database may be down: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Status update critical timeout for request {url_request_id}")
        
    except Exception as e:
        logger.error(f"Failed to update overall status for {url_request_id}: {e}")
        
        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task('update_overall_status', self.request.id, [url_request_id], {}, e)
        
        raise 