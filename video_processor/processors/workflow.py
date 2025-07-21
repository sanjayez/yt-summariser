from celery import shared_task, chain, group
from celery.exceptions import SoftTimeLimitExceeded
import logging

from api.models import URLRequestTable
from ..config import YOUTUBE_CONFIG
from ..validators import validate_youtube_url
from ..utils import handle_dead_letter_task
from .metadata import extract_video_metadata
from .transcript import extract_video_transcript
from .summary import generate_video_summary
from .content_classifier import classify_and_exclude_video_llm
from .status import update_overall_status

logger = logging.getLogger(__name__)


@shared_task(bind=True, 
             name='video_processor.process_youtube_video',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['workflow_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['workflow_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['metadata']['jitter'])
def process_youtube_video(self, url_request_id):
    """
    Process a single YouTube video through the complete pipeline.
    
    Executes the full workflow: metadata → transcript → summary → classification → status update
    Note: Embedding has been moved to a separate filtering layer
    
    Args:
        url_request_id (int): ID of the URLRequestTable to process
        
    Returns:
        str: Success message with request ID
        
    Raises:
        Exception: If workflow initiation fails
    """
    url_request = None
    
    try:
        # Validate input and get URL request with related data to avoid N+1 queries
        url_request = URLRequestTable.objects.select_related(
            'video_metadata',
            'video_metadata__video_transcript'
        ).get(id=url_request_id)
        validate_youtube_url(url_request.url)
        
        # Store task ID for progress tracking
        url_request.celery_task_id = self.request.id
        url_request.save()
        
        logger.info(f"Starting video processing pipeline for request {url_request_id}")
        
        # Execute workflow chain: each task receives previous result as first argument
        # Note: Embedding removed - will be handled by separate filtering layer
        workflow = chain(
            extract_video_metadata.s(url_request_id),
            extract_video_transcript.s(url_request_id),  # Pass url_request_id explicitly to ensure continuity
            generate_video_summary.s(url_request_id),    # Pass url_request_id explicitly to ensure continuity
            classify_and_exclude_video_llm.s(url_request_id),  # Pass url_request_id explicitly to ensure continuity
            update_overall_status.s(url_request_id)      # Pass url_request_id explicitly to ensure continuity
        )
        
        # Capture workflow result to enable result tracking and eliminate fire-and-forget behavior
        result = workflow.apply_async()
        
        # Store chain task ID for progress tracking and result retrieval
        url_request.chain_task_id = result.id
        url_request.save(update_fields=['chain_task_id'])
        
        logger.info(f"Initiated processing pipeline for request {url_request_id}")
        return f"Initiated complete processing pipeline for request {url_request_id}"
        
    except SoftTimeLimitExceeded:
        # Workflow orchestration is approaching timeout - this should not happen often
        logger.warning(f"Workflow orchestration soft timeout reached for request {url_request_id}")
        
        try:
            # Mark the URL request as failed due to orchestration timeout
            if url_request:
                url_request.status = 'failed'
                url_request.failure_reason = 'technical_failure'
                url_request.save()
                logger.error(f"Marked workflow as failed due to orchestration timeout: {url_request_id}")
                
        except Exception as cleanup_error:
            logger.error(f"Failed to update workflow status during timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Workflow orchestration timeout for request {url_request_id}")
        
    except Exception as e:
        logger.error(f"Failed to initiate processing pipeline for {url_request_id}: {e}")
        handle_dead_letter_task('process_youtube_video', self.request.id, [url_request_id], {}, e)
        raise


@shared_task(bind=True, 
             name='video_processor.process_parallel_videos',
             soft_time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_soft_limit'],
             time_limit=YOUTUBE_CONFIG['TASK_TIMEOUTS']['parallel_hard_limit'],
             autoretry_for=(Exception,),
             retry_backoff=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['backoff'],
             retry_jitter=YOUTUBE_CONFIG['RETRY_CONFIG']['parallel']['jitter'])
def process_parallel_videos(self, url_request_ids):
    """
    Process multiple videos in parallel.
    
    Each video goes through the complete pipeline independently and concurrently.
    
    Args:
        url_request_ids (list): List of URLRequestTable IDs to process
        
    Returns:
        dict: Processing result with parallel job information
    """
    logger.info(f"Starting parallel processing for {len(url_request_ids)} videos")
    
    try:
        # Validate all URL requests exist
        existing_requests = URLRequestTable.objects.filter(id__in=url_request_ids)
        if existing_requests.count() != len(url_request_ids):
            missing_ids = set(url_request_ids) - set(existing_requests.values_list('id', flat=True))
            logger.error(f"Missing URLRequestTable entries: {missing_ids}")
            return {
                'status': 'failed',
                'error': 'Missing URL request entries',
                'missing_ids': list(missing_ids)
            }
        
        # Create parallel group of individual video processing tasks
        video_processing_group = group(
            process_youtube_video.s(url_request_id) for url_request_id in url_request_ids
        )
        
        # Execute parallel processing
        result = video_processing_group.apply_async()
        
        logger.info(f"Launched parallel processing for {len(url_request_ids)} videos, group ID: {result.id}")
        
        return {
            'status': 'processing',
            'group_id': result.id,
            'total_videos': len(url_request_ids),
            'url_request_ids': url_request_ids,
            'processing_type': 'parallel'
        }
        
    except SoftTimeLimitExceeded:
        # Parallel orchestration is approaching timeout
        logger.warning(f"Parallel orchestration soft timeout reached for {len(url_request_ids)} videos")
        
        try:
            # Mark all URL requests as failed due to orchestration timeout
            URLRequestTable.objects.filter(id__in=url_request_ids).update(
                status='failed',
                failure_reason='technical_failure'
            )
            logger.error(f"Marked {len(url_request_ids)} parallel videos as failed due to orchestration timeout")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to update parallel workflow status during timeout cleanup: {cleanup_error}")
        
        # Re-raise to mark task as failed
        raise Exception(f"Parallel workflow orchestration timeout for {len(url_request_ids)} videos")
        
    except Exception as e:
        logger.error(f"Failed to start parallel processing: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to start parallel processing',
            'details': str(e)
        } 