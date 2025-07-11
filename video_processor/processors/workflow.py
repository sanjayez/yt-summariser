from celery import shared_task, chain, group, chord
import logging

from api.models import URLRequestTable
from ..validators import validate_youtube_url
from ..utils import handle_dead_letter_task
from .metadata import extract_video_metadata
from .transcript import extract_video_transcript
from .summary import generate_video_summary
from .embedding import embed_video_content
from .status import update_overall_status

logger = logging.getLogger(__name__)

# Main entry point - creates task chain
@shared_task(bind=True, name='video_processor.process_youtube_video')
def process_youtube_video(self, url_request_id):
    """
    Entry point that creates and executes the enhanced video processing chain.
    
    New workflow:
    1. Extract video metadata
    2. Extract transcript (parallel with metadata if needed)
    3. Generate video summary (requires transcript)
    4. Embed all content (requires summary for complete 4-layer embedding)
    5. Update overall status
    
    Uses Celery's chain primitive for proper task orchestration.
    """
    try:
        # Validate input
        url_request = URLRequestTable.objects.select_related('video_metadata').get(id=url_request_id)
        
        # Validate URL
        validate_youtube_url(url_request.url)
        
        logger.info(f"Starting enhanced video processing pipeline for request {url_request_id}")
        
        # Enhanced workflow: metadata → transcript → summary → embed → status
        workflow = chain(
            extract_video_metadata.s(url_request_id),
            extract_video_transcript.s(url_request_id),
            generate_video_summary.s(url_request_id),
            embed_video_content.s(url_request_id),
            update_overall_status.s(url_request_id)
        )
        
        # Execute workflow
        result = workflow.apply_async()
        
        logger.info(f"Initiated enhanced processing pipeline for request {url_request_id}")
        return f"Initiated enhanced processing pipeline for request {url_request_id}"
        
    except Exception as e:
        logger.error(f"Failed to initiate processing pipeline for {url_request_id}: {e}")
        handle_dead_letter_task('process_youtube_video', self.request.id, [url_request_id], {}, e)
        raise


@shared_task(bind=True, name='video_processor.process_parallel_videos')
def process_parallel_videos(self, url_request_ids):
    """
    Process multiple videos in parallel using Celery groups.
    
    This task is designed for parallel processing of search results,
    where multiple videos need to be processed simultaneously.
    
    Args:
        url_request_ids: List of URLRequestTable IDs to process
        
    Returns:
        dict: Processing result with parallel job information
    """
    logger.info(f"Starting parallel processing for {len(url_request_ids)} videos")
    
    try:
        # Validate all URL requests exist
        existing_requests = URLRequestTable.objects.filter(id__in=url_request_ids)
        if existing_requests.count() != len(url_request_ids):
            missing_ids = set(url_request_ids) - set(existing_requests.values_list('id', flat=True))
            error_msg = f"Missing URLRequestTable entries: {missing_ids}"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': 'Missing URL request entries',
                'missing_ids': list(missing_ids)
            }
        
        # Create group of video processing tasks
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
            'url_request_ids': url_request_ids
        }
        
    except Exception as e:
        logger.error(f"Failed to start parallel processing: {e}")
        return {
            'status': 'failed',
            'error': 'Failed to start parallel processing',
            'details': str(e)
        } 