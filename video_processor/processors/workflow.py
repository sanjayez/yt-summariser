from celery import shared_task, chain, group, chord
import logging

from api.models import URLRequestTable
from ..validators import validate_youtube_url
from ..utils import handle_dead_letter_task
from .metadata import extract_video_metadata
from .transcript import extract_video_transcript
from .status import update_overall_status

logger = logging.getLogger(__name__)

# Main entry point - creates task chain
@shared_task(bind=True)
def process_youtube_video(self, url_request_id):
    """
    Entry point that creates and executes the video processing chain.
    Uses Celery's chain primitive for proper task orchestration.
    """
    try:
        # Validate input
        url_request = URLRequestTable.objects.select_related('video_metadata', 'video_transcript').get(id=url_request_id)
        
        # Validate URL
        validate_youtube_url(url_request.url)
        
        logger.info(f"Starting video processing pipeline for request {url_request_id}")
        
        parallel_tasks = group(
            extract_video_transcript.s(url_request_id),
        )
        
        workflow = chain(
            extract_video_metadata.s(url_request_id),
            chord(parallel_tasks, update_overall_status.s(url_request_id))
        )
        
        # Execute workflow
        result = workflow.apply_async()
        
        return f"Initiated processing pipeline for request {url_request_id}"
        
    except Exception as e:
        logger.error(f"Failed to initiate processing pipeline for {url_request_id}: {e}")
        handle_dead_letter_task('process_youtube_video', self.request.id, [url_request_id], {}, e)
        raise 