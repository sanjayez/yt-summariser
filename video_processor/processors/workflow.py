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
@shared_task(bind=True)
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
        url_request = URLRequestTable.objects.select_related('video_metadata__video_transcript').get(id=url_request_id)
        
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