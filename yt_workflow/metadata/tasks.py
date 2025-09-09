import logging

from celery import shared_task

from yt_workflow.shared.clients.broker_client import get_metadata

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="yt_workflow.process_metadata")
def process_metadata(self, video_id: str):  # type: ignore
    """Process metadata for a single video"""
    try:
        logger.info(f"Fetching metadata for video {video_id}")

        # Fetch metadata from broker client
        metadata_response = get_metadata(video_id)

        logger.info(f"Successfully fetched metadata for video {video_id}")

        # Return the raw JSON response
        return {"success": True, "video_id": video_id, "metadata": metadata_response}

    except Exception as e:
        logger.error(f"Failed to fetch metadata for video {video_id}: {str(e)}")
        return {"success": False, "video_id": video_id, "error": str(e)}
