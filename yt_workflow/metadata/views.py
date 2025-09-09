import logging

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .tasks import process_metadata

logger = logging.getLogger(__name__)


@api_view(["GET"])
def test_metadata_fetch(request, video_id: str):
    """
    Simple demo view to test metadata fetching via Celery task.
    This is a blocking call that waits for the task to complete.
    """
    try:
        logger.info(f"Starting metadata test for video {video_id}")

        # Start the Celery task
        task_result = process_metadata.delay(video_id)

        # Wait for the task to complete (blocking)
        # This is fine for testing but not recommended for production
        result = task_result.get(timeout=30)  # 30 second timeout

        logger.info(f"Metadata task completed for video {video_id}")

        return Response(
            {"task_id": task_result.id, "result": result, "status": "completed"},
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Metadata test failed for video {video_id}: {str(e)}")
        return Response(
            {"error": str(e), "video_id": video_id, "status": "failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
