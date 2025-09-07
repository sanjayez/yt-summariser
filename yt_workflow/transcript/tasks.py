from celery import shared_task


@shared_task(bind=True, name="yt_workflow.process_transcript")
def process_transcript(self, video_id: str):  # type: ignore
    """Process transcript for a single video"""

    # try:
    #     # TODO: Implement actual transcript processing
    #     # from yt_workflow.transcript.services.transcript_service import TranscriptService
    #     # service = TranscriptService()
    #     # transcript = service.fetch_transcript(video_id)
    #     # processed = service.process_transcript(transcript)

    #     # For now, just mark as completed
    #     from yt_workflow.transcript.models import TranscriptResult
    #     TranscriptResult.objects.create(  # type: ignore
    #         video_id=video_id,
    #         status='success'
    #     )

    #     return {"video_id": video_id, "status": "success", "type": "transcript"}
    # except Exception as e:
    #     return {"video_id": video_id, "status": "failed", "type": "transcript", "error": str(e)}
