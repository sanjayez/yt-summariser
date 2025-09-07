from celery import shared_task


@shared_task(bind=True, name="yt_workflow.process_metadata")
def process_metadata(self, video_id: str):  # type: ignore
    """Process metadata for a single video"""

    # try:
    #     # Update video status
    #     video = VideoTable.objects.get(video_id=video_id)  # type: ignore
    #     video.status = 'processing'
    #     video.save()

    #     # TODO: Implement actual metadata processing
    #     # from yt_workflow.metadata.services.metadata_service import MetadataService
    #     # service = MetadataService()
    #     # metadata = service.fetch_metadata(video_id)
    #     # processed = service.process_metadata(metadata)

    #     # For now, just mark as completed
    #     MetadataResult.objects.create(  # type: ignore
    #         video_id=video_id,
    #         status='success'
    #     )

    #     return {"video_id": video_id, "status": "success", "type": "metadata"}
    # except Exception as e:
    #     return {"video_id": video_id, "status": "failed", "type": "metadata", "error": str(e)}
