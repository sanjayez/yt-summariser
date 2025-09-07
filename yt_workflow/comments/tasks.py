from celery import shared_task


@shared_task(bind=True, name="yt_workflow.process_comments")
def process_comments(self, video_id: str):  # type: ignore
    """Process comments for a single video"""

    # try:
    # TODO: Implement actual comments processing
    # from yt_workflow.comments.services.comments_service import CommentsService
    # service = CommentsService()
    # comments = service.fetch_comments(video_id)
    # insights = service.process_comments(comments)

    # For now, just mark as completed
    #     CommentsResult.objects.create(  # type: ignore
    #         video_id=video_id,
    #         status='success'
    #     )

    #     return {"video_id": video_id, "status": "success", "type": "comments"}
    # except Exception as e:
    #     return {"video_id": video_id, "status": "failed", "type": "comments", "error": str(e)}
