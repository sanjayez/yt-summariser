from celery import shared_task


@shared_task(bind=True, name="yt_workflow.comments.fetch")
def fetch_comments_task(self, execution_id: str):
    """Fetch comments for video"""
    # TODO: Implement comments task


@shared_task(bind=True, name="yt_workflow.comments.process")
def process_comments_task(self, execution_id: str):
    """Process comments with LLM"""
    # TODO: Implement comments processing task
