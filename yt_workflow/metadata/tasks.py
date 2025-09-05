from celery import shared_task


@shared_task(bind=True, name="yt_workflow.metadata.fetch")
def fetch_metadata_task(self, execution_id: str):
    """Fetch metadata for video"""
    # TODO: Implement metadata task


@shared_task(bind=True, name="yt_workflow.metadata.process")
def process_metadata_task(self, execution_id: str):
    """Process metadata with LLM"""
    # TODO: Implement metadata processing task
