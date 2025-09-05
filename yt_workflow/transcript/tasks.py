from celery import shared_task


@shared_task(bind=True, name="yt_workflow.transcript.fetch")
def fetch_transcript_task(self, execution_id: str):
    """Fetch transcript for video"""
    # TODO: Implement transcript task


@shared_task(bind=True, name="yt_workflow.transcript.process")
def process_transcript_task(self, execution_id: str):
    """Process transcript with LLM"""
    # TODO: Implement transcript processing task
