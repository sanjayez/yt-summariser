"""Main workflow orchestrator for parallel video processing"""

from celery import chain, group, shared_task

from yt_workflow.shared.models import WorkflowExecution
from yt_workflow.shared.utils import extract_video_id_from_url


@shared_task(bind=True, name="yt_workflow.orchestrate")
def orchestrate_video_processing(self, video_url: str) -> dict:  # type: ignore
    """Main orchestrator - launches parallel processing workflow"""

    # Extract video ID and create execution record
    video_id = extract_video_id_from_url(video_url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {video_url}")

    execution = WorkflowExecution.objects.create(  # type: ignore
        video_url=video_url, video_id=video_id, status="processing"
    )

    # Create parallel group for data fetching
    parallel_fetch = group(
        # Import tasks dynamically to avoid circular imports
        fetch_metadata_task.s(str(execution.execution_id)),
        fetch_transcript_task.s(str(execution.execution_id)),
        fetch_comments_task.s(str(execution.execution_id)),
    )

    # Chain parallel fetching with final aggregation
    workflow = chain(parallel_fetch, aggregate_results.s(str(execution.execution_id)))

    # Execute workflow
    result = workflow.apply_async()

    return {
        "execution_id": str(execution.execution_id),
        "workflow_task_id": result.id,
        "status": "processing",
    }


@shared_task(bind=True, name="yt_workflow.aggregate")
def aggregate_results(self, fetch_results, execution_id: str):
    """Aggregate results from parallel processing"""
    # TODO: Implement result aggregation


# Import tasks for group usage (avoid circular imports)
def _import_tasks():
    from yt_workflow.comments.tasks import fetch_comments_task
    from yt_workflow.metadata.tasks import fetch_metadata_task
    from yt_workflow.transcript.tasks import fetch_transcript_task

    return fetch_metadata_task, fetch_transcript_task, fetch_comments_task


fetch_metadata_task, fetch_transcript_task, fetch_comments_task = _import_tasks()
