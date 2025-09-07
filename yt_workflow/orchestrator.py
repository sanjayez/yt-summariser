"""Main workflow orchestrator for unified video processing pipeline"""

from celery import chain, shared_task

from query_processor.tasks import process_query_request
from yt_workflow.shared.tasks.pipeline import yt_pipeline


@shared_task(bind=True, name="yt_workflow.yt_orchestrator")
def yt_orchestrator(self, search_id: str):  # type: ignore
    """
    Entry point orchestrator that chains query processing with video pipeline and aggregation
    """
    return chain(
        process_query_request.s(search_id),
        yt_pipeline.s(search_id),
        # aggregate_results.s(search_id) future implementation
    ).apply_async()
