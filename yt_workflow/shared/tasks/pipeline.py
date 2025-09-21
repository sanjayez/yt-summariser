from celery import chord, group, shared_task
from django.db import transaction

from query_processor.models import QueryRequest

# from yt_workflow.comments.tasks import process_comments
# from yt_workflow.metadata.tasks import process_metadata
from yt_workflow.shared.models import StatusChoices, VideoTable, YTInsightRun
from yt_workflow.shared.utils import extract_video_id_from_url
from yt_workflow.transcript.tasks import process_transcript


@shared_task(bind=True, name="yt_workflow.yt_pipeline")
def yt_pipeline(self, query_result, search_id: str):  # type: ignore
    """
    Step 1: Read processed data from QueryRequest and create YTInsightRun
    Step 2: Check for existing videos
    Step 3: Launch parallel processing for new videos
    Step 4: Update status
    """

    # Step 1: Read the processed QueryRequest
    query_request = QueryRequest.objects.get(search_id=search_id)  # type: ignore

    # Always create YTInsightRun entry to track the request
    with transaction.atomic():
        insight_run = YTInsightRun.objects.create(  # type: ignore
            query_request=query_request,
            video_ids=[],  # Will be populated if successful
            status=StatusChoices.PENDING,
        )

    # Check if query processing was successful
    if query_request.status != "success" or not query_request.video_urls:
        insight_run.status = StatusChoices.FAILED
        insight_run.save()
        return  # Database tracks the failure, no return value needed

    # Extract video IDs from URLs stored in QueryRequest
    video_ids = []
    for url in query_request.video_urls:
        video_id = extract_video_id_from_url(url)
        if video_id:
            video_ids.append(video_id)

    # Update insight run with video IDs
    insight_run.video_ids = video_ids
    insight_run.save()

    # Step 3: Check for existing videos in VideoTable
    existing_videos = set(
        VideoTable.objects.filter(  # type: ignore
            video_id__in=video_ids, status=StatusChoices.SUCCESS
        ).values_list("video_id", flat=True)
    )

    new_video_ids = [vid for vid in video_ids if vid not in existing_videos]

    if not new_video_ids:
        # All videos already processed
        insight_run.status = StatusChoices.SUCCESS
        insight_run.save()
        return  # Database tracks completion, no return value needed

    # Step 4: Create VideoTable entries for new videos
    for video_id in new_video_ids:
        VideoTable.objects.get_or_create(  # type: ignore
            video_id=video_id, defaults={"status": StatusChoices.PENDING}
        )

    # Step 5: Launch parallel processing for new videos
    all_tasks = []
    for video_id in new_video_ids:
        all_tasks.extend(
            [
                # process_metadata.s(video_id),
                process_transcript.s(video_id),
                # process_comments.s(video_id),
            ]
        )

    # Execute all video processing in parallel with chord callback
    if all_tasks:
        from yt_workflow.shared.tasks.aggregate_results import aggregate_results

        extract_and_process = group(*all_tasks)
        return chord(extract_and_process)(aggregate_results.s(search_id))
