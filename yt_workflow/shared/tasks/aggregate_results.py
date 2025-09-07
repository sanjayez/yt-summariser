from celery import shared_task


@shared_task(bind=True, name="yt_workflow.aggregate_results")
def aggregate_results(self, results, search_id: str):
    """Aggregate results from parallel processing and update YTInsightRun status"""
    # try:
    #     # Get the YTInsightRun via search_id
    #     query_request = QueryRequest.objects.get(search_id=search_id)  # type: ignore
    #     insight_run = YTInsightRun.objects.get(query_request=query_request)  # type: ignore

    #     # Check all video statuses
    #     all_videos = insight_run.video_ids
    #     completed_videos = VideoTable.objects.filter(  # type: ignore
    #         video_id__in=all_videos,
    #         status='success'
    #     ).count()

    #     failed_videos = VideoTable.objects.filter(  # type: ignore
    #         video_id__in=all_videos,
    #         status='failed'
    #     ).count()

    #     # Update insight run status
    #     if completed_videos == len(all_videos):
    #         insight_run.status = 'success'
    #     elif failed_videos == len(all_videos):
    #         insight_run.status = 'failed'
    #     elif completed_videos > 0:
    #         insight_run.status = 'success'  # Partial success still counts as success
    #     else:
    #         insight_run.status = 'pending'  # Still processing

    #     insight_run.save()

    #     # Update individual video statuses based on results (bulk operation)
    #     completed_video_ids = []
    #     for video_results in results:
    #         if isinstance(video_results, list):
    #             for result in video_results:
    #                 if 'video_id' in result and result['status'] == 'success':
    #                     completed_video_ids.append(result['video_id'])

    #     # Bulk update all completed videos in one query
    #     if completed_video_ids:
    #         VideoTable.objects.filter(  # type: ignore
    #             video_id__in=completed_video_ids
    #         ).update(status='success')

    #     return {"search_id": search_id, "status": insight_run.status}
    # except Exception as e:
    #     return {"search_id": search_id, "status": "error", "error": str(e)}
