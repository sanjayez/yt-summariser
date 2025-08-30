import logging
import os
import time

from celery import chain, group, shared_task
from celery.exceptions import SoftTimeLimitExceeded

from api.models import URLRequestTable
from temp_file_logger import append_line

from ..config import YOUTUBE_CONFIG
from ..utils import handle_dead_letter_task
from ..validators import validate_youtube_url
from .content_analysis_finalization import content_analysis_finalization
from .content_analysis_preliminary import content_analysis_preliminary
from .content_classifier import classify_and_exclude_video_llm
from .embedding import embed_video_content
from .metadata import extract_video_metadata
from .status import update_overall_status
from .summary import generate_video_summary
from .transcript import extract_video_transcript

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="video_processor.log_stage_duration", ignore_result=True)
def log_stage_duration(self, parent_result, start_ts, stage_name, url_request_id):
    """
    Lightweight timing logger for pipeline stages. Triggered via Celery link callback.
    """
    try:
        elapsed = max(0.0, time.time() - float(start_ts))
        logger.info(
            f"⏱️ Stage timing: {stage_name} for request {url_request_id} took {elapsed:.2f}s"
        )
        # Also append to a simple file for later analysis
        append_line(
            file_path=os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
            message=f"stage={stage_name} request_id={url_request_id} elapsed_s={elapsed:.2f}",
        )
    except Exception as e:
        logger.warning(f"Failed to log timing for {stage_name} ({url_request_id}): {e}")
    # Do not return parent_result; this is a fire-and-forget logger
    return None


@shared_task(bind=True, name="video_processor.log_pipeline_total", ignore_result=True)
def log_pipeline_total(self, url_request_id, pipeline_start_ts):
    """Log total end-to-end pipeline time for a request."""
    try:
        total_elapsed = max(0.0, time.time() - float(pipeline_start_ts))
        logger.info(
            f"⏱️ Pipeline total: request {url_request_id} took {total_elapsed:.2f}s end-to-end"
        )
        append_line(
            file_path=os.getenv("TIMING_LOG_FILE", "logs/stage_timings.log"),
            message=f"pipeline_total request_id={url_request_id} elapsed_s={total_elapsed:.2f}",
        )
    except Exception as e:
        logger.warning(f"Failed to log pipeline total for {url_request_id}: {e}")
    return None


@shared_task(
    bind=True,
    name="video_processor.process_youtube_video",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["workflow_soft_limit"],
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["workflow_hard_limit"],
    autoretry_for=(Exception,),
    retry_backoff=YOUTUBE_CONFIG["RETRY_CONFIG"]["metadata"]["backoff"],
    retry_jitter=YOUTUBE_CONFIG["RETRY_CONFIG"]["metadata"]["jitter"],
)
def process_youtube_video(self, url_request_id):
    """
    Process a single YouTube video through the complete pipeline.

    Executes the full workflow: metadata → transcript → [content_analysis_preliminary + summary + classification] → embedding → content_analysis_finalization → status update
    Features parallel processing of content analysis, summary, and classification tasks

    Args:
        url_request_id (str): UUID of the URLRequestTable to process

    Returns:
        str: Success message with request ID

    Raises:
        Exception: If workflow initiation fails
    """
    url_request = None

    try:
        # Validate input and get URL request with related data to avoid N+1 queries
        url_request = URLRequestTable.objects.select_related(
            "video_metadata", "video_metadata__video_transcript"
        ).get(request_id=url_request_id)
        validate_youtube_url(url_request.url)

        # Store task ID for progress tracking
        url_request.celery_task_id = self.request.id
        url_request.save()

        logger.info(f"Starting video processing pipeline for request {url_request_id}")

        # Execute workflow chain: each task receives previous result as first argument
        # New pipeline: metadata → transcript → [content_analysis_preliminary + summary + classification] → embedding → content_analysis_finalization → status
        logger.info(f"Constructing workflow chain for request {url_request_id}")

        # Add timing links to each stage (non-blocking callbacks)
        meta_start = time.time()
        meta_sig = extract_video_metadata.s(url_request_id).set(
            link=log_stage_duration.s(
                meta_start, "extract_video_metadata", url_request_id
            )
        )

        transcript_start = time.time()
        transcript_sig = extract_video_transcript.s(url_request_id).set(
            link=log_stage_duration.s(
                transcript_start, "extract_video_transcript", url_request_id
            )
        )

        # Parallel group: add timing links per child task
        ca_start = time.time()
        summary_start = time.time()
        classify_start = time.time()
        parallel_group = group(
            content_analysis_preliminary.s(url_request_id).set(
                link=log_stage_duration.s(
                    ca_start, "content_analysis_preliminary", url_request_id
                )
            ),
            generate_video_summary.s(url_request_id).set(
                link=log_stage_duration.s(
                    summary_start, "generate_video_summary", url_request_id
                )
            ),
            classify_and_exclude_video_llm.s(url_request_id).set(
                link=log_stage_duration.s(
                    classify_start, "classify_and_exclude_video_llm", url_request_id
                )
            ),
        )

        embed_start = time.time()
        embed_sig = embed_video_content.s(url_request_id).set(
            link=log_stage_duration.s(
                embed_start, "embed_video_content", url_request_id
            )
        )

        finalize_start = time.time()
        finalize_sig = content_analysis_finalization.s(url_request_id).set(
            link=log_stage_duration.s(
                finalize_start, "content_analysis_finalization", url_request_id
            )
        )

        status_start = time.time()
        status_sig = update_overall_status.s(url_request_id).set(
            link=log_stage_duration.s(
                status_start, "update_overall_status", url_request_id
            )
        )

        pipeline_start = time.time()
        workflow = chain(
            meta_sig,
            transcript_sig,
            parallel_group,
            embed_sig,
            finalize_sig,
            status_sig.set(link=log_pipeline_total.s(url_request_id, pipeline_start)),
        )

        logger.info(
            f"Workflow chain constructed with parallel content analysis processing for request {url_request_id}"
        )

        # Capture workflow result to enable result tracking and eliminate fire-and-forget behavior
        logger.info(f"Starting workflow execution for request {url_request_id}")
        result = workflow.apply_async()

        # Store chain task ID for progress tracking and result retrieval
        url_request.chain_task_id = result.id
        url_request.save(update_fields=["chain_task_id"])
        logger.info(
            f"Workflow initiated with chain_task_id {result.id} for request {url_request_id}"
        )

        logger.info(
            f"Successfully initiated two-phase content analysis pipeline for request {url_request_id}"
        )
        return f"Initiated complete processing pipeline with parallel content analysis for request {url_request_id}"

    except SoftTimeLimitExceeded:
        # Workflow orchestration is approaching timeout - this should not happen often
        logger.warning(
            f"Workflow orchestration soft timeout reached for request {url_request_id}"
        )

        try:
            # Mark the URL request as failed due to orchestration timeout
            if url_request:
                url_request.status = "failed"
                url_request.failure_reason = "technical_failure"
                url_request.save()
                logger.error(
                    f"Marked workflow as failed due to orchestration timeout: {url_request_id}"
                )

        except Exception as cleanup_error:
            logger.error(
                f"Failed to update workflow status during timeout cleanup: {cleanup_error}"
            )

        # Re-raise to mark task as failed
        raise Exception(f"Workflow orchestration timeout for request {url_request_id}")

    except Exception as e:
        logger.error(
            f"Failed to initiate processing pipeline for {url_request_id}: {e}"
        )
        handle_dead_letter_task(
            "process_youtube_video", self.request.id, [url_request_id], {}, e
        )
        raise


@shared_task(
    bind=True,
    name="video_processor.process_parallel_videos",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["parallel_soft_limit"],
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["parallel_hard_limit"],
    autoretry_for=(Exception,),
    retry_backoff=YOUTUBE_CONFIG["RETRY_CONFIG"]["parallel"]["backoff"],
    retry_jitter=YOUTUBE_CONFIG["RETRY_CONFIG"]["parallel"]["jitter"],
)
def process_parallel_videos(self, url_request_ids):
    """
    Process multiple videos in parallel.

    Each video goes through the complete pipeline independently and concurrently.

    Args:
        url_request_ids (list): List of URLRequestTable UUIDs to process

    Returns:
        dict: Processing result with parallel job information
    """
    logger.info(f"Starting parallel processing for {len(url_request_ids)} videos")

    try:
        # Validate all URL requests exist
        existing_requests = URLRequestTable.objects.filter(
            request_id__in=url_request_ids
        )
        if existing_requests.count() != len(url_request_ids):
            missing_ids = set(url_request_ids) - set(
                existing_requests.values_list("request_id", flat=True)
            )
            logger.error(f"Missing URLRequestTable entries: {missing_ids}")
            return {
                "status": "failed",
                "error": "Missing URL request entries",
                "missing_ids": list(missing_ids),
            }

        # Create parallel group of individual video processing tasks
        video_processing_group = group(
            process_youtube_video.s(url_request_id)
            for url_request_id in url_request_ids
        )

        # Execute parallel processing
        result = video_processing_group.apply_async()

        logger.info(
            f"Launched parallel processing for {len(url_request_ids)} videos, group ID: {result.id}"
        )

        return {
            "status": "processing",
            "group_id": result.id,
            "total_videos": len(url_request_ids),
            "url_request_ids": url_request_ids,
            "processing_type": "parallel",
        }

    except SoftTimeLimitExceeded:
        # Parallel orchestration is approaching timeout
        logger.warning(
            f"Parallel orchestration soft timeout reached for {len(url_request_ids)} videos"
        )

        try:
            # Mark all URL requests as failed due to orchestration timeout
            URLRequestTable.objects.filter(request_id__in=url_request_ids).update(
                status="failed", failure_reason="technical_failure"
            )
            logger.error(
                f"Marked {len(url_request_ids)} parallel videos as failed due to orchestration timeout"
            )

        except Exception as cleanup_error:
            logger.error(
                f"Failed to update parallel workflow status during timeout cleanup: {cleanup_error}"
            )

        # Re-raise to mark task as failed
        raise Exception(
            f"Parallel workflow orchestration timeout for {len(url_request_ids)} videos"
        )

    except Exception as e:
        logger.error(f"Failed to start parallel processing: {e}")
        return {
            "status": "failed",
            "error": "Failed to start parallel processing",
            "details": str(e),
        }
