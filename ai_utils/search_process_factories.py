"""
Factory functions for easy instantiation of search-to-process integration models.
These factories provide convenient ways to create model instances with sensible defaults.
"""

import uuid
from datetime import datetime, timedelta

from .search_process_models import (
    ErrorType,
    ParallelProcessingConfig,
    ParallelProcessingOrchestration,
    ProcessingError,
    ProcessingPriority,
    RetryConfig,
    RetryTask,
    SearchProcessingStatus,
    SearchProgressRequest,
    SearchProgressResponse,
    SearchProgressUpdate,
    SearchToProcessRequest,
    SearchToProcessResponse,
    StatusCheckRequest,
    StatusCheckResponse,
    VideoProcessingRequest,
    VideoProcessingResult,
    VideoProcessingStatus,
    VideoProcessingStatusAggregation,
    WorkerStatus,
)

# ========== Search Progress Factories ==========


def create_search_progress_request(
    session_id: str,
    query: str,
    max_videos: int = 5,
    priority: ProcessingPriority = ProcessingPriority.NORMAL,
    enable_parallel: bool = True,
    **kwargs,
) -> SearchProgressRequest:
    """
    Create a SearchProgressRequest with sensible defaults.

    Args:
        session_id: Search session identifier
        query: Search query
        max_videos: Maximum number of videos to process
        priority: Processing priority
        enable_parallel: Enable parallel processing
        **kwargs: Additional parameters

    Returns:
        SearchProgressRequest instance
    """
    defaults = {
        "session_id": session_id,
        "query": query,
        "max_videos": max_videos,
        "priority": priority,
        "enable_parallel": enable_parallel,
        "max_parallel_workers": 3,
        "timeout_seconds": 1800,
        "include_metadata": True,
        "include_transcript": True,
        "include_embedding": True,
        "include_summary": True,
        "search_filters": None,
    }
    defaults.update(kwargs)
    return SearchProgressRequest(**defaults)


def create_search_progress_response(
    request_id: str,
    session_id: str,
    query: str,
    status: SearchProcessingStatus = SearchProcessingStatus.INITIALIZING,
    **kwargs,
) -> SearchProgressResponse:
    """
    Create a SearchProgressResponse with sensible defaults.

    Args:
        request_id: Request identifier
        session_id: Session identifier
        query: Search query
        status: Processing status
        **kwargs: Additional parameters

    Returns:
        SearchProgressResponse instance
    """
    defaults = {
        "request_id": request_id,
        "session_id": session_id,
        "status": status,
        "query": query,
        "total_videos": 0,
        "processed_videos": 0,
        "successful_videos": 0,
        "failed_videos": 0,
        "video_urls": [],
        "processing_results": [],
        "error_message": None,
        "error_type": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return SearchProgressResponse(**defaults)


def create_search_progress_update(
    request_id: str,
    status: SearchProcessingStatus,
    progress_percentage: float = 0.0,
    message: str | None = None,
) -> SearchProgressUpdate:
    """
    Create a SearchProgressUpdate.

    Args:
        request_id: Request identifier
        status: Updated status
        progress_percentage: Progress percentage
        message: Optional progress message

    Returns:
        SearchProgressUpdate instance
    """
    return SearchProgressUpdate(
        request_id=request_id,
        status=status,
        progress_percentage=progress_percentage,
        message=message,
        updated_at=datetime.now(),
    )


# ========== Video Processing Factories ==========


def create_video_processing_request(
    video_url: str,
    video_id: str,
    parent_request_id: str,
    priority: ProcessingPriority = ProcessingPriority.NORMAL,
    **kwargs,
) -> VideoProcessingRequest:
    """
    Create a VideoProcessingRequest with sensible defaults.

    Args:
        video_url: YouTube video URL
        video_id: YouTube video ID
        parent_request_id: Parent request identifier
        priority: Processing priority
        **kwargs: Additional parameters

    Returns:
        VideoProcessingRequest instance
    """
    defaults = {
        "video_url": video_url,
        "video_id": video_id,
        "parent_request_id": parent_request_id,
        "priority": priority,
        "include_metadata": True,
        "include_transcript": True,
        "include_embedding": True,
        "include_summary": True,
        "max_retries": 3,
        "retry_delay": 60,
    }
    defaults.update(kwargs)
    return VideoProcessingRequest(**defaults)


def create_video_processing_result(
    video_id: str,
    video_url: str,
    status: VideoProcessingStatus = VideoProcessingStatus.PENDING,
    **kwargs,
) -> VideoProcessingResult:
    """
    Create a VideoProcessingResult with sensible defaults.

    Args:
        video_id: YouTube video ID
        video_url: YouTube video URL
        status: Processing status
        **kwargs: Additional parameters

    Returns:
        VideoProcessingResult instance
    """
    defaults = {
        "video_id": video_id,
        "video_url": video_url,
        "status": status,
        "metadata_status": VideoProcessingStatus.PENDING,
        "transcript_status": VideoProcessingStatus.PENDING,
        "embedding_status": VideoProcessingStatus.PENDING,
        "summary_status": VideoProcessingStatus.PENDING,
        "metadata_available": False,
        "transcript_available": False,
        "embeddings_available": False,
        "summary_available": False,
        "start_time": None,
        "end_time": None,
        "processing_time_seconds": None,
        "error_message": None,
        "error_type": None,
        "retry_count": 0,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return VideoProcessingResult(**defaults)


def create_video_processing_status_aggregation(
    request_id: str, total_videos: int, **kwargs
) -> VideoProcessingStatusAggregation:
    """
    Create a VideoProcessingStatusAggregation with sensible defaults.

    Args:
        request_id: Request identifier
        total_videos: Total number of videos
        **kwargs: Additional parameters

    Returns:
        VideoProcessingStatusAggregation instance
    """
    defaults = {
        "request_id": request_id,
        "total_videos": total_videos,
        "pending_count": total_videos,
        "processing_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "cancelled_count": 0,
        "metadata_completed": 0,
        "transcript_completed": 0,
        "embedding_completed": 0,
        "summary_completed": 0,
        "overall_progress": 0.0,
        "estimated_completion_time": None,
        "average_processing_time": None,
        "throughput_videos_per_minute": None,
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return VideoProcessingStatusAggregation(**defaults)


# ========== Parallel Processing Factories ==========


def create_parallel_processing_config(
    max_workers: int = 3, **kwargs
) -> ParallelProcessingConfig:
    """
    Create a ParallelProcessingConfig with sensible defaults.

    Args:
        max_workers: Maximum number of parallel workers
        **kwargs: Additional parameters

    Returns:
        ParallelProcessingConfig instance
    """
    defaults = {
        "max_workers": max_workers,
        "chunk_size": 5,
        "enable_load_balancing": True,
        "worker_timeout": 900,
        "max_memory_mb": 1024,
        "max_cpu_percent": 80,
        "max_retries": 3,
        "retry_delay": 60,
        "backoff_factor": 2.0,
    }
    defaults.update(kwargs)
    return ParallelProcessingConfig(**defaults)


def create_worker_status(
    worker_id: str, status: str = "initializing", **kwargs
) -> WorkerStatus:
    """
    Create a WorkerStatus with sensible defaults.

    Args:
        worker_id: Worker identifier
        status: Worker status
        **kwargs: Additional parameters

    Returns:
        WorkerStatus instance
    """
    defaults = {
        "worker_id": worker_id,
        "status": status,
        "current_task": None,
        "videos_processed": 0,
        "videos_failed": 0,
        "start_time": datetime.now(),
        "last_activity": datetime.now(),
        "processing_time_seconds": 0.0,
        "memory_usage_mb": None,
        "cpu_usage_percent": None,
    }
    defaults.update(kwargs)
    return WorkerStatus(**defaults)


def create_parallel_processing_orchestration(
    orchestration_id: str,
    request_id: str,
    config: ParallelProcessingConfig | None = None,
    **kwargs,
) -> ParallelProcessingOrchestration:
    """
    Create a ParallelProcessingOrchestration with sensible defaults.

    Args:
        orchestration_id: Orchestration identifier
        request_id: Request identifier
        config: Parallel processing configuration
        **kwargs: Additional parameters

    Returns:
        ParallelProcessingOrchestration instance
    """
    if config is None:
        config = create_parallel_processing_config()

    defaults = {
        "orchestration_id": orchestration_id,
        "request_id": request_id,
        "config": config,
        "workers": [],
        "active_workers": 0,
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "pending_tasks": 0,
        "throughput_tasks_per_minute": None,
        "average_task_time": None,
        "status": "initializing",
        "start_time": None,
        "end_time": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return ParallelProcessingOrchestration(**defaults)


# ========== Error Handling Factories ==========


def create_processing_error(
    request_id: str,
    error_type: ErrorType,
    error_message: str,
    processing_stage: str,
    video_id: str | None = None,
    **kwargs,
) -> ProcessingError:
    """
    Create a ProcessingError with sensible defaults.

    Args:
        request_id: Request identifier
        error_type: Type of error
        error_message: Error message
        processing_stage: Processing stage where error occurred
        video_id: Optional video ID
        **kwargs: Additional parameters

    Returns:
        ProcessingError instance
    """
    defaults = {
        "error_id": str(uuid.uuid4()),
        "request_id": request_id,
        "video_id": video_id,
        "error_type": error_type,
        "error_message": error_message,
        "error_details": None,
        "processing_stage": processing_stage,
        "worker_id": None,
        "retry_count": 0,
        "can_retry": True,
        "next_retry_at": None,
        "occurred_at": datetime.now(),
        "resolved_at": None,
    }
    defaults.update(kwargs)
    return ProcessingError(**defaults)


def create_retry_config(max_retries: int = 3, **kwargs) -> RetryConfig:
    """
    Create a RetryConfig with sensible defaults.

    Args:
        max_retries: Maximum number of retries
        **kwargs: Additional parameters

    Returns:
        RetryConfig instance
    """
    defaults = {
        "max_retries": max_retries,
        "initial_delay": 60,
        "max_delay": 600,
        "backoff_factor": 2.0,
        "retryable_errors": [
            ErrorType.NETWORK_ERROR,
            ErrorType.PROCESSING_TIMEOUT,
            ErrorType.QUOTA_EXCEEDED,
        ],
    }
    defaults.update(kwargs)
    return RetryConfig(**defaults)


def create_retry_task(
    task_id: str,
    request_id: str,
    processing_stage: str,
    original_error: ProcessingError,
    video_id: str | None = None,
    **kwargs,
) -> RetryTask:
    """
    Create a RetryTask with sensible defaults.

    Args:
        task_id: Task identifier
        request_id: Request identifier
        processing_stage: Processing stage to retry
        original_error: Original error that triggered retry
        video_id: Optional video ID
        **kwargs: Additional parameters

    Returns:
        RetryTask instance
    """
    defaults = {
        "task_id": task_id,
        "request_id": request_id,
        "video_id": video_id,
        "processing_stage": processing_stage,
        "retry_count": 0,
        "max_retries": 3,
        "next_retry_at": datetime.now() + timedelta(minutes=1),
        "original_error": original_error,
        "status": "pending",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return RetryTask(**defaults)


# ========== Integration Factories ==========


def create_search_to_process_request(
    session_id: str, query: str, max_videos: int = 5, **kwargs
) -> SearchToProcessRequest:
    """
    Create a SearchToProcessRequest with sensible defaults.

    Args:
        session_id: Search session identifier
        query: Search query
        max_videos: Maximum number of videos to process
        **kwargs: Additional parameters

    Returns:
        SearchToProcessRequest instance
    """
    # Create default nested objects
    processing_config = create_video_processing_request(
        video_url="https://www.youtube.com/watch?v=placeholder",
        video_id="placeholder",
        parent_request_id="placeholder",
    )

    parallel_config = create_parallel_processing_config()
    retry_config = create_retry_config()

    defaults = {
        "session_id": session_id,
        "query": query,
        "max_videos": max_videos,
        "search_filters": None,
        "processing_config": processing_config,
        "parallel_config": parallel_config,
        "retry_config": retry_config,
        "priority": ProcessingPriority.NORMAL,
        "timeout_seconds": 1800,
        "user_ip": None,
        "user_agent": None,
    }
    defaults.update(kwargs)
    return SearchToProcessRequest(**defaults)


def create_search_to_process_response(
    request_id: str,
    session_id: str,
    status: SearchProcessingStatus = SearchProcessingStatus.INITIALIZING,
    **kwargs,
) -> SearchToProcessResponse:
    """
    Create a SearchToProcessResponse with sensible defaults.

    Args:
        request_id: Request identifier
        session_id: Session identifier
        status: Processing status
        **kwargs: Additional parameters

    Returns:
        SearchToProcessResponse instance
    """
    # Create default nested objects
    progress = create_search_progress_response(
        request_id=request_id, session_id=session_id, query="placeholder", status=status
    )

    aggregated_status = create_video_processing_status_aggregation(
        request_id=request_id, total_videos=0
    )

    defaults = {
        "request_id": request_id,
        "session_id": session_id,
        "status": status,
        "progress": progress,
        "video_statuses": [],
        "aggregated_status": aggregated_status,
        "orchestration": None,
        "errors": [],
        "retry_tasks": [],
        "total_processing_time": None,
        "videos_per_minute": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return SearchToProcessResponse(**defaults)


# ========== Status Check Factories ==========


def create_status_check_request(
    request_id: str,
    include_details: bool = True,
    include_errors: bool = True,
    include_performance: bool = False,
) -> StatusCheckRequest:
    """
    Create a StatusCheckRequest with sensible defaults.

    Args:
        request_id: Request identifier
        include_details: Include detailed status information
        include_errors: Include error information
        include_performance: Include performance metrics

    Returns:
        StatusCheckRequest instance
    """
    return StatusCheckRequest(
        request_id=request_id,
        include_details=include_details,
        include_errors=include_errors,
        include_performance=include_performance,
    )


def create_status_check_response(
    request_id: str,
    status: SearchProcessingStatus,
    progress_percentage: float = 0.0,
    total_videos: int = 0,
    completed_videos: int = 0,
    failed_videos: int = 0,
    **kwargs,
) -> StatusCheckResponse:
    """
    Create a StatusCheckResponse with sensible defaults.

    Args:
        request_id: Request identifier
        status: Current status
        progress_percentage: Progress percentage
        total_videos: Total number of videos
        completed_videos: Number of completed videos
        failed_videos: Number of failed videos
        **kwargs: Additional parameters

    Returns:
        StatusCheckResponse instance
    """
    defaults = {
        "request_id": request_id,
        "status": status,
        "progress_percentage": progress_percentage,
        "total_videos": total_videos,
        "completed_videos": completed_videos,
        "failed_videos": failed_videos,
        "details": None,
        "estimated_completion": None,
        "checked_at": datetime.now(),
    }
    defaults.update(kwargs)
    return StatusCheckResponse(**defaults)


# ========== Batch Factory Functions ==========


def create_batch_video_processing_results(
    video_urls: list[str],
    video_ids: list[str],
    status: VideoProcessingStatus = VideoProcessingStatus.PENDING,
) -> list[VideoProcessingResult]:
    """
    Create a batch of VideoProcessingResult instances.

    Args:
        video_urls: List of video URLs
        video_ids: List of video IDs
        status: Initial status for all videos

    Returns:
        List of VideoProcessingResult instances
    """
    if len(video_urls) != len(video_ids):
        raise ValueError("video_urls and video_ids must have the same length")

    results = []
    for url, video_id in zip(video_urls, video_ids, strict=False):
        result = create_video_processing_result(
            video_id=video_id, video_url=url, status=status
        )
        results.append(result)

    return results


def create_batch_workers(
    worker_count: int,
    worker_id_prefix: str = "worker",
) -> list[WorkerStatus]:
    """
    Create a batch of WorkerStatus instances.

    Args:
        worker_count: Number of workers to create
        worker_id_prefix: Prefix for worker IDs

    Returns:
        List of WorkerStatus instances
    """
    workers = []
    for i in range(worker_count):
        worker_id = f"{worker_id_prefix}_{i + 1}"
        worker = create_worker_status(worker_id=worker_id)
        workers.append(worker)

    return workers


# ========== Utility Functions ==========


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def generate_orchestration_id() -> str:
    """Generate a unique orchestration ID."""
    return str(uuid.uuid4())


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())


def calculate_progress_percentage(completed: int, total: int) -> float:
    """Calculate progress percentage."""
    if total == 0:
        return 0.0
    return min(100.0, (completed / total) * 100.0)


def estimate_completion_time(
    completed: int,
    total: int,
    start_time: datetime,
    current_time: datetime | None = None,
) -> datetime | None:
    """Estimate completion time based on current progress."""
    if current_time is None:
        current_time = datetime.now()

    if completed == 0 or total == 0:
        return None

    elapsed = (current_time - start_time).total_seconds()
    if elapsed <= 0:
        return None

    rate = completed / elapsed  # items per second
    remaining = total - completed

    if rate <= 0:
        return None

    estimated_seconds = remaining / rate
    return current_time + timedelta(seconds=estimated_seconds)


def calculate_throughput(
    completed: int, start_time: datetime, current_time: datetime | None = None
) -> float | None:
    """Calculate throughput in items per minute."""
    if current_time is None:
        current_time = datetime.now()

    if completed == 0:
        return None

    elapsed = (current_time - start_time).total_seconds()
    if elapsed <= 0:
        return None

    return (completed / elapsed) * 60.0  # items per minute
