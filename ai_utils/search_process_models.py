"""
Pydantic models for search-to-process integration.
These models handle data validation, serialization, and type safety for
search progress tracking, video processing status, and parallel processing orchestration.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ========== Enums ==========


class SearchProcessingStatus(str, Enum):
    """Status of search-to-process operations"""

    INITIALIZING = "initializing"
    SEARCHING = "searching"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_FAILED = "search_failed"
    PROCESSING_VIDEOS = "processing_videos"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VideoProcessingStatus(str, Enum):
    """Status of individual video processing"""

    PENDING = "pending"
    METADATA_PROCESSING = "metadata_processing"
    TRANSCRIPT_PROCESSING = "transcript_processing"
    EMBEDDING_PROCESSING = "embedding_processing"
    SUMMARY_PROCESSING = "summary_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPriority(str, Enum):
    """Priority levels for processing tasks"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ErrorType(str, Enum):
    """Types of errors that can occur during processing"""

    SEARCH_ERROR = "search_error"
    VIDEO_METADATA_ERROR = "video_metadata_error"
    TRANSCRIPT_ERROR = "transcript_error"
    EMBEDDING_ERROR = "embedding_error"
    SUMMARY_ERROR = "summary_error"
    NETWORK_ERROR = "network_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_VIDEO = "invalid_video"
    PROCESSING_TIMEOUT = "processing_timeout"
    UNKNOWN_ERROR = "unknown_error"


# ========== Search Progress Tracking Models ==========


class SearchProgressRequest(BaseModel):
    """Request model for initiating search-to-process operations"""

    session_id: str = Field(..., description="Search session identifier")
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_videos: int = Field(
        default=5, ge=1, le=20, description="Maximum number of videos to process"
    )
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL, description="Processing priority"
    )
    enable_parallel: bool = Field(
        default=True, description="Enable parallel video processing"
    )
    max_parallel_workers: int = Field(
        default=3, ge=1, le=10, description="Maximum parallel workers"
    )
    timeout_seconds: int = Field(
        default=1800, ge=60, le=3600, description="Overall timeout in seconds"
    )

    # Processing configuration
    include_metadata: bool = Field(
        default=True, description="Include video metadata processing"
    )
    include_transcript: bool = Field(
        default=True, description="Include transcript processing"
    )
    include_embedding: bool = Field(
        default=True, description="Include embedding processing"
    )
    include_summary: bool = Field(
        default=True, description="Include summary generation"
    )

    # Search configuration
    search_filters: dict[str, Any] | None = Field(
        default=None, description="Search filters"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchProgressResponse(BaseModel):
    """Response model for search-to-process operations"""

    request_id: str = Field(..., description="Unique request identifier")
    session_id: str = Field(..., description="Search session identifier")
    status: SearchProcessingStatus = Field(..., description="Current processing status")
    query: str = Field(..., description="Original search query")

    # Progress tracking
    total_videos: int = Field(default=0, description="Total videos found")
    processed_videos: int = Field(default=0, description="Number of videos processed")
    successful_videos: int = Field(
        default=0, description="Number of successfully processed videos"
    )
    failed_videos: int = Field(default=0, description="Number of failed videos")

    # Timing information
    search_start_time: datetime | None = Field(
        default=None, description="Search start timestamp"
    )
    search_end_time: datetime | None = Field(
        default=None, description="Search end timestamp"
    )
    processing_start_time: datetime | None = Field(
        default=None, description="Processing start timestamp"
    )
    processing_end_time: datetime | None = Field(
        default=None, description="Processing end timestamp"
    )

    # Results
    video_urls: list[str] = Field(
        default_factory=list, description="List of video URLs"
    )
    processing_results: list["VideoProcessingResult"] = Field(
        default_factory=list, description="Video processing results"
    )

    # Error handling
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    error_type: ErrorType | None = Field(default=None, description="Type of error")

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


class SearchProgressUpdate(BaseModel):
    """Model for search progress updates"""

    request_id: str = Field(..., description="Request identifier")
    status: SearchProcessingStatus = Field(..., description="Updated status")
    progress_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    message: str | None = Field(default=None, description="Progress message")
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Update timestamp"
    )


# ========== Video Processing Status Models ==========


class VideoProcessingRequest(BaseModel):
    """Request model for individual video processing"""

    video_url: str = Field(..., description="YouTube video URL")
    video_id: str = Field(..., description="YouTube video ID")
    parent_request_id: str = Field(..., description="Parent search request ID")
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL, description="Processing priority"
    )

    # Processing configuration
    include_metadata: bool = Field(
        default=True, description="Include metadata processing"
    )
    include_transcript: bool = Field(
        default=True, description="Include transcript processing"
    )
    include_embedding: bool = Field(
        default=True, description="Include embedding processing"
    )
    include_summary: bool = Field(
        default=True, description="Include summary generation"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    retry_delay: int = Field(
        default=60, ge=0, le=300, description="Retry delay in seconds"
    )

    @field_validator("video_url")
    @classmethod
    def validate_video_url(cls, v):
        if not any(domain in v.lower() for domain in ["youtube.com", "youtu.be"]):
            raise ValueError("Must be a valid YouTube URL")
        return v


class VideoProcessingResult(BaseModel):
    """Result model for individual video processing"""

    video_id: str = Field(..., description="YouTube video ID")
    video_url: str = Field(..., description="YouTube video URL")
    status: VideoProcessingStatus = Field(..., description="Processing status")

    # Processing stages
    metadata_status: VideoProcessingStatus = Field(
        default=VideoProcessingStatus.PENDING, description="Metadata processing status"
    )
    transcript_status: VideoProcessingStatus = Field(
        default=VideoProcessingStatus.PENDING,
        description="Transcript processing status",
    )
    embedding_status: VideoProcessingStatus = Field(
        default=VideoProcessingStatus.PENDING, description="Embedding processing status"
    )
    summary_status: VideoProcessingStatus = Field(
        default=VideoProcessingStatus.PENDING, description="Summary processing status"
    )

    # Results
    metadata_available: bool = Field(
        default=False, description="Whether metadata is available"
    )
    transcript_available: bool = Field(
        default=False, description="Whether transcript is available"
    )
    embeddings_available: bool = Field(
        default=False, description="Whether embeddings are available"
    )
    summary_available: bool = Field(
        default=False, description="Whether summary is available"
    )

    # Timing information
    start_time: datetime | None = Field(
        default=None, description="Processing start time"
    )
    end_time: datetime | None = Field(default=None, description="Processing end time")
    processing_time_seconds: float | None = Field(
        default=None, description="Total processing time"
    )

    # Error handling
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    error_type: ErrorType | None = Field(default=None, description="Type of error")
    retry_count: int = Field(default=0, description="Number of retries attempted")

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


class VideoProcessingStatusAggregation(BaseModel):
    """Aggregated status for multiple video processing operations"""

    request_id: str = Field(..., description="Request identifier")
    total_videos: int = Field(..., description="Total number of videos")

    # Status counts
    pending_count: int = Field(default=0, description="Number of pending videos")
    processing_count: int = Field(default=0, description="Number of processing videos")
    completed_count: int = Field(default=0, description="Number of completed videos")
    failed_count: int = Field(default=0, description="Number of failed videos")
    cancelled_count: int = Field(default=0, description="Number of cancelled videos")

    # Processing stages
    metadata_completed: int = Field(
        default=0, description="Videos with metadata completed"
    )
    transcript_completed: int = Field(
        default=0, description="Videos with transcript completed"
    )
    embedding_completed: int = Field(
        default=0, description="Videos with embeddings completed"
    )
    summary_completed: int = Field(
        default=0, description="Videos with summary completed"
    )

    # Progress metrics
    overall_progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall progress percentage"
    )
    estimated_completion_time: datetime | None = Field(
        default=None, description="Estimated completion time"
    )

    # Performance metrics
    average_processing_time: float | None = Field(
        default=None, description="Average processing time per video"
    )
    throughput_videos_per_minute: float | None = Field(
        default=None, description="Processing throughput"
    )

    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


# ========== Parallel Processing Models ==========


class ParallelProcessingConfig(BaseModel):
    """Configuration for parallel processing"""

    max_workers: int = Field(
        default=3, ge=1, le=10, description="Maximum parallel workers"
    )
    chunk_size: int = Field(
        default=5, ge=1, le=20, description="Number of videos per chunk"
    )
    enable_load_balancing: bool = Field(
        default=True, description="Enable load balancing"
    )
    worker_timeout: int = Field(
        default=900, ge=60, le=1800, description="Worker timeout in seconds"
    )

    # Resource limits
    max_memory_mb: int = Field(
        default=1024, ge=512, le=4096, description="Maximum memory per worker"
    )
    max_cpu_percent: int = Field(
        default=80, ge=10, le=100, description="Maximum CPU usage percent"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    retry_delay: int = Field(
        default=60, ge=0, le=300, description="Retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )


class WorkerStatus(BaseModel):
    """Status of a parallel processing worker"""

    worker_id: str = Field(..., description="Worker identifier")
    status: str = Field(..., description="Worker status")
    current_task: str | None = Field(
        default=None, description="Current task being processed"
    )
    videos_processed: int = Field(default=0, description="Number of videos processed")
    videos_failed: int = Field(default=0, description="Number of videos failed")

    # Performance metrics
    start_time: datetime = Field(
        default_factory=datetime.now, description="Worker start time"
    )
    last_activity: datetime = Field(
        default_factory=datetime.now, description="Last activity timestamp"
    )
    processing_time_seconds: float = Field(
        default=0.0, description="Total processing time"
    )

    # Resource usage
    memory_usage_mb: float | None = Field(
        default=None, description="Memory usage in MB"
    )
    cpu_usage_percent: float | None = Field(
        default=None, description="CPU usage percentage"
    )


class ParallelProcessingOrchestration(BaseModel):
    """Orchestration model for parallel processing operations"""

    orchestration_id: str = Field(..., description="Orchestration identifier")
    request_id: str = Field(..., description="Parent request identifier")
    config: ParallelProcessingConfig = Field(
        ..., description="Processing configuration"
    )

    # Worker management
    workers: list[WorkerStatus] = Field(
        default_factory=list, description="Worker status list"
    )
    active_workers: int = Field(default=0, description="Number of active workers")

    # Task distribution
    total_tasks: int = Field(default=0, description="Total number of tasks")
    completed_tasks: int = Field(default=0, description="Number of completed tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    pending_tasks: int = Field(default=0, description="Number of pending tasks")

    # Performance metrics
    throughput_tasks_per_minute: float | None = Field(
        default=None, description="Task throughput"
    )
    average_task_time: float | None = Field(
        default=None, description="Average task processing time"
    )

    # Status and timing
    status: str = Field(default="initializing", description="Orchestration status")
    start_time: datetime | None = Field(default=None, description="Start time")
    end_time: datetime | None = Field(default=None, description="End time")

    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


# ========== Error Handling and Retry Models ==========


class ProcessingError(BaseModel):
    """Model for processing errors"""

    error_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Error identifier"
    )
    request_id: str = Field(..., description="Request identifier")
    video_id: str | None = Field(default=None, description="Video ID if applicable")
    error_type: ErrorType = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    error_details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )

    # Context information
    processing_stage: str = Field(
        ..., description="Processing stage where error occurred"
    )
    worker_id: str | None = Field(default=None, description="Worker ID if applicable")

    # Retry information
    retry_count: int = Field(default=0, description="Number of retries attempted")
    can_retry: bool = Field(default=True, description="Whether error is retryable")
    next_retry_at: datetime | None = Field(
        default=None, description="Next retry timestamp"
    )

    # Metadata
    occurred_at: datetime = Field(
        default_factory=datetime.now, description="Error occurrence timestamp"
    )
    resolved_at: datetime | None = Field(
        default=None, description="Error resolution timestamp"
    )


class RetryConfig(BaseModel):
    """Configuration for retry logic"""

    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    initial_delay: int = Field(
        default=60, ge=0, le=300, description="Initial retry delay in seconds"
    )
    max_delay: int = Field(
        default=600, ge=60, le=3600, description="Maximum retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )

    # Error-specific retry settings
    retryable_errors: list[ErrorType] = Field(
        default_factory=lambda: [
            ErrorType.NETWORK_ERROR,
            ErrorType.PROCESSING_TIMEOUT,
            ErrorType.QUOTA_EXCEEDED,
        ],
        description="List of retryable error types",
    )


class RetryTask(BaseModel):
    """Model for retry tasks"""

    task_id: str = Field(..., description="Task identifier")
    request_id: str = Field(..., description="Request identifier")
    video_id: str | None = Field(default=None, description="Video ID if applicable")
    processing_stage: str = Field(..., description="Processing stage to retry")

    # Retry information
    retry_count: int = Field(default=0, description="Current retry count")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    next_retry_at: datetime = Field(..., description="Next retry timestamp")

    # Original error
    original_error: ProcessingError = Field(
        ..., description="Original error that triggered retry"
    )

    # Status
    status: str = Field(default="pending", description="Retry task status")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


# ========== Integration Models ==========


class SearchToProcessRequest(BaseModel):
    """Main request model for search-to-process integration"""

    session_id: str = Field(..., description="Search session identifier")
    query: str = Field(..., description="Search query")

    # Search configuration
    max_videos: int = Field(
        default=5, ge=1, le=20, description="Maximum videos to process"
    )
    search_filters: dict[str, Any] | None = Field(
        default=None, description="Search filters"
    )

    # Processing configuration
    processing_config: VideoProcessingRequest = Field(
        ..., description="Video processing configuration"
    )
    parallel_config: ParallelProcessingConfig = Field(
        default_factory=ParallelProcessingConfig,
        description="Parallel processing configuration",
    )
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )

    # Priority and timing
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL, description="Processing priority"
    )
    timeout_seconds: int = Field(
        default=1800, ge=60, le=3600, description="Overall timeout"
    )

    # Metadata
    user_ip: str | None = Field(default=None, description="User IP address")
    user_agent: str | None = Field(default=None, description="User agent")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchToProcessResponse(BaseModel):
    """Main response model for search-to-process integration"""

    request_id: str = Field(..., description="Request identifier")
    session_id: str = Field(..., description="Session identifier")
    status: SearchProcessingStatus = Field(..., description="Overall status")

    # Progress information
    progress: SearchProgressResponse = Field(..., description="Search progress")
    video_statuses: list[VideoProcessingResult] = Field(
        default_factory=list, description="Video processing results"
    )
    aggregated_status: VideoProcessingStatusAggregation = Field(
        ..., description="Aggregated status"
    )

    # Parallel processing information
    orchestration: ParallelProcessingOrchestration | None = Field(
        default=None, description="Parallel processing orchestration"
    )

    # Error handling
    errors: list[ProcessingError] = Field(
        default_factory=list, description="Processing errors"
    )
    retry_tasks: list[RetryTask] = Field(
        default_factory=list, description="Retry tasks"
    )

    # Performance metrics
    total_processing_time: float | None = Field(
        default=None, description="Total processing time in seconds"
    )
    videos_per_minute: float | None = Field(
        default=None, description="Processing throughput"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


# ========== Status Check Models ==========


class StatusCheckRequest(BaseModel):
    """Request model for status checks"""

    request_id: str = Field(..., description="Request identifier")
    include_details: bool = Field(
        default=True, description="Include detailed status information"
    )
    include_errors: bool = Field(default=True, description="Include error information")
    include_performance: bool = Field(
        default=False, description="Include performance metrics"
    )


class StatusCheckResponse(BaseModel):
    """Response model for status checks"""

    request_id: str = Field(..., description="Request identifier")
    status: SearchProcessingStatus = Field(..., description="Current status")
    progress_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )

    # Quick status overview
    total_videos: int = Field(default=0, description="Total videos")
    completed_videos: int = Field(default=0, description="Completed videos")
    failed_videos: int = Field(default=0, description="Failed videos")

    # Detailed information (optional)
    details: SearchToProcessResponse | None = Field(
        default=None, description="Detailed status information"
    )

    # Estimated completion
    estimated_completion: datetime | None = Field(
        default=None, description="Estimated completion time"
    )

    # Response metadata
    checked_at: datetime = Field(
        default_factory=datetime.now, description="Status check timestamp"
    )


# Forward reference resolution
SearchProgressResponse.model_rebuild()
