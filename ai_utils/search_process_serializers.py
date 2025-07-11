"""
Django REST Framework serializers for search-to-process integration.
These serializers provide Django-compatible serialization/deserialization
while maintaining compatibility with Pydantic models.
"""

from rest_framework import serializers
from datetime import datetime
from typing import Dict, Any, List, Optional
from .search_process_models import (
    SearchProcessingStatus,
    VideoProcessingStatus,
    ProcessingPriority,
    ErrorType,
    SearchProgressRequest,
    SearchProgressResponse,
    VideoProcessingRequest,
    VideoProcessingResult,
    VideoProcessingStatusAggregation,
    ParallelProcessingConfig,
    WorkerStatus,
    ParallelProcessingOrchestration,
    ProcessingError,
    RetryConfig,
    RetryTask,
    SearchToProcessRequest,
    SearchToProcessResponse,
    StatusCheckRequest,
    StatusCheckResponse,
)

# ========== Choice Field Serializers ==========

class SearchProcessingStatusSerializer(serializers.ChoiceField):
    """Serializer for SearchProcessingStatus enum"""
    def __init__(self, **kwargs):
        choices = [(status.value, status.value) for status in SearchProcessingStatus]
        super().__init__(choices=choices, **kwargs)

class VideoProcessingStatusSerializer(serializers.ChoiceField):
    """Serializer for VideoProcessingStatus enum"""
    def __init__(self, **kwargs):
        choices = [(status.value, status.value) for status in VideoProcessingStatus]
        super().__init__(choices=choices, **kwargs)

class ProcessingPrioritySerializer(serializers.ChoiceField):
    """Serializer for ProcessingPriority enum"""
    def __init__(self, **kwargs):
        choices = [(priority.value, priority.value) for priority in ProcessingPriority]
        super().__init__(choices=choices, **kwargs)

class ErrorTypeSerializer(serializers.ChoiceField):
    """Serializer for ErrorType enum"""
    def __init__(self, **kwargs):
        choices = [(error.value, error.value) for error in ErrorType]
        super().__init__(choices=choices, **kwargs)

# ========== Search Progress Serializers ==========

class SearchProgressRequestSerializer(serializers.Serializer):
    """Serializer for SearchProgressRequest"""
    session_id = serializers.CharField(max_length=255)
    query = serializers.CharField(min_length=1, max_length=500)
    max_videos = serializers.IntegerField(min_value=1, max_value=20, default=5)
    priority = ProcessingPrioritySerializer(default=ProcessingPriority.NORMAL.value)
    enable_parallel = serializers.BooleanField(default=True)
    max_parallel_workers = serializers.IntegerField(min_value=1, max_value=10, default=3)
    timeout_seconds = serializers.IntegerField(min_value=60, max_value=3600, default=1800)
    
    # Processing configuration
    include_metadata = serializers.BooleanField(default=True)
    include_transcript = serializers.BooleanField(default=True)
    include_embedding = serializers.BooleanField(default=True)
    include_summary = serializers.BooleanField(default=True)
    
    # Search configuration
    search_filters = serializers.JSONField(required=False, allow_null=True)
    
    def validate_query(self, value):
        """Validate query field"""
        if not value.strip():
            raise serializers.ValidationError("Query cannot be empty")
        return value.strip()
    
    def to_pydantic(self) -> SearchProgressRequest:
        """Convert to Pydantic model"""
        return SearchProgressRequest(**self.validated_data)

class VideoProcessingResultSerializer(serializers.Serializer):
    """Serializer for VideoProcessingResult"""
    video_id = serializers.CharField(max_length=50)
    video_url = serializers.URLField()
    status = VideoProcessingStatusSerializer()
    
    # Processing stages
    metadata_status = VideoProcessingStatusSerializer(default=VideoProcessingStatus.PENDING.value)
    transcript_status = VideoProcessingStatusSerializer(default=VideoProcessingStatus.PENDING.value)
    embedding_status = VideoProcessingStatusSerializer(default=VideoProcessingStatus.PENDING.value)
    summary_status = VideoProcessingStatusSerializer(default=VideoProcessingStatus.PENDING.value)
    
    # Results
    metadata_available = serializers.BooleanField(default=False)
    transcript_available = serializers.BooleanField(default=False)
    embeddings_available = serializers.BooleanField(default=False)
    summary_available = serializers.BooleanField(default=False)
    
    # Timing information
    start_time = serializers.DateTimeField(allow_null=True, required=False)
    end_time = serializers.DateTimeField(allow_null=True, required=False)
    processing_time_seconds = serializers.FloatField(allow_null=True, required=False)
    
    # Error handling
    error_message = serializers.CharField(max_length=1000, allow_null=True, required=False)
    error_type = ErrorTypeSerializer(allow_null=True, required=False)
    retry_count = serializers.IntegerField(default=0)
    
    # Metadata
    created_at = serializers.DateTimeField(default=datetime.now)
    updated_at = serializers.DateTimeField(default=datetime.now)

class SearchProgressResponseSerializer(serializers.Serializer):
    """Serializer for SearchProgressResponse"""
    request_id = serializers.CharField(max_length=255)
    session_id = serializers.CharField(max_length=255)
    status = SearchProcessingStatusSerializer()
    query = serializers.CharField(max_length=500)
    
    # Progress tracking
    total_videos = serializers.IntegerField(default=0)
    processed_videos = serializers.IntegerField(default=0)
    successful_videos = serializers.IntegerField(default=0)
    failed_videos = serializers.IntegerField(default=0)
    
    # Timing information
    search_start_time = serializers.DateTimeField(allow_null=True, required=False)
    search_end_time = serializers.DateTimeField(allow_null=True, required=False)
    processing_start_time = serializers.DateTimeField(allow_null=True, required=False)
    processing_end_time = serializers.DateTimeField(allow_null=True, required=False)
    
    # Results
    video_urls = serializers.ListField(child=serializers.URLField(), default=list)
    processing_results = VideoProcessingResultSerializer(many=True, default=list)
    
    # Error handling
    error_message = serializers.CharField(max_length=1000, allow_null=True, required=False)
    error_type = ErrorTypeSerializer(allow_null=True, required=False)
    
    # Metadata
    created_at = serializers.DateTimeField(default=datetime.now)
    updated_at = serializers.DateTimeField(default=datetime.now)

# ========== Video Processing Serializers ==========

class VideoProcessingRequestSerializer(serializers.Serializer):
    """Serializer for VideoProcessingRequest"""
    video_url = serializers.URLField()
    video_id = serializers.CharField(max_length=50)
    parent_request_id = serializers.CharField(max_length=255)
    priority = ProcessingPrioritySerializer(default=ProcessingPriority.NORMAL.value)
    
    # Processing configuration
    include_metadata = serializers.BooleanField(default=True)
    include_transcript = serializers.BooleanField(default=True)
    include_embedding = serializers.BooleanField(default=True)
    include_summary = serializers.BooleanField(default=True)
    
    # Retry configuration
    max_retries = serializers.IntegerField(min_value=0, max_value=10, default=3)
    retry_delay = serializers.IntegerField(min_value=0, max_value=300, default=60)
    
    def validate_video_url(self, value):
        """Validate video URL"""
        if not any(domain in value.lower() for domain in ['youtube.com', 'youtu.be']):
            raise serializers.ValidationError("Must be a valid YouTube URL")
        return value
    
    def to_pydantic(self) -> VideoProcessingRequest:
        """Convert to Pydantic model"""
        return VideoProcessingRequest(**self.validated_data)

class VideoProcessingStatusAggregationSerializer(serializers.Serializer):
    """Serializer for VideoProcessingStatusAggregation"""
    request_id = serializers.CharField(max_length=255)
    total_videos = serializers.IntegerField()
    
    # Status counts
    pending_count = serializers.IntegerField(default=0)
    processing_count = serializers.IntegerField(default=0)
    completed_count = serializers.IntegerField(default=0)
    failed_count = serializers.IntegerField(default=0)
    cancelled_count = serializers.IntegerField(default=0)
    
    # Processing stages
    metadata_completed = serializers.IntegerField(default=0)
    transcript_completed = serializers.IntegerField(default=0)
    embedding_completed = serializers.IntegerField(default=0)
    summary_completed = serializers.IntegerField(default=0)
    
    # Progress metrics
    overall_progress = serializers.FloatField(min_value=0.0, max_value=100.0, default=0.0)
    estimated_completion_time = serializers.DateTimeField(allow_null=True, required=False)
    
    # Performance metrics
    average_processing_time = serializers.FloatField(allow_null=True, required=False)
    throughput_videos_per_minute = serializers.FloatField(allow_null=True, required=False)
    
    updated_at = serializers.DateTimeField(default=datetime.now)

# ========== Parallel Processing Serializers ==========

class ParallelProcessingConfigSerializer(serializers.Serializer):
    """Serializer for ParallelProcessingConfig"""
    max_workers = serializers.IntegerField(min_value=1, max_value=10, default=3)
    chunk_size = serializers.IntegerField(min_value=1, max_value=20, default=5)
    enable_load_balancing = serializers.BooleanField(default=True)
    worker_timeout = serializers.IntegerField(min_value=60, max_value=1800, default=900)
    
    # Resource limits
    max_memory_mb = serializers.IntegerField(min_value=512, max_value=4096, default=1024)
    max_cpu_percent = serializers.IntegerField(min_value=10, max_value=100, default=80)
    
    # Retry configuration
    max_retries = serializers.IntegerField(min_value=0, max_value=10, default=3)
    retry_delay = serializers.IntegerField(min_value=0, max_value=300, default=60)
    backoff_factor = serializers.FloatField(min_value=1.0, max_value=10.0, default=2.0)
    
    def to_pydantic(self) -> ParallelProcessingConfig:
        """Convert to Pydantic model"""
        return ParallelProcessingConfig(**self.validated_data)

class WorkerStatusSerializer(serializers.Serializer):
    """Serializer for WorkerStatus"""
    worker_id = serializers.CharField(max_length=255)
    status = serializers.CharField(max_length=50)
    current_task = serializers.CharField(max_length=255, allow_null=True, required=False)
    videos_processed = serializers.IntegerField(default=0)
    videos_failed = serializers.IntegerField(default=0)
    
    # Performance metrics
    start_time = serializers.DateTimeField(default=datetime.now)
    last_activity = serializers.DateTimeField(default=datetime.now)
    processing_time_seconds = serializers.FloatField(default=0.0)
    
    # Resource usage
    memory_usage_mb = serializers.FloatField(allow_null=True, required=False)
    cpu_usage_percent = serializers.FloatField(allow_null=True, required=False)

class ParallelProcessingOrchestrationSerializer(serializers.Serializer):
    """Serializer for ParallelProcessingOrchestration"""
    orchestration_id = serializers.CharField(max_length=255)
    request_id = serializers.CharField(max_length=255)
    config = ParallelProcessingConfigSerializer()
    
    # Worker management
    workers = WorkerStatusSerializer(many=True, default=list)
    active_workers = serializers.IntegerField(default=0)
    
    # Task distribution
    total_tasks = serializers.IntegerField(default=0)
    completed_tasks = serializers.IntegerField(default=0)
    failed_tasks = serializers.IntegerField(default=0)
    pending_tasks = serializers.IntegerField(default=0)
    
    # Performance metrics
    throughput_tasks_per_minute = serializers.FloatField(allow_null=True, required=False)
    average_task_time = serializers.FloatField(allow_null=True, required=False)
    
    # Status and timing
    status = serializers.CharField(max_length=50, default="initializing")
    start_time = serializers.DateTimeField(allow_null=True, required=False)
    end_time = serializers.DateTimeField(allow_null=True, required=False)
    
    created_at = serializers.DateTimeField(default=datetime.now)
    updated_at = serializers.DateTimeField(default=datetime.now)

# ========== Error Handling Serializers ==========

class ProcessingErrorSerializer(serializers.Serializer):
    """Serializer for ProcessingError"""
    error_id = serializers.CharField(max_length=255)
    request_id = serializers.CharField(max_length=255)
    video_id = serializers.CharField(max_length=50, allow_null=True, required=False)
    error_type = ErrorTypeSerializer()
    error_message = serializers.CharField(max_length=1000)
    error_details = serializers.JSONField(allow_null=True, required=False)
    
    # Context information
    processing_stage = serializers.CharField(max_length=100)
    worker_id = serializers.CharField(max_length=255, allow_null=True, required=False)
    
    # Retry information
    retry_count = serializers.IntegerField(default=0)
    can_retry = serializers.BooleanField(default=True)
    next_retry_at = serializers.DateTimeField(allow_null=True, required=False)
    
    # Metadata
    occurred_at = serializers.DateTimeField(default=datetime.now)
    resolved_at = serializers.DateTimeField(allow_null=True, required=False)

class RetryConfigSerializer(serializers.Serializer):
    """Serializer for RetryConfig"""
    max_retries = serializers.IntegerField(min_value=0, max_value=10, default=3)
    initial_delay = serializers.IntegerField(min_value=0, max_value=300, default=60)
    max_delay = serializers.IntegerField(min_value=60, max_value=3600, default=600)
    backoff_factor = serializers.FloatField(min_value=1.0, max_value=10.0, default=2.0)
    
    # Error-specific retry settings
    retryable_errors = serializers.ListField(
        child=ErrorTypeSerializer(),
        default=lambda: [ErrorType.NETWORK_ERROR.value, ErrorType.PROCESSING_TIMEOUT.value, ErrorType.QUOTA_EXCEEDED.value]
    )
    
    def to_pydantic(self) -> RetryConfig:
        """Convert to Pydantic model"""
        return RetryConfig(**self.validated_data)

class RetryTaskSerializer(serializers.Serializer):
    """Serializer for RetryTask"""
    task_id = serializers.CharField(max_length=255)
    request_id = serializers.CharField(max_length=255)
    video_id = serializers.CharField(max_length=50, allow_null=True, required=False)
    processing_stage = serializers.CharField(max_length=100)
    
    # Retry information
    retry_count = serializers.IntegerField(default=0)
    max_retries = serializers.IntegerField(default=3)
    next_retry_at = serializers.DateTimeField()
    
    # Original error
    original_error = ProcessingErrorSerializer()
    
    # Status
    status = serializers.CharField(max_length=50, default="pending")
    created_at = serializers.DateTimeField(default=datetime.now)
    updated_at = serializers.DateTimeField(default=datetime.now)

# ========== Main Integration Serializers ==========

class SearchToProcessRequestSerializer(serializers.Serializer):
    """Serializer for SearchToProcessRequest"""
    session_id = serializers.CharField(max_length=255)
    query = serializers.CharField(min_length=1, max_length=500)
    
    # Search configuration
    max_videos = serializers.IntegerField(min_value=1, max_value=20, default=5)
    search_filters = serializers.JSONField(allow_null=True, required=False)
    
    # Processing configuration
    processing_config = VideoProcessingRequestSerializer()
    parallel_config = ParallelProcessingConfigSerializer(default=dict)
    retry_config = RetryConfigSerializer(default=dict)
    
    # Priority and timing
    priority = ProcessingPrioritySerializer(default=ProcessingPriority.NORMAL.value)
    timeout_seconds = serializers.IntegerField(min_value=60, max_value=3600, default=1800)
    
    # Metadata
    user_ip = serializers.IPAddressField(allow_null=True, required=False)
    user_agent = serializers.CharField(max_length=500, allow_null=True, required=False)
    
    def validate_query(self, value):
        """Validate query field"""
        if not value.strip():
            raise serializers.ValidationError("Query cannot be empty")
        return value.strip()
    
    def to_pydantic(self) -> SearchToProcessRequest:
        """Convert to Pydantic model"""
        validated_data = self.validated_data.copy()
        
        # Convert nested serializers to Pydantic models
        if 'processing_config' in validated_data:
            processing_serializer = VideoProcessingRequestSerializer(data=validated_data['processing_config'])
            if processing_serializer.is_valid():
                validated_data['processing_config'] = processing_serializer.to_pydantic()
        
        if 'parallel_config' in validated_data:
            parallel_serializer = ParallelProcessingConfigSerializer(data=validated_data['parallel_config'])
            if parallel_serializer.is_valid():
                validated_data['parallel_config'] = parallel_serializer.to_pydantic()
        
        if 'retry_config' in validated_data:
            retry_serializer = RetryConfigSerializer(data=validated_data['retry_config'])
            if retry_serializer.is_valid():
                validated_data['retry_config'] = retry_serializer.to_pydantic()
        
        return SearchToProcessRequest(**validated_data)

class SearchToProcessResponseSerializer(serializers.Serializer):
    """Serializer for SearchToProcessResponse"""
    request_id = serializers.CharField(max_length=255)
    session_id = serializers.CharField(max_length=255)
    status = SearchProcessingStatusSerializer()
    
    # Progress information
    progress = SearchProgressResponseSerializer()
    video_statuses = VideoProcessingResultSerializer(many=True, default=list)
    aggregated_status = VideoProcessingStatusAggregationSerializer()
    
    # Parallel processing information
    orchestration = ParallelProcessingOrchestrationSerializer(allow_null=True, required=False)
    
    # Error handling
    errors = ProcessingErrorSerializer(many=True, default=list)
    retry_tasks = RetryTaskSerializer(many=True, default=list)
    
    # Performance metrics
    total_processing_time = serializers.FloatField(allow_null=True, required=False)
    videos_per_minute = serializers.FloatField(allow_null=True, required=False)
    
    # Timestamps
    created_at = serializers.DateTimeField(default=datetime.now)
    updated_at = serializers.DateTimeField(default=datetime.now)

# ========== Status Check Serializers ==========

class StatusCheckRequestSerializer(serializers.Serializer):
    """Serializer for StatusCheckRequest"""
    request_id = serializers.CharField(max_length=255)
    include_details = serializers.BooleanField(default=True)
    include_errors = serializers.BooleanField(default=True)
    include_performance = serializers.BooleanField(default=False)
    
    def to_pydantic(self) -> StatusCheckRequest:
        """Convert to Pydantic model"""
        return StatusCheckRequest(**self.validated_data)

class StatusCheckResponseSerializer(serializers.Serializer):
    """Serializer for StatusCheckResponse"""
    request_id = serializers.CharField(max_length=255)
    status = SearchProcessingStatusSerializer()
    progress_percentage = serializers.FloatField(min_value=0.0, max_value=100.0, default=0.0)
    
    # Quick status overview
    total_videos = serializers.IntegerField(default=0)
    completed_videos = serializers.IntegerField(default=0)
    failed_videos = serializers.IntegerField(default=0)
    
    # Detailed information (optional)
    details = SearchToProcessResponseSerializer(allow_null=True, required=False)
    
    # Estimated completion
    estimated_completion = serializers.DateTimeField(allow_null=True, required=False)
    
    # Response metadata
    checked_at = serializers.DateTimeField(default=datetime.now)

# ========== Helper Functions ==========

def pydantic_to_dict(pydantic_model) -> Dict[str, Any]:
    """Convert Pydantic model to dictionary compatible with DRF serializers"""
    if hasattr(pydantic_model, 'model_dump'):
        return pydantic_model.model_dump()
    elif hasattr(pydantic_model, 'dict'):
        return pydantic_model.dict()
    else:
        return dict(pydantic_model)

def dict_to_pydantic(data: Dict[str, Any], model_class):
    """Convert dictionary to Pydantic model"""
    return model_class(**data)

def serialize_pydantic_for_drf(pydantic_model, serializer_class):
    """Serialize Pydantic model using DRF serializer"""
    data = pydantic_to_dict(pydantic_model)
    serializer = serializer_class(data=data)
    serializer.is_valid(raise_exception=True)
    return serializer.validated_data

def deserialize_drf_to_pydantic(drf_data: Dict[str, Any], pydantic_class):
    """Deserialize DRF data to Pydantic model"""
    return pydantic_class(**drf_data)