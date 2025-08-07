"""
AI Utils Package
Provides modular AI utilities for embeddings, vector storage, and language models
"""

__version__ = "0.1.0"

from .config import get_config, AIConfig
from .providers import OpenAIEmbeddingProvider, PineconeVectorStoreProvider, GeminiLLMProvider
from .services import EmbeddingService, VectorService, LLMService
from .models import (
    VectorDocument, VectorQuery, VectorSearchResult, VectorSearchResponse,
    EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse,
    RAGQuery, RAGResponse, ProcessingJob, ProcessingStatus,
    ChatMessage, ChatRequest, ChatResponse, ChatRole,
    TextGenerationRequest, TextGenerationResponse
)
from .search_process_models import (
    SearchProcessingStatus, VideoProcessingStatus, ProcessingPriority, ErrorType,
    SearchProgressRequest, SearchProgressResponse, SearchProgressUpdate,
    VideoProcessingRequest, VideoProcessingResult, VideoProcessingStatusAggregation,
    ParallelProcessingConfig, WorkerStatus, ParallelProcessingOrchestration,
    ProcessingError, RetryConfig, RetryTask,
    SearchToProcessRequest, SearchToProcessResponse,
    StatusCheckRequest, StatusCheckResponse
)
from .search_process_serializers import (
    SearchProgressRequestSerializer, SearchProgressResponseSerializer,
    VideoProcessingRequestSerializer, VideoProcessingResultSerializer,
    VideoProcessingStatusAggregationSerializer,
    ParallelProcessingConfigSerializer, WorkerStatusSerializer,
    ParallelProcessingOrchestrationSerializer,
    ProcessingErrorSerializer, RetryConfigSerializer, RetryTaskSerializer,
    SearchToProcessRequestSerializer, SearchToProcessResponseSerializer,
    StatusCheckRequestSerializer, StatusCheckResponseSerializer
)
from .search_process_factories import (
    create_search_progress_request, create_search_progress_response,
    create_video_processing_request, create_video_processing_result,
    create_video_processing_status_aggregation,
    create_parallel_processing_config, create_worker_status,
    create_parallel_processing_orchestration,
    create_processing_error, create_retry_config, create_retry_task,
    create_search_to_process_request, create_search_to_process_response,
    create_status_check_request, create_status_check_response,
    generate_request_id, generate_session_id, generate_orchestration_id,
    calculate_progress_percentage, estimate_completion_time, calculate_throughput
)

__all__ = [
    # Configuration
    "get_config", "AIConfig",
    
    # Providers
    "OpenAIEmbeddingProvider", "PineconeVectorStoreProvider", "GeminiLLMProvider",
    
    # Services
    "EmbeddingService", "VectorService", "LLMService",
    
    # Models
    "VectorDocument", "VectorQuery", "VectorSearchResult", "VectorSearchResponse",
    "EmbeddingRequest", "EmbeddingResponse", "BatchEmbeddingRequest", "BatchEmbeddingResponse",
    "RAGQuery", "RAGResponse", "ProcessingJob", "ProcessingStatus",
    "ChatMessage", "ChatRequest", "ChatResponse", "ChatRole",
    "TextGenerationRequest", "TextGenerationResponse",
    
    # Search-to-Process Models
    "SearchProcessingStatus", "VideoProcessingStatus", "ProcessingPriority", "ErrorType",
    "SearchProgressRequest", "SearchProgressResponse", "SearchProgressUpdate",
    "VideoProcessingRequest", "VideoProcessingResult", "VideoProcessingStatusAggregation",
    "ParallelProcessingConfig", "WorkerStatus", "ParallelProcessingOrchestration",
    "ProcessingError", "RetryConfig", "RetryTask",
    "SearchToProcessRequest", "SearchToProcessResponse",
    "StatusCheckRequest", "StatusCheckResponse",
    
    # Search-to-Process Serializers
    "SearchProgressRequestSerializer", "SearchProgressResponseSerializer",
    "VideoProcessingRequestSerializer", "VideoProcessingResultSerializer",
    "VideoProcessingStatusAggregationSerializer",
    "ParallelProcessingConfigSerializer", "WorkerStatusSerializer",
    "ParallelProcessingOrchestrationSerializer",
    "ProcessingErrorSerializer", "RetryConfigSerializer", "RetryTaskSerializer",
    "SearchToProcessRequestSerializer", "SearchToProcessResponseSerializer",
    "StatusCheckRequestSerializer", "StatusCheckResponseSerializer",
    
    # Search-to-Process Factories
    "create_search_progress_request", "create_search_progress_response",
    "create_video_processing_request", "create_video_processing_result",
    "create_video_processing_status_aggregation",
    "create_parallel_processing_config", "create_worker_status",
    "create_parallel_processing_orchestration",
    "create_processing_error", "create_retry_config", "create_retry_task",
    "create_search_to_process_request", "create_search_to_process_response",
    "create_status_check_request", "create_status_check_response",
    "generate_request_id", "generate_session_id", "generate_orchestration_id",
    "calculate_progress_percentage", "estimate_completion_time", "calculate_throughput"
] 