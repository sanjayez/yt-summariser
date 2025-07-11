# Search-to-Process Integration Pydantic Models

This document provides a comprehensive overview of the Pydantic models created for the search-to-process integration feature.

## Overview

The search-to-process integration models provide type-safe data structures for:
- Search progress tracking
- Video processing status management  
- Parallel processing orchestration
- Error handling and retry logic
- API request/response validation

## File Structure

### Core Files

- `/ai_utils/search_process_models.py` - Main Pydantic models
- `/ai_utils/search_process_serializers.py` - Django REST Framework serializers  
- `/ai_utils/search_process_factories.py` - Factory functions for model creation
- `/examples/search_process_integration_demo.py` - Comprehensive usage examples

### Updated Files

- `/ai_utils/__init__.py` - Updated to export all new models and utilities

## Model Categories

### 1. Search Progress Tracking Models

#### SearchProgressRequest
Handles search initiation with validation for:
- Query validation (1-500 characters)
- Video limits (1-20 videos)
- Processing configuration options
- Parallel processing settings
- Timeout configuration

#### SearchProgressResponse  
Tracks search execution progress:
- Real-time status updates
- Video count tracking
- Timing information
- Error handling
- Processing results

#### SearchProgressUpdate
Lightweight progress updates:
- Status changes
- Progress percentage
- Update messages
- Timestamps

### 2. Video Processing Status Models

#### VideoProcessingRequest
Individual video processing configuration:
- YouTube URL validation
- Processing stage selection
- Retry configuration
- Priority levels

#### VideoProcessingResult
Detailed processing results per video:
- Multi-stage status tracking (metadata, transcript, embedding, summary)
- Availability flags
- Timing metrics
- Error details
- Retry counts

#### VideoProcessingStatusAggregation
Aggregated status across all videos:
- Count summaries by status
- Progress metrics
- Performance statistics
- Completion estimates

### 3. Parallel Processing Models

#### ParallelProcessingConfig
Configuration for parallel execution:
- Worker limits (1-10)
- Resource constraints
- Timeout settings
- Retry policies

#### WorkerStatus
Individual worker tracking:
- Task assignment
- Performance metrics
- Resource usage
- Activity monitoring

#### ParallelProcessingOrchestration
Overall orchestration management:
- Worker coordination
- Task distribution
- Performance monitoring
- Load balancing

### 4. Error Handling Models

#### ProcessingError
Comprehensive error tracking:
- Categorized error types
- Context information
- Retry eligibility
- Resolution tracking

#### RetryConfig
Configurable retry policies:
- Exponential backoff
- Error-specific rules
- Timeout limits
- Maximum attempts

#### RetryTask
Individual retry operations:
- Task scheduling
- Dependency tracking
- Status monitoring
- History preservation

### 5. Integration Models

#### SearchToProcessRequest
Main integration request:
- Combines all configuration options
- User context tracking
- Validation across all parameters

#### SearchToProcessResponse
Comprehensive response:
- Nested status information
- Complete progress tracking
- Error aggregation
- Performance metrics

### 6. Status Check Models

#### StatusCheckRequest
Flexible status queries:
- Detail level control
- Error inclusion options
- Performance metrics toggle

#### StatusCheckResponse
Quick status overview:
- Essential progress info
- Completion estimates
- Optional detailed data

## Key Features

### Validation
- Pydantic v2 field validation
- Custom validators for URLs, queries, and ranges
- Cross-field validation with model validators
- Enum-based status management

### Serialization
- JSON serialization/deserialization
- Django REST Framework compatibility
- Bidirectional Pydantic ↔ DRF conversion
- Type-safe data transfer

### Factory Functions
- Default value management
- Batch creation utilities
- ID generation
- Progress calculation helpers
- Time estimation utilities

### Error Handling
- Categorized error types
- Retry logic configuration
- Context preservation
- Resolution tracking

## Usage Examples

### Basic Search Request
```python
from ai_utils.search_process_factories import create_search_progress_request, generate_session_id

session_id = generate_session_id()
request = create_search_progress_request(
    session_id=session_id,
    query="machine learning tutorials",
    max_videos=5,
    priority=ProcessingPriority.HIGH
)
```

### Video Processing Workflow
```python
from ai_utils.search_process_factories import (
    create_batch_video_processing_results,
    create_video_processing_status_aggregation
)

# Create batch processing results
video_results = create_batch_video_processing_results(
    video_urls=["https://youtube.com/watch?v=abc123"],
    video_ids=["abc123"],
    status=VideoProcessingStatus.PENDING
)

# Create aggregated status
aggregated = create_video_processing_status_aggregation(
    request_id="req-123",
    total_videos=len(video_results)
)
```

### Parallel Processing Setup
```python
from ai_utils.search_process_factories import (
    create_parallel_processing_config,
    create_batch_workers,
    create_parallel_processing_orchestration
)

# Configure parallel processing
config = create_parallel_processing_config(
    max_workers=3,
    chunk_size=2,
    worker_timeout=600
)

# Create workers
workers = create_batch_workers(worker_count=3)

# Setup orchestration
orchestration = create_parallel_processing_orchestration(
    orchestration_id=generate_orchestration_id(),
    request_id="req-123",
    config=config
)
```

### Error Handling
```python
from ai_utils.search_process_factories import (
    create_processing_error,
    create_retry_config,
    create_retry_task
)

# Create error
error = create_processing_error(
    request_id="req-123",
    error_type=ErrorType.NETWORK_ERROR,
    error_message="Connection timeout",
    processing_stage="metadata_extraction"
)

# Configure retries
retry_config = create_retry_config(
    max_retries=3,
    initial_delay=60,
    backoff_factor=2.0
)

# Create retry task
retry_task = create_retry_task(
    task_id=generate_request_id(),
    request_id="req-123",
    processing_stage="metadata_extraction",
    original_error=error
)
```

### Django REST Framework Integration
```python
from ai_utils.search_process_serializers import (
    SearchProgressRequestSerializer,
    pydantic_to_dict
)

# Convert Pydantic to DRF
request_dict = pydantic_to_dict(pydantic_model)
serializer = SearchProgressRequestSerializer(data=request_dict)

if serializer.is_valid():
    # Convert back to Pydantic
    pydantic_model = serializer.to_pydantic()
```

## Status Enums

### SearchProcessingStatus
- `INITIALIZING` - Starting up
- `SEARCHING` - Finding videos  
- `SEARCH_COMPLETED` - Search finished
- `SEARCH_FAILED` - Search error
- `PROCESSING_VIDEOS` - Processing videos
- `PROCESSING_COMPLETED` - All processing done
- `PROCESSING_FAILED` - Processing error
- `COMPLETED` - Fully complete
- `CANCELLED` - User cancelled

### VideoProcessingStatus
- `PENDING` - Not started
- `METADATA_PROCESSING` - Extracting metadata
- `TRANSCRIPT_PROCESSING` - Processing transcript
- `EMBEDDING_PROCESSING` - Creating embeddings
- `SUMMARY_PROCESSING` - Generating summary
- `COMPLETED` - All stages done
- `FAILED` - Processing failed
- `CANCELLED` - Processing cancelled

### ErrorType
- `SEARCH_ERROR` - Search API errors
- `VIDEO_METADATA_ERROR` - Metadata extraction issues
- `TRANSCRIPT_ERROR` - Transcript processing issues
- `EMBEDDING_ERROR` - Embedding generation issues
- `SUMMARY_ERROR` - Summary generation issues
- `NETWORK_ERROR` - Network connectivity issues
- `QUOTA_EXCEEDED` - API quota limits
- `INVALID_VIDEO` - Invalid video content
- `PROCESSING_TIMEOUT` - Operation timeouts
- `UNKNOWN_ERROR` - Unclassified errors

## Performance Utilities

### Progress Calculation
```python
from ai_utils.search_process_factories import calculate_progress_percentage

progress = calculate_progress_percentage(completed=3, total=10)
# Returns: 30.0
```

### Throughput Calculation
```python
from ai_utils.search_process_factories import calculate_throughput

throughput = calculate_throughput(
    completed=10,
    start_time=start_time,
    current_time=datetime.now()
)
# Returns: items per minute
```

### Completion Estimation
```python
from ai_utils.search_process_factories import estimate_completion_time

estimated = estimate_completion_time(
    completed=3,
    total=10,
    start_time=start_time
)
# Returns: estimated completion datetime
```

## Integration with Existing Models

The search-process models are designed to work seamlessly with existing Django models:

### VideoMetadata Integration
- `video_id` fields match `VideoMetadata.video_id`
- Status tracking complements Django model status
- Processing flags align with embedding tracking

### URLRequestTable Integration  
- `request_id` links to `URLRequestTable.request_id`
- Status synchronization with Django models
- Search request relationships preserved

### SearchRequest Integration
- `session_id` links to search sessions
- Query tracking and enhancement support
- Result URL management

## Testing

A comprehensive demo script is available at:
```bash
python examples/search_process_integration_demo.py
```

This demonstrates all model features including:
- Model creation and validation
- Status tracking workflows
- Error handling scenarios
- Serializer integration
- Performance calculations

## Best Practices

### Model Creation
- Use factory functions for consistent defaults
- Generate IDs with provided utilities
- Validate data early with Pydantic models

### Status Management
- Update timestamps on status changes
- Use appropriate enum values
- Track progress percentages accurately

### Error Handling
- Categorize errors appropriately
- Preserve context information
- Configure retries based on error types

### Performance
- Monitor resource usage
- Calculate throughput metrics
- Estimate completion times

### Serialization
- Use provided DRF serializers
- Convert between Pydantic and DRF as needed
- Validate data at API boundaries

## Future Enhancements

Potential future additions:
- Caching layer models
- Notification models
- Analytics and reporting models
- Advanced scheduling models
- Resource optimization models