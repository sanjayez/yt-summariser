# Parallel Processing Integration for Search-to-Process

This document describes the core parallel processing orchestrator that enables efficient processing of multiple videos found through search queries.

## Overview

The parallel processing integration bridges the gap between search functionality and video processing, allowing multiple videos to be processed simultaneously using Celery groups and chords.

## Architecture Components

### 1. Parallel Processing Orchestrator (`topic/parallel_tasks.py`)

The main orchestrator that handles:
- Creating URLRequestTable entries for each video found in search results
- Launching parallel video processing using Celery groups
- Monitoring completion and updating search request status
- Handling the UUID vs ID relationship between models

**Key Tasks:**
- `process_search_results`: Main orchestrator for parallel processing
- `finalize_search_processing`: Completion monitoring and status updates
- `get_search_processing_status`: Status checking for ongoing processes

### 2. Video Processing Adapter (`video_processor/processors/search_adapter.py`)

Adapts the existing video processing pipeline to work with search-generated videos:
- Links SearchRequest → URLRequestTable → VideoMetadata
- Ensures proper relationship handling
- Provides search-specific video processing tasks
- Handles status updates across the relationship chain

**Key Tasks:**
- `process_search_video`: Video processing adapter for search results
- `update_search_video_status`: Status updates for search videos
- `get_search_video_results`: Retrieves processing results

### 3. Enhanced Workflow (`video_processor/processors/workflow.py`)

Extended existing workflow to support parallel processing:
- `process_parallel_videos`: Handles multiple videos using Celery groups
- Maintains backward compatibility with single video processing
- Integrates with the existing processing chain

### 4. Integrated Search Tasks (`topic/tasks.py`)

Enhanced search tasks that can trigger video processing:
- `process_search_with_videos`: Integrated search and video processing
- Maintains existing search-only functionality
- Provides option to start video processing after search completion

## Data Flow

```
1. Search Request Created
   ↓
2. Search Processing (find videos)
   ↓
3. URLRequestTable entries created for each video
   ↓
4. Parallel Video Processing (Celery group)
   ├── Video 1: metadata → transcript → summary → embedding
   ├── Video 2: metadata → transcript → summary → embedding
   └── Video N: metadata → transcript → summary → embedding
   ↓
5. Completion Monitoring (Celery chord callback)
   ↓
6. Status Updates and Results Aggregation
```

## Key Features

### UUID vs ID Mismatch Resolution
- **Problem**: SearchRequest uses UUID primary keys, URLRequestTable uses integer IDs
- **Solution**: Proper field mapping and relationship handling
- **Implementation**: Use `id` field for URLRequestTable operations, `request_id` for SearchRequest

### Efficient Database Operations
- Bulk creation of URLRequestTable entries
- Select_related and prefetch_related for optimal queries
- Atomic transactions for consistency

### Error Handling
- Graceful handling of partial failures
- Proper status tracking at all levels
- Retry logic for failed video processing

### Performance Optimizations
- Celery groups for true parallelism
- Efficient database queries
- Proper task routing and worker management

## Usage Examples

### 1. Search Only
```python
from topic.tasks import process_search_query

# Create search request (via API or direct model creation)
result = process_search_query.apply_async(args=[search_request_id])
```

### 2. Parallel Processing After Search
```python
from topic.parallel_tasks import process_search_results

# Assumes search is already complete
result = process_search_results.apply_async(args=[search_request_id])
```

### 3. Integrated Workflow
```python
from topic.tasks import process_search_with_videos

# Search and then process videos
result = process_search_with_videos.apply_async(
    args=[search_request_id],
    kwargs={'start_video_processing': True}
)
```

### 4. Monitor Progress
```python
from topic.parallel_tasks import get_search_processing_status

# Check status
status = get_search_processing_status.apply_async(args=[search_request_id])
```

## API Integration

### Starting Parallel Processing
```python
# In your API view
search_request = SearchRequest.objects.get(request_id=search_request_id)
if search_request.status == 'success' and search_request.video_urls:
    # Start parallel processing
    task = process_search_results.apply_async(args=[str(search_request_id)])
    return {'task_id': task.id, 'status': 'processing_started'}
```

### Checking Status
```python
# Status endpoint
status_task = get_search_processing_status.apply_async(args=[search_request_id])
status = status_task.get()
return JsonResponse(status)
```

### Getting Results
```python
# Results endpoint
from video_processor.processors.search_adapter import get_search_video_results
results_task = get_search_video_results.apply_async(args=[search_request_id])
results = results_task.get()
return JsonResponse(results)
```

## Database Schema

### Relationships
```
SearchSession (1) → (N) SearchRequest
SearchRequest (1) → (N) URLRequestTable
URLRequestTable (1) → (1) VideoMetadata
VideoMetadata (1) → (1) VideoTranscript
VideoTranscript (1) → (N) TranscriptSegment
```

### Key Fields
- `SearchRequest.request_id`: UUID primary key
- `URLRequestTable.id`: Integer primary key (for video processing)
- `URLRequestTable.request_id`: UUID field (for API consistency)
- `URLRequestTable.search_request`: Foreign key to SearchRequest

## Testing

### Management Command
```bash
python manage.py test_parallel_processing --query "machine learning" --test-mode integrated --wait-for-completion
```

### Example Script
```bash
cd examples/
python parallel_processing_example.py integrated
```

## Configuration

### Celery Settings
```python
# In settings.py
CELERY_TASK_ROUTES = {
    'topic.process_search_results': {'queue': 'search_processing'},
    'video_processor.process_youtube_video': {'queue': 'video_processing'},
}

# For parallel processing
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ACKS_LATE = True
```

### Performance Tuning
- Worker count: Adjust based on available CPU cores
- Queue configuration: Separate queues for search vs video processing
- Memory management: Monitor worker memory usage with long-running tasks

## Error Handling

### Partial Failures
- Individual video processing failures don't stop the entire batch
- Status tracking allows identification of failed videos
- Retry logic can be implemented for specific failure types

### Database Consistency
- Atomic transactions ensure data integrity
- Proper status updates at all relationship levels
- Rollback mechanisms for critical failures

## Monitoring

### Task Status
- Celery task monitoring via flower
- Database status fields for persistence
- Logging at all levels for debugging

### Performance Metrics
- Processing time per video
- Success/failure rates
- Resource utilization

## Backward Compatibility

The parallel processing system maintains full backward compatibility:
- Existing single video processing continues to work
- No changes required to existing API endpoints
- Existing search functionality unchanged

## Future Enhancements

1. **Dynamic Scaling**: Auto-adjust worker count based on load
2. **Priority Queues**: Prioritize processing based on user tiers
3. **Caching**: Cache processed videos to avoid reprocessing
4. **Streaming Results**: Real-time updates as videos complete processing
5. **Batch Size Optimization**: Intelligent grouping of videos for optimal performance