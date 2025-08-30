#!/usr/bin/env python
"""
Demo script showing how to use the search-to-process integration Pydantic models.

This script demonstrates:
1. Creating search-to-process requests
2. Tracking video processing progress
3. Managing parallel processing orchestration
4. Handling errors and retries
5. Using Django REST Framework serializers
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

# Import factory functions
from ai_utils.search_process_factories import (
    calculate_progress_percentage,
    calculate_throughput,
    create_batch_video_processing_results,
    create_batch_workers,
    create_parallel_processing_config,
    create_parallel_processing_orchestration,
    create_processing_error,
    create_retry_config,
    create_retry_task,
    create_search_progress_request,
    create_search_progress_response,
    create_search_to_process_request,
    create_search_to_process_response,
    create_status_check_request,
    create_status_check_response,
    create_video_processing_status_aggregation,
    estimate_completion_time,
    generate_orchestration_id,
    generate_request_id,
    generate_session_id,
)

# Import the new search-process integration models
from ai_utils.search_process_models import (
    ErrorType,
    ProcessingPriority,
    SearchProcessingStatus,
    VideoProcessingStatus,
)

# Import serializers
from ai_utils.search_process_serializers import (
    SearchProgressRequestSerializer,
    pydantic_to_dict,
)


def demo_basic_search_request():
    """Demonstrate creating and using a basic search request."""
    print("=== Demo: Basic Search Request ===")

    # Generate IDs
    session_id = generate_session_id()
    request_id = generate_request_id()

    # Create a search progress request
    search_request = create_search_progress_request(
        session_id=session_id,
        query="machine learning tutorials",
        max_videos=5,
        priority=ProcessingPriority.HIGH,
    )

    print("Created search request:")
    print(f"  Session ID: {search_request.session_id}")
    print(f"  Query: {search_request.query}")
    print(f"  Max Videos: {search_request.max_videos}")
    print(f"  Priority: {search_request.priority}")
    print(f"  Parallel Enabled: {search_request.enable_parallel}")

    # Create a search progress response
    search_response = create_search_progress_response(
        request_id=request_id,
        session_id=session_id,
        query=search_request.query,
        status=SearchProcessingStatus.SEARCHING,
    )

    print("\nCreated search response:")
    print(f"  Request ID: {search_response.request_id}")
    print(f"  Status: {search_response.status}")
    print(f"  Total Videos: {search_response.total_videos}")

    return search_request, search_response


def demo_video_processing_workflow():
    """Demonstrate video processing workflow with multiple videos."""
    print("\n=== Demo: Video Processing Workflow ===")

    # Sample video data
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "https://www.youtube.com/watch?v=oHg5SJYRHA0",
    ]
    video_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw", "oHg5SJYRHA0"]

    request_id = generate_request_id()

    # Create batch of video processing results
    video_results = create_batch_video_processing_results(
        video_urls=video_urls, video_ids=video_ids, status=VideoProcessingStatus.PENDING
    )

    print(f"Created {len(video_results)} video processing results:")
    for i, result in enumerate(video_results):
        print(f"  {i + 1}. {result.video_id} - {result.status}")

    # Create aggregated status
    aggregated_status = create_video_processing_status_aggregation(
        request_id=request_id, total_videos=len(video_results)
    )

    print("\nAggregated status:")
    print(f"  Total Videos: {aggregated_status.total_videos}")
    print(f"  Pending: {aggregated_status.pending_count}")
    print(f"  Overall Progress: {aggregated_status.overall_progress}%")

    # Simulate processing progress
    print("\nSimulating processing progress...")

    # Update first video to processing
    video_results[0].status = VideoProcessingStatus.METADATA_PROCESSING
    video_results[0].metadata_status = VideoProcessingStatus.METADATA_PROCESSING
    aggregated_status.pending_count = 2
    aggregated_status.processing_count = 1
    aggregated_status.overall_progress = calculate_progress_percentage(1, 3)

    print(f"  Video 1 status: {video_results[0].status}")
    print(f"  Overall progress: {aggregated_status.overall_progress:.1f}%")

    # Complete first video
    video_results[0].status = VideoProcessingStatus.COMPLETED
    video_results[0].metadata_status = VideoProcessingStatus.COMPLETED
    video_results[0].transcript_status = VideoProcessingStatus.COMPLETED
    video_results[0].embedding_status = VideoProcessingStatus.COMPLETED
    video_results[0].summary_status = VideoProcessingStatus.COMPLETED
    video_results[0].metadata_available = True
    video_results[0].transcript_available = True
    video_results[0].embeddings_available = True
    video_results[0].summary_available = True
    video_results[0].end_time = datetime.now()
    video_results[0].processing_time_seconds = 45.2

    aggregated_status.pending_count = 2
    aggregated_status.processing_count = 0
    aggregated_status.completed_count = 1
    aggregated_status.metadata_completed = 1
    aggregated_status.transcript_completed = 1
    aggregated_status.embedding_completed = 1
    aggregated_status.summary_completed = 1
    aggregated_status.overall_progress = calculate_progress_percentage(1, 3)

    print(f"  Video 1 completed in {video_results[0].processing_time_seconds}s")
    print(f"  Overall progress: {aggregated_status.overall_progress:.1f}%")

    return video_results, aggregated_status


def demo_parallel_processing():
    """Demonstrate parallel processing orchestration."""
    print("\n=== Demo: Parallel Processing ===")

    # Create parallel processing configuration
    parallel_config = create_parallel_processing_config(
        max_workers=3, chunk_size=2, worker_timeout=600
    )

    print("Parallel processing config:")
    print(f"  Max Workers: {parallel_config.max_workers}")
    print(f"  Chunk Size: {parallel_config.chunk_size}")
    print(f"  Worker Timeout: {parallel_config.worker_timeout}s")

    # Create workers
    workers = create_batch_workers(
        worker_count=parallel_config.max_workers, worker_id_prefix="video_processor"
    )

    print(f"\nCreated {len(workers)} workers:")
    for worker in workers:
        print(f"  {worker.worker_id} - {worker.status}")

    # Create orchestration
    orchestration_id = generate_orchestration_id()
    request_id = generate_request_id()

    orchestration = create_parallel_processing_orchestration(
        orchestration_id=orchestration_id, request_id=request_id, config=parallel_config
    )

    orchestration.workers = workers
    orchestration.active_workers = len(workers)
    orchestration.total_tasks = 6
    orchestration.pending_tasks = 6
    orchestration.status = "running"
    orchestration.start_time = datetime.now()

    print("\nOrchestration started:")
    print(f"  ID: {orchestration.orchestration_id}")
    print(f"  Active Workers: {orchestration.active_workers}")
    print(f"  Total Tasks: {orchestration.total_tasks}")
    print(f"  Status: {orchestration.status}")

    # Simulate task completion
    orchestration.completed_tasks = 2
    orchestration.pending_tasks = 4
    throughput = calculate_throughput(
        completed=orchestration.completed_tasks, start_time=orchestration.start_time
    )
    orchestration.throughput_tasks_per_minute = throughput

    print("\nProgress update:")
    print(f"  Completed Tasks: {orchestration.completed_tasks}")
    print(f"  Throughput: {orchestration.throughput_tasks_per_minute:.1f} tasks/min")

    return orchestration


def demo_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n=== Demo: Error Handling ===")

    request_id = generate_request_id()
    video_id = "dQw4w9WgXcQ"

    # Create an error
    error = create_processing_error(
        request_id=request_id,
        error_type=ErrorType.NETWORK_ERROR,
        error_message="Connection timeout while fetching video metadata",
        processing_stage="metadata_extraction",
        video_id=video_id,
    )

    print("Processing error occurred:")
    print(f"  Error ID: {error.error_id}")
    print(f"  Type: {error.error_type}")
    print(f"  Message: {error.error_message}")
    print(f"  Stage: {error.processing_stage}")
    print(f"  Can Retry: {error.can_retry}")

    # Create retry configuration
    retry_config = create_retry_config(
        max_retries=3, initial_delay=60, backoff_factor=2.0
    )

    print("\nRetry configuration:")
    print(f"  Max Retries: {retry_config.max_retries}")
    print(f"  Initial Delay: {retry_config.initial_delay}s")
    print(f"  Backoff Factor: {retry_config.backoff_factor}")

    # Create retry task
    task_id = generate_request_id()
    retry_task = create_retry_task(
        task_id=task_id,
        request_id=request_id,
        processing_stage="metadata_extraction",
        original_error=error,
        video_id=video_id,
    )

    print("\nRetry task created:")
    print(f"  Task ID: {retry_task.task_id}")
    print(f"  Next Retry: {retry_task.next_retry_at}")
    print(f"  Status: {retry_task.status}")

    return error, retry_config, retry_task


def demo_full_integration():
    """Demonstrate full search-to-process integration."""
    print("\n=== Demo: Full Integration ===")

    session_id = generate_session_id()

    # Create main integration request
    main_request = create_search_to_process_request(
        session_id=session_id,
        query="Python programming tutorials for beginners",
        max_videos=3,
    )

    print("Main integration request:")
    print(f"  Session ID: {main_request.session_id}")
    print(f"  Query: {main_request.query}")
    print(f"  Max Videos: {main_request.max_videos}")
    print(f"  Priority: {main_request.priority}")

    # Create main integration response
    request_id = generate_request_id()
    main_response = create_search_to_process_response(
        request_id=request_id,
        session_id=session_id,
        status=SearchProcessingStatus.PROCESSING_VIDEOS,
    )

    # Update progress information
    main_response.progress.query = main_request.query
    main_response.progress.total_videos = 3
    main_response.progress.processed_videos = 1
    main_response.progress.successful_videos = 1
    main_response.progress.video_urls = [
        "https://www.youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=def456",
        "https://www.youtube.com/watch?v=ghi789",
    ]

    main_response.aggregated_status.total_videos = 3
    main_response.aggregated_status.completed_count = 1
    main_response.aggregated_status.processing_count = 1
    main_response.aggregated_status.pending_count = 1
    main_response.aggregated_status.overall_progress = calculate_progress_percentage(
        1, 3
    )

    print("\nMain integration response:")
    print(f"  Request ID: {main_response.request_id}")
    print(f"  Status: {main_response.status}")
    print(f"  Total Videos: {main_response.aggregated_status.total_videos}")
    print(f"  Progress: {main_response.aggregated_status.overall_progress:.1f}%")

    return main_request, main_response


def demo_status_checks():
    """Demonstrate status checking functionality."""
    print("\n=== Demo: Status Checks ===")

    request_id = generate_request_id()

    # Create status check request
    status_request = create_status_check_request(
        request_id=request_id,
        include_details=True,
        include_errors=True,
        include_performance=True,
    )

    print("Status check request:")
    print(f"  Request ID: {status_request.request_id}")
    print(f"  Include Details: {status_request.include_details}")
    print(f"  Include Errors: {status_request.include_errors}")
    print(f"  Include Performance: {status_request.include_performance}")

    # Create status check response
    status_response = create_status_check_response(
        request_id=request_id,
        status=SearchProcessingStatus.PROCESSING_VIDEOS,
        progress_percentage=66.7,
        total_videos=3,
        completed_videos=2,
        failed_videos=0,
    )

    # Add estimated completion
    start_time = datetime.now() - timedelta(minutes=5)
    status_response.estimated_completion = estimate_completion_time(
        completed=2, total=3, start_time=start_time
    )

    print("\nStatus check response:")
    print(f"  Status: {status_response.status}")
    print(f"  Progress: {status_response.progress_percentage:.1f}%")
    print(
        f"  Completed: {status_response.completed_videos}/{status_response.total_videos}"
    )
    print(f"  Estimated Completion: {status_response.estimated_completion}")

    return status_request, status_response


def demo_serializer_integration():
    """Demonstrate Django REST Framework serializer integration."""
    print("\n=== Demo: Serializer Integration ===")

    # Create a search request
    session_id = generate_session_id()
    search_request = create_search_progress_request(
        session_id=session_id, query="AI and machine learning", max_videos=5
    )

    print("Original Pydantic model:")
    print(f"  Query: {search_request.query}")
    print(f"  Max Videos: {search_request.max_videos}")

    # Convert to dictionary for DRF
    request_dict = pydantic_to_dict(search_request)
    print(f"\nConverted to dictionary: {len(request_dict)} fields")

    # Use DRF serializer
    serializer = SearchProgressRequestSerializer(data=request_dict)
    if serializer.is_valid():
        print("✓ DRF serializer validation passed")

        # Convert back to Pydantic
        pydantic_model = serializer.to_pydantic()
        print(f"✓ Converted back to Pydantic: {pydantic_model.query}")
    else:
        print(f"✗ DRF serializer validation failed: {serializer.errors}")

    return search_request, serializer


def main():
    """Run all demonstrations."""
    print("Search-to-Process Integration Models Demo")
    print("=" * 50)

    try:
        # Run all demos
        demo_basic_search_request()
        demo_video_processing_workflow()
        demo_parallel_processing()
        demo_error_handling()
        demo_full_integration()
        demo_status_checks()
        demo_serializer_integration()

        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Search progress tracking")
        print("  • Video processing status management")
        print("  • Parallel processing orchestration")
        print("  • Error handling and retry logic")
        print("  • Full integration workflow")
        print("  • Status checking capabilities")
        print("  • Django REST Framework serialization")

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
