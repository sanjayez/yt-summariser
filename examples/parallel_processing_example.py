"""
Example usage of the parallel processing system for search-to-process integration

This example demonstrates how to:
1. Create a search request
2. Process it to find videos
3. Start parallel video processing
4. Monitor progress and get results
"""

import json
import os
import sys
import time

import django
from django.db import transaction

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_summariser.settings")
django.setup()

from topic.models import SearchRequest, SearchSession
from topic.parallel_tasks import get_search_processing_status, process_search_results
from topic.tasks import process_search_query, process_search_with_videos
from video_processor.processors.search_adapter import get_search_video_results


def create_search_request(query: str, user_ip: str = "127.0.0.1"):
    """Create a search session and request"""
    with transaction.atomic():
        session = SearchSession.objects.create(user_ip=user_ip, status="processing")

        search_request = SearchRequest.objects.create(
            search_session=session, original_query=query, status="processing"
        )

        return session, search_request


def example_search_only():
    """Example 1: Search only (no video processing)"""
    print("=== Example 1: Search Only ===")

    # Create search request
    session, search_request = create_search_request("machine learning basics")
    print(f"Created search request: {search_request.request_id}")

    # Start search task
    search_task = process_search_query.apply_async(
        args=[str(search_request.request_id)]
    )
    print(f"Started search task: {search_task.id}")

    # Wait for completion
    result = search_task.get()
    print(f"Search completed: {json.dumps(result, indent=2)}")

    return search_request


def example_parallel_processing():
    """Example 2: Search followed by parallel video processing"""
    print("\n=== Example 2: Parallel Processing ===")

    # Create search request
    session, search_request = create_search_request("python tutorial")
    print(f"Created search request: {search_request.request_id}")

    # First complete search
    search_task = process_search_query.apply_async(
        args=[str(search_request.request_id)]
    )
    search_result = search_task.get()

    if search_result["status"] != "success":
        print(f"Search failed: {search_result}")
        return

    print(f"Search found {search_result['total_videos']} videos")

    # Start parallel video processing
    processing_task = process_search_results.apply_async(
        args=[str(search_request.request_id)]
    )
    print(f"Started parallel processing task: {processing_task.id}")

    # Monitor progress
    monitor_progress(search_request, timeout=300)

    return search_request


def example_integrated_workflow():
    """Example 3: Integrated search and video processing"""
    print("\n=== Example 3: Integrated Workflow ===")

    # Create search request
    session, search_request = create_search_request("data science tutorial")
    print(f"Created search request: {search_request.request_id}")

    # Start integrated workflow
    integrated_task = process_search_with_videos.apply_async(
        args=[str(search_request.request_id)], kwargs={"start_video_processing": True}
    )
    print(f"Started integrated workflow task: {integrated_task.id}")

    # Wait for search to complete
    result = integrated_task.get()
    print(f"Integrated task result: {json.dumps(result, indent=2)}")

    # Monitor video processing progress
    if result.get("status") == "processing_videos":
        print("Video processing started, monitoring progress...")
        monitor_progress(search_request, timeout=300)

    return search_request


def monitor_progress(search_request, timeout=300):
    """Monitor processing progress"""
    print(f"Monitoring progress for search request: {search_request.request_id}")

    start_time = time.time()

    while time.time() - start_time < timeout:
        # Get status
        status_task = get_search_processing_status.apply_async(
            args=[str(search_request.request_id)]
        )
        status = status_task.get()

        if status.get("status") == "success":
            processing_stats = status.get("processing_stats", {})
            print(f"Processing stats: {json.dumps(processing_stats, indent=2)}")

            # Check if all processing is complete
            if processing_stats.get("processing_videos", 0) == 0:
                print("All video processing completed!")
                break
        else:
            print(f"Status check: {status}")

        time.sleep(10)

    # Get final results
    get_final_results(search_request)


def get_final_results(search_request):
    """Get and display final results"""
    print(f"\n=== Final Results for {search_request.request_id} ===")

    # Refresh search request
    search_request.refresh_from_db()
    print(f"Search Status: {search_request.status}")
    print(f"Total Videos: {search_request.total_videos}")
    print(f"Video URLs: {search_request.video_urls}")

    # Get video processing results
    video_results_task = get_search_video_results.apply_async(
        args=[str(search_request.request_id)]
    )
    video_results = video_results_task.get()

    if video_results.get("status") == "success":
        summary = video_results.get("summary", {})
        print("Video Processing Summary:")
        print(f"  Total Videos: {summary.get('total_videos', 0)}")
        print(f"  Successful: {summary.get('successful_videos', 0)}")
        print(f"  Failed: {summary.get('failed_videos', 0)}")
        print(f"  Processing: {summary.get('processing_videos', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")

        # Show details of processed videos
        video_results_list = video_results.get("video_results", [])
        for i, video in enumerate(video_results_list[:3]):  # Show first 3
            print(f"\n  Video {i + 1}:")
            print(f"    URL: {video.get('url', 'N/A')}")
            print(f"    Status: {video.get('status', 'N/A')}")
            if video.get("video_metadata"):
                metadata = video["video_metadata"]
                print(f"    Title: {metadata.get('title', 'N/A')[:50]}...")
                print(f"    Duration: {metadata.get('duration', 'N/A')} seconds")
                print(f"    Channel: {metadata.get('channel_name', 'N/A')}")
    else:
        print(f"Failed to get video results: {video_results}")


if __name__ == "__main__":
    print("Parallel Processing Integration Examples")
    print("=" * 50)

    try:
        # Run examples
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            if mode == "search":
                example_search_only()
            elif mode == "parallel":
                example_parallel_processing()
            elif mode == "integrated":
                example_integrated_workflow()
            else:
                print("Available modes: search, parallel, integrated")
        else:
            print("Running all examples...")
            example_search_only()
            example_parallel_processing()
            example_integrated_workflow()

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
