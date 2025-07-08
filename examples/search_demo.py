#!/usr/bin/env python3
"""
YouTube Search Demo
Demonstrates how to use the ScrapeTubeProvider and YouTubeSearchService
"""

import sys
import os
import logging
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from topic.services.search_service import (
    YouTubeSearchService, 
    SearchRequest, 
    create_youtube_search_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_scrapetube_provider():
    """Demonstrate ScrapeTubeProvider usage"""
    print("=== ScrapeTubeProvider Demo ===")
    
    # Create provider instance with filtering enabled
    provider = ScrapeTubeProvider(
        max_results=3,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=60
    )
    
    # Show provider info
    print("Provider Info:")
    for key, value in provider.get_provider_info().items():
        print(f"  {key}: {value}")
    
    # Test query validation
    test_queries = ["python tutorial", "", "a", "x" * 200]
    print("\nQuery Validation Tests:")
    for query in test_queries:
        is_valid = provider.validate_query(query)
        print(f"  '{query[:50]}...': {is_valid}")
    
    # Perform a search
    search_query = "python programming tutorial"
    print(f"\nSearching for: '{search_query}' (with English-only and shorts filtering)")
    
    try:
        # Simple search
        urls = provider.search(search_query, max_results=3)
        print(f"Found {len(urls)} filtered URLs:")
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
        
        # Search with metadata
        print(f"\nSearching with metadata for: '{search_query}'")
        results = provider.search_with_metadata(search_query, max_results=3)
        print(f"Found {len(results)} filtered results with metadata:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.title}")
            print(f"     URL: {result.url}")
            print(f"     Channel: {result.channel_name}")
            print(f"     Duration: {result.duration}")
            print(f"     Views: {result.view_count}")
            print(f"     Upload Date: {result.upload_date}")
            print()
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        logger.error(f"Search failed: {str(e)}")

def demo_filtering_comparison():
    """Demonstrate filtering vs non-filtering comparison"""
    print("\n=== Filtering Comparison Demo ===")
    
    # Provider without filtering
    provider_unfiltered = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=False,
        english_only=False
    )
    
    # Provider with filtering
    provider_filtered = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=60
    )
    
    search_query = "tutorial"
    
    try:
        print("Without filtering:")
        unfiltered_results = provider_unfiltered.search_with_metadata(search_query, max_results=3)
        for i, result in enumerate(unfiltered_results, 1):
            print(f"  {i}. {result.title} ({result.duration})")
        
        print("\nWith filtering (English-only, no shorts):")
        filtered_results = provider_filtered.search_with_metadata(search_query, max_results=3)
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. {result.title} ({result.duration})")
            
    except Exception as e:
        print(f"Error during filtering comparison: {str(e)}")
        logger.error(f"Filtering comparison failed: {str(e)}")

def demo_youtube_search_service():
    """Demonstrate YouTubeSearchService usage"""
    print("\n=== YouTubeSearchService Demo ===")
    
    # Create service with default provider
    service = create_youtube_search_service()
    
    # Show service info
    print("Service Info:")
    for key, value in service.get_service_info().items():
        print(f"  {key}: {value}")
    
    # Test health check
    print(f"\nHealth Check: {service.health_check()}")
    
    # Simple search
    search_query = "django tutorial"
    print(f"\nSimple search for: '{search_query}'")
    
    try:
        urls = service.search_simple(search_query, max_results=3)
        print(f"Found {len(urls)} URLs:")
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
            video_id = service.get_video_id_from_url(url)
            print(f"     Video ID: {video_id}")
        
    except Exception as e:
        print(f"Error during simple search: {str(e)}")
        logger.error(f"Simple search failed: {str(e)}")
    
    # Search with SearchRequest
    print(f"\nAdvanced search with SearchRequest:")
    
    try:
        request = SearchRequest(
            query="machine learning basics",
            max_results=2,
            include_metadata=True
        )
        
        response = service.search(request)
        print(f"Query: {response.query}")
        print(f"Total Results: {response.total_results}")
        print(f"Search Time: {response.search_time_ms:.2f}ms")
        print(f"Results:")
        
        for i, url in enumerate(response.results, 1):
            print(f"  {i}. {url}")
        
        if response.metadata:
            print(f"Metadata:")
            for i, result in enumerate(response.metadata, 1):
                print(f"  {i}. {result.title}")
                print(f"     Channel: {result.channel_name}")
                print(f"     Duration: {result.duration}")
        
    except Exception as e:
        print(f"Error during advanced search: {str(e)}")
        logger.error(f"Advanced search failed: {str(e)}")

def demo_custom_provider():
    """Demonstrate custom provider configuration"""
    print("\n=== Custom Provider Demo ===")
    
    # Create custom provider with strict filtering
    custom_provider = ScrapeTubeProvider(
        max_results=5, 
        timeout=45,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=300  # 5 minutes minimum for educational content
    )
    
    # Create service with custom provider
    service = YouTubeSearchService(custom_provider)
    
    print("Custom Provider Info:")
    provider_info = service.get_provider_info()
    for key, value in provider_info.items():
        print(f"  {key}: {value}")
    
    # Test with custom provider
    try:
        results = service.search_with_metadata("machine learning course", max_results=2)
        print(f"\nFound {len(results)} long-form educational videos:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.title}")
            print(f"     Duration: {result.duration}")
            print(f"     URL: {result.url}")
        
    except Exception as e:
        print(f"Error with custom provider: {str(e)}")
        logger.error(f"Custom provider search failed: {str(e)}")

def demo_iterative_fetching():
    """Demonstrate iterative fetching until target results found"""
    print("\n=== Iterative Fetching Demo ===")
    
    # Create provider with strict filtering to demonstrate the iterative approach
    provider = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=180  # 3 minutes - quite strict to show iteration
    )
    
    search_query = "tutorial"  # Broad query that will have many results but many filtered out
    
    print(f"Searching for: '{search_query}'")
    print(f"Target: 5 English videos longer than 3 minutes")
    print("This will demonstrate iterative fetching until we find enough filtered results...")
    
    try:
        # Enable debug logging to see the filtering process
        import logging
        logging.getLogger('topic.services.providers.scrapetube_provider').setLevel(logging.INFO)
        
        start_time = time.time()
        results = provider.search_with_metadata(search_query, max_results=5)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Successfully found {len(results)} filtered results in {elapsed:.2f}s:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.title}")
            print(f"     Duration: {result.duration}")
            print(f"     Channel: {result.channel_name}")
            print()
        
        if len(results) < 5:
            print(f"⚠️  Note: Only found {len(results)}/5 results - may have hit the 100 video safety limit")
            print("    This is normal for very strict filters or niche queries")
        
    except Exception as e:
        print(f"Error during iterative fetching demo: {str(e)}")
        logger.error(f"Iterative fetching demo failed: {str(e)}")

def main():
    """Main demo function"""
    print("YouTube Search Demo with Filtering")
    print("=" * 50)
    
    try:
        # Demo ScrapeTubeProvider with filtering
        demo_scrapetube_provider()
        
        # Demo filtering comparison
        demo_filtering_comparison()
        
        # NEW: Demo iterative fetching
        demo_iterative_fetching()
        
        # Demo YouTubeSearchService
        demo_youtube_search_service()
        
        # Demo custom provider with strict filtering
        demo_custom_provider()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")
    
    print("\nDemo completed!")
    print("\nFiltering Features Demonstrated:")
    print("✓ English-only video filtering")
    print("✓ YouTube Shorts filtering by duration")
    print("✓ Customizable duration thresholds")
    print("✓ Comparison between filtered and unfiltered results")
    print("✓ Iterative fetching until target results found")

if __name__ == "__main__":
    main()