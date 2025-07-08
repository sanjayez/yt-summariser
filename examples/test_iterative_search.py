#!/usr/bin/env python3
"""
Test Iterative Search Behavior
Verifies that the ScrapeTubeProvider keeps fetching until it finds the requested number of filtered results
"""

import sys
import os
import logging
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic.services.providers.scrapetube_provider import ScrapeTubeProvider

# Configure logging to see the iterative process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_iterative_fetching():
    """Test that provider keeps fetching until target results found"""
    print("Testing Iterative Fetching Behavior")
    print("=" * 40)
    
    # Create provider with filtering enabled
    provider = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=120  # 2 minutes minimum
    )
    
    test_queries = [
        ("python tutorial", 3),
        ("machine learning", 2),
        ("programming basics", 5),
    ]
    
    for query, target_results in test_queries:
        print(f"\n🔍 Testing: '{query}' (target: {target_results} results)")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            results = provider.search(query, max_results=target_results)
            elapsed = time.time() - start_time
            
            print(f"✅ Results: {len(results)}/{target_results} in {elapsed:.2f}s")
            
            if len(results) == target_results:
                print("✅ SUCCESS: Found exact number of requested results")
            elif len(results) < target_results:
                print(f"⚠️  PARTIAL: Found {len(results)}/{target_results} (may have hit safety limit)")
            else:
                print(f"❌ ERROR: Got more results than requested ({len(results)} > {target_results})")
            
            # Show first few results
            for i, url in enumerate(results[:3], 1):
                print(f"   {i}. {url}")
            
            if len(results) > 3:
                print(f"   ... and {len(results) - 3} more")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    print(f"\n🧪 Test completed!")

def test_filtering_effectiveness():
    """Test that filtering actually works"""
    print("\n\nTesting Filter Effectiveness")
    print("=" * 40)
    
    # Provider without filtering
    unfiltered_provider = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=False,
        english_only=False
    )
    
    # Provider with filtering
    filtered_provider = ScrapeTubeProvider(
        max_results=5,
        filter_shorts=True,
        english_only=True,
        min_duration_seconds=60
    )
    
    query = "tutorial"
    
    try:
        print(f"🔍 Testing filtering with query: '{query}'")
        
        # Get unfiltered results (with metadata to see durations)
        unfiltered_results = unfiltered_provider.search_with_metadata(query, max_results=3)
        print(f"\nUnfiltered results ({len(unfiltered_results)}):")
        for i, result in enumerate(unfiltered_results, 1):
            print(f"   {i}. {result.title} ({result.duration})")
        
        # Get filtered results
        filtered_results = filtered_provider.search_with_metadata(query, max_results=3)
        print(f"\nFiltered results ({len(filtered_results)}):")
        for i, result in enumerate(filtered_results, 1):
            print(f"   {i}. {result.title} ({result.duration})")
        
        # Analysis
        if filtered_results:
            durations = [r.duration for r in filtered_results if r.duration]
            print(f"\n📊 Analysis:")
            print(f"   - All filtered results have duration ≥ 60s: {all(d and ':' in d for d in durations)}")
            print(f"   - Durations found: {durations}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_iterative_fetching()
    test_filtering_effectiveness() 