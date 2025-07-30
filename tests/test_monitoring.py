#!/usr/bin/env python3
"""
Test script for monitoring utilities.
This can be run to verify the monitoring functionality.
"""

import time
import random
from core.utils import (
    PerformanceMonitor,
    log_memory_usage,
    log_system_stats,
    monitor_resources,
    profile_memory,
    memory_tracker,
    log_function_resources,
    get_process_info,
    get_global_monitor,
    setup_logging
)

# Setup logging for the test
setup_logging(level="DEBUG")


def test_basic_monitoring():
    """Test basic monitoring functions."""
    print("\n=== Testing Basic Monitoring Functions ===")
    
    # Test memory usage logging
    print("\n1. Testing log_memory_usage:")
    stats = log_memory_usage("Test Label")
    print(f"Memory stats: RSS={stats['rss_mb']:.2f}MB, Percent={stats['percent']:.1f}%")
    
    # Test system stats logging
    print("\n2. Testing log_system_stats:")
    sys_stats = log_system_stats(include_disk=True)
    print(f"CPU: {sys_stats['cpu']['total_percent']:.1f}%, Memory: {sys_stats['memory']['percent']:.1f}%")
    
    # Test process info
    print("\n3. Testing get_process_info:")
    proc_info = get_process_info()
    print(f"Process: PID={proc_info['pid']}, Threads={proc_info['num_threads']}")


def test_performance_monitor():
    """Test PerformanceMonitor class."""
    print("\n=== Testing PerformanceMonitor ===")
    
    monitor = PerformanceMonitor("TestMonitor")
    
    # Test context manager monitoring
    print("\n1. Testing monitor context manager:")
    with monitor.monitor("test_operation") as metrics:
        # Simulate some work
        time.sleep(0.5)
        data = [random.random() for _ in range(1000000)]
        metrics.additional_metrics['items_processed'] = len(data)
    
    # Test multiple operations
    print("\n2. Testing multiple operations:")
    for i in range(3):
        with monitor.monitor("repeated_operation"):
            time.sleep(0.1 + random.random() * 0.2)
    
    # Log summary
    print("\n3. Performance summary:")
    monitor.log_summary()
    
    # Get metrics summary
    summary = monitor.get_metrics_summary()
    print(f"\nDetailed summary: {summary}")


@monitor_resources(threshold_memory_percent=50.0, log_interval=0.5)
def test_resource_monitoring_decorator():
    """Test resource monitoring decorator."""
    print("\n=== Testing @monitor_resources decorator ===")
    
    # Simulate memory-intensive operation
    data = []
    for i in range(5):
        print(f"Iteration {i+1}/5")
        data.append([random.random() for _ in range(500000)])
        time.sleep(0.3)
    
    return len(data)


@profile_memory(log_top_consumers=5)
def test_memory_profiling_decorator():
    """Test memory profiling decorator."""
    print("\n=== Testing @profile_memory decorator ===")
    
    # Create and manipulate large data structures
    large_list = list(range(1000000))
    large_dict = {i: f"value_{i}" for i in range(100000)}
    
    # Simulate processing
    result = sum(large_list[:10000])
    
    return result


@log_function_resources
def test_simple_resource_logging():
    """Test simple resource logging decorator."""
    print("\n=== Testing @log_function_resources decorator ===")
    
    # Some computation
    data = [i ** 2 for i in range(100000)]
    time.sleep(0.2)
    
    return sum(data)


def test_memory_tracker():
    """Test memory tracker context manager."""
    print("\n=== Testing memory_tracker context manager ===")
    
    with memory_tracker("data_allocation") as mem_stats:
        # Allocate memory
        data = [random.random() for _ in range(2000000)]
        
        # Process data
        processed = [x * 2 for x in data[:100000]]
    
    print(f"\nMemory tracker results:")
    print(f"Memory change: {mem_stats['memory_change_mb']:.2f}MB")
    print(f"Duration: {mem_stats['duration_seconds']:.3f}s")


def test_global_monitor():
    """Test global monitor instance."""
    print("\n=== Testing Global Monitor ===")
    
    monitor = get_global_monitor()
    
    # Use global monitor for tracking
    with monitor.monitor("global_operation_1"):
        time.sleep(0.2)
    
    with monitor.monitor("global_operation_2"):
        time.sleep(0.3)
    
    # Get summary from global monitor
    print("\nGlobal monitor summary:")
    monitor.log_summary()


def test_alert_callback():
    """Test alert callback functionality."""
    print("\n=== Testing Alert Callback ===")
    
    alerts = []
    
    def alert_handler(alert_info):
        alerts.append(alert_info)
        print(f"ALERT: {alert_info['type']} threshold exceeded in {alert_info['function']}")
    
    @monitor_resources(
        threshold_memory_percent=30.0,  # Low threshold to trigger
        threshold_cpu_percent=50.0,
        alert_callback=alert_handler
    )
    def trigger_alerts():
        # Create some load
        data = [random.random() for _ in range(5000000)]
        time.sleep(0.5)
        return len(data)
    
    result = trigger_alerts()
    print(f"\nProcessed {result} items")
    print(f"Alerts triggered: {len(alerts)}")


def main():
    """Run all tests."""
    print("YouTube Summarizer - Monitoring Utilities Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_monitoring,
        test_performance_monitor,
        test_resource_monitoring_decorator,
        test_memory_profiling_decorator,
        test_simple_resource_logging,
        test_memory_tracker,
        test_global_monitor,
        test_alert_callback
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"\n✓ {test_func.__name__} completed successfully")
        except Exception as e:
            print(f"\n✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-" * 60)
    
    print("\n=== All tests completed ===")
    
    # Final system stats
    print("\nFinal system statistics:")
    log_system_stats()


if __name__ == "__main__":
    main()