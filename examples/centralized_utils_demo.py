#!/usr/bin/env python3
"""
Comprehensive demo and test script for centralized utilities.

This script demonstrates all the features of the new core utilities:
- Logging utilities
- Timing utilities  
- Monitoring utilities
- Exception handling utilities
- Resilience utilities

Run this script to verify all utilities are working correctly.
"""

import asyncio
import time
import random
import sys
import os
from typing import Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import all centralized telemetry utilities
from telemetry.utils import (
    # Logging
    get_logger, setup_logging, log_function_call,
    
    # Timing
    timed_operation, TimingContext, measure_time, PerformanceTimer,
    
    # Monitoring
    PerformanceMonitor, log_memory_usage, log_system_stats, monitor_resources,
    
    # Exception handling
    handle_exceptions, with_error_context, VideoProcessingError, ExternalServiceError,
    
    # Resilience
    circuit_breaker, retry_with_backoff, timeout_handler, resilient_external_call
)

# Set up logging for the demo
setup_logging(level="INFO")
logger = get_logger(__name__)

class UtilitiesDemo:
    """Demo class showcasing all centralized utilities."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        
    def demo_logging_utilities(self):
        """Demonstrate logging utilities."""
        logger.info("=== Logging Utilities Demo ===")
        
        # Different log levels
        test_logger = get_logger("test_module")
        test_logger.debug("Debug message (may not show if level is INFO)")
        test_logger.info("Info message - normal operation")
        test_logger.warning("Warning message - something needs attention")
        test_logger.error("Error message - something went wrong")
        
        # Function call logging
        import logging
        @log_function_call(level=logging.INFO)
        def sample_function(name: str, value: int = 42) -> str:
            """Sample function for logging demo."""
            time.sleep(0.1)  # Simulate work
            return f"Processed {name} with value {value}"
        
        result = sample_function("demo_data", 100)
        logger.info(f"Function result: {result}")
        
    def demo_timing_utilities(self):
        """Demonstrate timing utilities."""
        logger.info("=== Timing Utilities Demo ===")
        
        # Timing decorator
        @timed_operation()
        def slow_operation():
            """Simulate a slow operation."""
            time.sleep(0.5)
            return "Operation completed"
        
        result = slow_operation()
        logger.info(f"Slow operation result: {result}")
        
        # Timing context manager
        with TimingContext("database_query"):
            time.sleep(0.2)  # Simulate database query
            logger.info("Database query executed")
        
        # Manual timing
        result, elapsed = measure_time(lambda: sum(range(1000000)))
        logger.info(f"Manual timing: computed sum in {elapsed:.2f}ms")
        
        # Performance timer for multi-stage operations
        timer = PerformanceTimer()
        timer.start_request("multi_stage_demo")
        
        with timer.time_stage("stage_1"):
            time.sleep(0.1)
            
        with timer.time_stage("stage_2"):
            time.sleep(0.2)
            
        with timer.time_stage("stage_3"):
            time.sleep(0.15)
            
        timer.log_request_summary("Multi-stage operation demo")
        
    def demo_monitoring_utilities(self):
        """Demonstrate monitoring utilities."""
        logger.info("=== Monitoring Utilities Demo ===")
        
        # Basic monitoring
        log_memory_usage("Demo start")
        log_system_stats()
        
        # Monitor with context manager
        with self.monitor.monitor("memory_intensive_task") as metrics:
            # Simulate memory-intensive work
            data = [i ** 2 for i in range(100000)]
            metrics.additional_metrics['items_processed'] = len(data)
            del data  # Clean up
            
        # Resource monitoring decorator
        @monitor_resources(threshold_memory_percent=50.0)
        def memory_task():
            """Task that uses memory."""
            return [random.random() for _ in range(50000)]
        
        result = memory_task()
        logger.info(f"Generated {len(result)} random numbers")
        
        log_memory_usage("Demo end")
        
    def demo_exception_utilities(self):
        """Demonstrate exception handling utilities."""
        logger.info("=== Exception Handling Demo ===")
        
        # Custom exceptions
        try:
            raise VideoProcessingError("Failed to extract video metadata", 
                                     details={"video_id": "demo123", "reason": "network_timeout"})
        except VideoProcessingError as e:
            logger.error(f"Caught VideoProcessingError: {e}")
            
        # Exception handling decorator - graceful handling
        @handle_exceptions(reraise=False, default_return="fallback_result")
        def risky_operation(should_fail: bool = False):
            """Operation that might fail."""
            if should_fail:
                raise ValueError("Something went wrong!")
            return "success_result"
        
        # This will succeed
        result1 = risky_operation(False)
        logger.info(f"Risky operation result: {result1}")
        
        # This will fail gracefully
        result2 = risky_operation(True)
        logger.info(f"Failed operation result: {result2}")
        
        # Error context decorator
        @with_error_context("Processing user data")
        def process_user_data(user_id: str):
            """Process user data with context."""
            if user_id == "invalid":
                raise ValueError("Invalid user ID")
            return f"Processed user {user_id}"
        
        try:
            process_user_data("invalid")
        except ValueError as e:
            logger.error(f"Error with context: {e}")
            
    def demo_resilience_utilities(self):
        """Demonstrate resilience utilities."""
        logger.info("=== Resilience Utilities Demo ===")
        
        # Circuit breaker demo
        @circuit_breaker(name="demo_service", failure_threshold=2, recovery_timeout=5.0)
        def unreliable_service(should_fail: bool = False):
            """Simulate an unreliable external service."""
            if should_fail:
                raise ExternalServiceError("Service temporarily unavailable")
            return "Service response"
        
        # Test successful calls
        for i in range(3):
            try:
                result = unreliable_service(False)
                logger.info(f"Service call {i+1}: {result}")
            except Exception as e:
                logger.error(f"Service call {i+1} failed: {e}")
        
        # Test failing calls (will trigger circuit breaker)
        for i in range(4):
            try:
                result = unreliable_service(True)
                logger.info(f"Failing call {i+1}: {result}")
            except Exception as e:
                logger.error(f"Failing call {i+1}: {e}")
        
        # Retry with backoff demo
        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        def flaky_operation(attempt_count: list):
            """Operation that fails first few times."""
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError(f"Connection failed (attempt {attempt_count[0]})")
            return f"Success on attempt {attempt_count[0]}"
        
        attempt_counter = [0]
        try:
            result = flaky_operation(attempt_counter)
            logger.info(f"Flaky operation result: {result}")
        except Exception as e:
            logger.error(f"Flaky operation finally failed: {e}")
        
        # Timeout handler demo
        @timeout_handler(timeout_seconds=1.0)
        def slow_operation_with_timeout():
            """Operation that might take too long."""
            time.sleep(0.5)  # This should succeed
            return "Completed within timeout"
        
        try:
            result = slow_operation_with_timeout()
            logger.info(f"Timeout demo result: {result}")
        except Exception as e:
            logger.error(f"Timeout demo failed: {e}")
            
    async def demo_async_features(self):
        """Demonstrate async support in utilities."""
        logger.info("=== Async Features Demo ===")
        
        # Async timing
        @timed_operation
        async def async_operation():
            """Async operation with timing."""
            await asyncio.sleep(0.3)
            return "Async operation completed"
        
        result = await async_operation()
        logger.info(f"Async timing result: {result}")
        
        # Async exception handling
        @handle_exceptions(reraise=False, default_return="async_fallback")
        async def async_risky_operation(should_fail: bool = False):
            """Async operation that might fail."""
            await asyncio.sleep(0.1)
            if should_fail:
                raise ValueError("Async operation failed!")
            return "async_success"
        
        result1 = await async_risky_operation(False)
        logger.info(f"Async success: {result1}")
        
        result2 = await async_risky_operation(True)
        logger.info(f"Async fallback: {result2}")
        
        # Async resilience
        @resilient_external_call("async_service", max_attempts=2, timeout_seconds=2.0)
        async def async_external_call():
            """Async external service call."""
            await asyncio.sleep(0.2)
            return "Async external service response"
        
        try:
            result = await async_external_call()
            logger.info(f"Async resilience result: {result}")
        except Exception as e:
            logger.error(f"Async resilience failed: {e}")
            
    def run_all_demos(self):
        """Run all utility demonstrations."""
        logger.info("🎯 Starting Centralized Telemetry Demo")
        logger.info("=" * 50)
        
        try:
            # Run sync demos
            self.demo_logging_utilities()
            self.demo_timing_utilities()
            self.demo_monitoring_utilities()
            self.demo_exception_utilities()
            self.demo_resilience_utilities()
            
            # Run async demos
            asyncio.run(self.demo_async_features())
            
            logger.info("=" * 50)
            logger.info("✅ All demos completed successfully!")
            
            # Final system stats
            log_system_stats()
            log_memory_usage("Demo completion")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}", exc_info=True)
            raise

def main():
    """Main function to run the demo."""
    try:
        demo = UtilitiesDemo()
        demo.run_all_demos()
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())