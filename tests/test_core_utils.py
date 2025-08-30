#!/usr/bin/env python3
"""
Simple test script to verify core utilities are working.
"""

import os
import sys
import time

# Add project root to Python path
sys.path.insert(0, os.getcwd())


def test_logging():
    """Test logging utilities."""
    print("Testing logging utilities...")
    from telemetry.utils import get_logger

    logger = get_logger("test_logging")
    logger.info("✅ Logging utilities working!")
    return True


def test_timing():
    """Test timing utilities."""
    print("Testing timing utilities...")
    from telemetry.utils import TimingContext

    with TimingContext("test_operation"):
        time.sleep(0.1)

    print("✅ Timing utilities working!")
    return True


def test_monitoring():
    """Test monitoring utilities."""
    print("Testing monitoring utilities...")
    from telemetry.utils import log_memory_usage

    log_memory_usage("test")
    print("✅ Monitoring utilities working!")
    return True


def test_exceptions():
    """Test exception utilities."""
    print("Testing exception utilities...")
    from telemetry.utils import VideoProcessingError

    try:
        raise VideoProcessingError("Test error")
    except VideoProcessingError:
        print("✅ Exception utilities working!")
        return True


def test_resilience():
    """Test resilience utilities."""
    print("Testing resilience utilities...")
    from telemetry.utils import circuit_breaker

    @circuit_breaker(name="test_service", failure_threshold=2)
    def test_service():
        return "OK"

    result = test_service()
    print(f"✅ Resilience utilities working! Result: {result}")
    return True


def main():
    """Run all tests."""
    tests = [
        test_logging,
        test_timing,
        test_monitoring,
        test_exceptions,
        test_resilience,
    ]

    print("🧪 Testing Core Utilities")
    print("=" * 40)

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print("=" * 40)
    print(f"✅ {passed}/{len(tests)} tests passed!")

    if passed == len(tests):
        print("🎉 All core utilities are working correctly!")
        return 0
    else:
        print("⚠️  Some utilities need attention.")
        return 1


if __name__ == "__main__":
    exit(main())
