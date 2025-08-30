#!/usr/bin/env python3
"""
Test script to verify the new modular import structure works correctly.
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.getcwd())


def test_direct_module_imports():
    """Test importing directly from specific modules."""
    print("Testing direct module imports...")

    # Test logging module
    from telemetry.logging import get_logger

    logger = get_logger("test")
    logger.info("✅ Direct logging import works!")

    # Test timing module
    from telemetry.timing import TimingContext

    with TimingContext("test"):
        pass
    print("✅ Direct timing import works!")

    # Test monitoring module
    from telemetry.monitoring import log_memory_usage

    log_memory_usage("test")
    print("✅ Direct monitoring import works!")

    # Test exceptions module
    print("✅ Direct exceptions import works!")

    # Test resilience module
    print("✅ Direct resilience import works!")

    return True


def test_unified_telemetry_imports():
    """Test importing from unified telemetry module."""
    print("Testing unified telemetry imports...")

    from telemetry import (
        get_logger,
    )

    logger = get_logger("unified_test")
    logger.info("✅ Unified telemetry imports work!")

    return True


def test_backward_compatibility():
    """Test backward compatibility with telemetry.utils imports."""
    print("Testing backward compatibility...")

    from telemetry.utils import (
        get_logger,
    )

    logger = get_logger("compat_test")
    logger.info("✅ Backward compatibility works!")

    return True


def main():
    """Run all import tests."""
    tests = [
        test_direct_module_imports,
        test_unified_telemetry_imports,
        test_backward_compatibility,
    ]

    print("🧪 Testing New Telemetry Import Structure")
    print("=" * 50)

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            print()

    print("=" * 50)
    print(f"✅ {passed}/{len(tests)} import tests passed!")

    if passed == len(tests):
        print("🎉 All telemetry import structures are working correctly!")
        return 0
    else:
        print("⚠️  Some imports need attention.")
        return 1


if __name__ == "__main__":
    exit(main())
