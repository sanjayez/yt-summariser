#!/usr/bin/env python3
"""
Test runner for unified session management system
Runs Django unit tests with proper configuration
"""

import glob
import os
import sys

import django
from django.conf import settings
from django.test.utils import get_runner


def run_tests():
    """Run all tests for unified session management"""

    # Ensure we're in the project directory
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_summariser.settings")
    django.setup()

    # Get Django test runner
    TestRunner = get_runner(settings)
    test_runner = TestRunner()

    test_files = glob.glob("tests/test_*.py")
    api_test_files = glob.glob("api/tests/test_*.py")

    # Exclude problematic test files
    excluded_tests = [
        "tests/test_monitoring.py",  # Uses non-existent 'core' module
    ]

    test_files = [f for f in test_files if f not in excluded_tests]
    all_test_files = test_files + api_test_files
    test_modules = [f.replace("/", ".").replace(".py", "") for f in all_test_files]

    # Run specific test modules
    # test_modules = [
    #     'tests.test_models',
    #     'tests.test_session_service',
    #     'tests.test_gateway_views',
    #     'tests.test_schemas'
    # ]

    print("🧪 Running Unified Session Management Tests")
    print("=" * 50)
    print(f"📝 Found {len(test_files)} core tests and {len(api_test_files)} API tests")
    print("=" * 50)

    # Run tests
    failures = test_runner.run_tests(test_modules)

    if failures:
        print(f"\n❌ {failures} test(s) failed!")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
