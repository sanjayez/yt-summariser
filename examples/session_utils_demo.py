"""
Example demonstrating topic session utilities usage.

This script shows how to use the session management utilities in the topic app.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "yt_summariser.settings")

import django

django.setup()

from django.test import RequestFactory

from topic.models import SearchSession
from topic.utils import get_or_create_session
from topic.utils.session_utils import update_session_status


def demo_session_creation():
    """Demonstrate session creation and retrieval"""
    print("=== Session Creation Demo ===")

    # Create a mock request
    factory = RequestFactory()
    request = factory.get("/")
    request.META["REMOTE_ADDR"] = "192.168.1.100"

    # Get or create session
    session = get_or_create_session(request)
    print(f"Created session: {session.session_id}")
    print(f"IP Address: {session.user_ip}")
    print(f"Status: {session.status}")
    print(f"Created at: {session.created_at}")

    # Try to get the same session again
    session2 = get_or_create_session(request)
    print(f"\nRetrieved session: {session2.session_id}")
    print(f"Same session? {session.session_id == session2.session_id}")


def demo_session_status_update():
    """Demonstrate session status updates"""
    print("\n=== Session Status Update Demo ===")

    # Create a mock request
    factory = RequestFactory()
    request = factory.get("/")
    request.META["REMOTE_ADDR"] = "192.168.1.101"

    # Get or create session
    session = get_or_create_session(request)
    print(f"Initial status: {session.status}")

    # Update status to success
    update_session_status(session, "success")
    print(f"Updated status: {session.status}")

    # Update to failed
    update_session_status(session, "failed")
    print(f"Final status: {session.status}")


def demo_forwarded_ip():
    """Demonstrate IP extraction from X-Forwarded-For header"""
    print("\n=== Forwarded IP Demo ===")

    # Create a mock request with X-Forwarded-For header
    factory = RequestFactory()
    request = factory.get("/")
    request.META["HTTP_X_FORWARDED_FOR"] = "203.0.113.1, 192.168.1.100"
    request.META["REMOTE_ADDR"] = "192.168.1.100"

    # Get or create session
    session = get_or_create_session(request)
    print(f"Client IP extracted: {session.user_ip}")
    print("Should be the first IP from X-Forwarded-For: 203.0.113.1")


def demo_session_filtering():
    """Demonstrate session filtering behavior"""
    print("\n=== Session Filtering Demo ===")

    # Create a mock request
    factory = RequestFactory()
    request = factory.get("/")
    request.META["REMOTE_ADDR"] = "192.168.1.102"

    # Create initial session
    session1 = get_or_create_session(request)
    print(f"Created session 1: {session1.session_id}")

    # Mark it as failed
    update_session_status(session1, "failed")
    print(f"Marked session 1 as failed: {session1.status}")

    # Get session again - should create a new one since previous failed
    session2 = get_or_create_session(request)
    print(f"Created session 2: {session2.session_id}")
    print(f"Different session? {session1.session_id != session2.session_id}")


if __name__ == "__main__":
    print("Topic Session Utils Demo")
    print("=" * 50)

    # Clean up any existing sessions for demo
    SearchSession.objects.all().delete()

    demo_session_creation()
    demo_session_status_update()
    demo_forwarded_ip()
    demo_session_filtering()

    print("\nDemo completed!")
    print(f"Total sessions created: {SearchSession.objects.count()}")
