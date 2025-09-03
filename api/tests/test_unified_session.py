"""
UnifiedSession Django Test Suite
Comprehensive tests for session creation, rate limiting, validation, and edge cases.
"""

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

# Test data
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy4Qth_4wQi_Q4"
VALID_TOPIC_QUERY = "machine learning tutorial"


class UnifiedSessionValidationTest(TestCase):
    """Test validation functionality"""

    def setUp(self):
        self.client = APIClient()
        self.url = reverse("unified_process")

    def test_invalid_json(self):
        """Test invalid JSON handling"""
        response = self.client.post(
            self.url, data="invalid json", content_type="application/json"
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)
        self.assertEqual(data["message"], "Invalid JSON format")

    def test_missing_required_fields(self):
        """Test missing required field validation"""
        # Missing type field
        response = self.client.post(self.url, {"content": "test"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)
        self.assertIn("request type", data["message"])

        # Missing content field
        response = self.client.post(self.url, {"type": "video"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)
        self.assertIn("content", data["message"])

    def test_invalid_video_urls(self):
        """Test invalid video URL validation"""
        test_cases = [
            ("not-a-url", "video", "Please provide a valid YouTube URL"),
            ("https://www.google.com", "video", "Please provide a valid YouTube URL"),
            (VALID_VIDEO_URL, "playlist", "YouTube playlist, not a video"),
        ]

        for i, (content, req_type, expected_msg) in enumerate(test_cases):
            with self.subTest(case=i):
                response = self.client.post(
                    self.url, {"content": content, "type": req_type}, format="json"
                )
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                data = response.json()
                self.assertEqual(data["status"], "error")
                self.assertIn(expected_msg, data["message"])

    def test_topic_query_validation(self):
        """Test topic query validation"""
        # Too short
        response = self.client.post(
            self.url, {"content": "hi", "type": "topic"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("4 characters", data["message"])

        # Malicious content
        response = self.client.post(
            self.url,
            {"content": "<script>alert('xss')</script>", "type": "topic"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("unsafe content", data["message"])

    @patch("api.utils.get_client_ip.get_client_ip")
    def test_valid_requests(self, mock_get_ip):
        """Test that valid requests pass validation"""
        test_cases = [
            (VALID_VIDEO_URL, "video"),
            (VALID_PLAYLIST_URL, "playlist"),
            (VALID_TOPIC_QUERY, "topic"),
            ("https://youtu.be/dQw4w9WgXcQ", "video"),  # Short URL
        ]

        for i, (content, req_type) in enumerate(test_cases):
            with self.subTest(case=i):
                mock_get_ip.return_value = (
                    f"10.0.1.{i + 1}"  # Different IP for each test
                )

                response = self.client.post(
                    self.url, {"content": content, "type": req_type}, format="json"
                )
                self.assertEqual(response.status_code, status.HTTP_200_OK)
                data = response.json()
                self.assertEqual(data["status"], "processing")
                self.assertIn("remaining_limit", data)
                self.assertIn("message", data)


class UnifiedSessionManagementTest(TestCase):
    """Test session creation and management"""

    def setUp(self):
        self.client = APIClient()
        self.url = reverse("unified_process")

    @patch("api.utils.get_client_ip.get_client_ip")
    def test_new_session_creation(self, mock_get_ip):
        """Test new session creation"""
        mock_get_ip.return_value = "10.0.2.100"

        response = self.client.post(
            self.url, {"content": VALID_VIDEO_URL, "type": "video"}, format="json"
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["remaining_limit"], 2)
        self.assertIn("session_id", data)
        self.assertIn("Video request accepted", data["message"])

    @patch("api.utils.get_client_ip.get_client_ip")
    def test_session_reuse(self, mock_get_ip):
        """Test session reuse with existing session ID"""
        mock_get_ip.return_value = "10.0.2.101"

        # Create initial session
        response1 = self.client.post(
            self.url, {"content": VALID_VIDEO_URL, "type": "video"}, format="json"
        )
        self.assertEqual(response1.status_code, status.HTTP_200_OK)
        session_id = response1.json()["session_id"]

        # Reuse session
        response2 = self.client.post(
            self.url,
            {"content": VALID_PLAYLIST_URL, "type": "playlist"},
            format="json",
            HTTP_X_SESSION_ID=session_id,
        )

        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        data = response2.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["remaining_limit"], 1)
        self.assertIn("message", data)
        self.assertNotIn(
            "session_id", data
        )  # Should not return session_id for existing sessions

    def test_invalid_session_id(self):
        """Test invalid session ID handling"""
        response = self.client.post(
            self.url,
            {"content": VALID_VIDEO_URL, "type": "video"},
            format="json",
            HTTP_X_SESSION_ID="invalid-uuid",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("message", data)
        self.assertIn("Invalid session", data["message"])


class UnifiedSessionRateLimitingTest(TestCase):
    """Test rate limiting functionality"""

    def setUp(self):
        self.client = APIClient()
        self.url = reverse("unified_process")

    @patch("api.utils.get_client_ip.get_client_ip")
    def test_progressive_rate_limiting(self, mock_get_ip):
        """Test progressive rate limiting (3 requests per day)"""
        mock_get_ip.return_value = "10.0.3.100"

        session_id = None

        # First request
        response = self.client.post(
            self.url, {"content": VALID_VIDEO_URL, "type": "video"}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["remaining_limit"], 2)
        session_id = data["session_id"]

        # Second request
        response = self.client.post(
            self.url,
            {"content": VALID_PLAYLIST_URL, "type": "playlist"},
            format="json",
            HTTP_X_SESSION_ID=session_id,
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["remaining_limit"], 1)

        # Third request
        response = self.client.post(
            self.url,
            {"content": VALID_TOPIC_QUERY, "type": "topic"},
            format="json",
            HTTP_X_SESSION_ID=session_id,
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["remaining_limit"], 0)
        self.assertIn("message", data)

        # Fourth request (should be rate limited)
        response = self.client.post(
            self.url,
            {"content": "python programming", "type": "topic"},
            format="json",
            HTTP_X_SESSION_ID=session_id,
        )
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
        data = response.json()
        self.assertEqual(data["status"], "rate_limited")
        self.assertIn("message", data)
        self.assertIn("Daily limit reached", data["message"])

    @patch("api.utils.get_client_ip.get_client_ip")
    def test_validate_first_approach(self, mock_get_ip):
        """Test that validation happens before rate limiting"""
        mock_get_ip.return_value = "10.0.3.101"

        # Use up all requests
        session_id = None
        for i in range(3):
            response = self.client.post(
                self.url,
                {"content": VALID_VIDEO_URL, "type": "video"},
                format="json",
                HTTP_X_SESSION_ID=session_id,
            )
            if i == 0:
                session_id = response.json()["session_id"]

        # Try invalid request after rate limit exhausted
        response = self.client.post(
            self.url,
            {"content": "invalid-url", "type": "video"},
            format="json",
            HTTP_X_SESSION_ID=session_id,
        )

        # Should return 400 (validation error), not 429 (rate limit)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("YouTube URL", data["message"])
