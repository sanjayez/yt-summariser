"""
Unit tests for UnifiedGatewayView
Tests request handling, validation, rate limiting integration, and response format
"""

import json
import uuid
from unittest.mock import patch

from django.test import Client, TestCase
from django.urls import reverse

from api.models import UnifiedSession


class UnifiedGatewayViewTest(TestCase):
    """Test UnifiedGatewayView functionality"""

    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.url = reverse("unified_process")  # /api/process/
        self.test_ip = "192.168.1.100"

        # Valid test data
        self.valid_video_data = {
            "content": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "type": "video",
        }

        self.valid_playlist_data = {
            "content": "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy4Qth_4wQi_Q4",
            "type": "playlist",
        }

        self.valid_topic_data = {
            "content": "python machine learning tutorial",
            "type": "topic",
        }

    def test_post_method_only(self):
        """Test that only POST method is allowed"""
        # GET should not be allowed
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

        # PUT should not be allowed
        response = self.client.put(self.url)
        self.assertEqual(response.status_code, 405)

        # DELETE should not be allowed
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 405)

    @patch("api.services.session_service.get_client_ip")
    def test_valid_video_request(self, mock_get_ip):
        """Test valid video request processing"""
        mock_get_ip.return_value = self.test_ip

        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_video_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("session_id", data)
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["remaining_limit"], 2)

        # Verify session was created
        session_id = data["session_id"]
        session = UnifiedSession.objects.get(session_id=session_id)
        self.assertEqual(session.video_requests, 1)
        self.assertEqual(session.user_ip, self.test_ip)

    @patch("api.services.session_service.get_client_ip")
    def test_valid_playlist_request(self, mock_get_ip):
        """Test valid playlist request processing"""
        mock_get_ip.return_value = self.test_ip

        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_playlist_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["remaining_limit"], 2)

        # Verify session was created with playlist request
        session_id = data["session_id"]
        session = UnifiedSession.objects.get(session_id=session_id)
        self.assertEqual(session.playlist_requests, 1)

    @patch("api.services.session_service.get_client_ip")
    def test_valid_topic_request(self, mock_get_ip):
        """Test valid topic request processing"""
        mock_get_ip.return_value = self.test_ip

        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_topic_data),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["remaining_limit"], 2)

        # Verify session was created with topic request
        session_id = data["session_id"]
        session = UnifiedSession.objects.get(session_id=session_id)
        self.assertEqual(session.topic_requests, 1)

    def test_missing_content_field(self):
        """Test request with missing content field"""
        data = {"type": "video"}

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

        response_data = response.json()
        self.assertIn("error", response_data)
        self.assertIn("details", response_data)

    def test_missing_type_field(self):
        """Test request with missing type field"""
        data = {"content": "https://www.youtube.com/watch?v=test"}

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

        response_data = response.json()
        self.assertIn("error", response_data)

    def test_invalid_type_field(self):
        """Test request with invalid type field"""
        data = {
            "content": "https://www.youtube.com/watch?v=test",
            "type": "invalid_type",
        }

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_url_for_video(self):
        """Test non-YouTube URL for video request"""
        data = {"content": "https://www.google.com", "type": "video"}

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_url_for_playlist(self):
        """Test non-YouTube URL for playlist request"""
        data = {"content": "https://www.example.com/playlist", "type": "playlist"}

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

    def test_short_topic_query(self):
        """Test topic query that's too short"""
        data = {
            "content": "ai",  # Only 2 characters
            "type": "topic",
        }

        response = self.client.post(
            self.url, data=json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

    @patch("api.services.session_service.get_client_ip")
    def test_existing_session_usage(self, mock_get_ip):
        """Test using existing session with X-Session-ID header"""
        mock_get_ip.return_value = self.test_ip

        # Create initial session
        response1 = self.client.post(
            self.url,
            data=json.dumps(self.valid_video_data),
            content_type="application/json",
        )

        session_id = response1.json()["session_id"]

        # Use existing session
        response2 = self.client.post(
            self.url,
            data=json.dumps(self.valid_playlist_data),
            content_type="application/json",
            HTTP_X_SESSION_ID=session_id,
        )

        self.assertEqual(response2.status_code, 200)

        data2 = response2.json()
        # session_id should NOT be in response for existing sessions
        self.assertNotIn("session_id", data2)  # Not returned for existing sessions
        self.assertEqual(data2["remaining_limit"], 1)  # Decremented

        # Verify session has both requests
        session = UnifiedSession.objects.get(session_id=session_id)
        self.assertEqual(session.video_requests, 1)
        self.assertEqual(session.playlist_requests, 1)

    @patch("api.services.session_service.get_client_ip")
    def test_rate_limiting_enforcement(self, mock_get_ip):
        """Test rate limiting enforcement (3 requests max)"""
        mock_get_ip.return_value = self.test_ip

        # Make 3 successful requests
        session_id = None
        for i in range(3):
            headers = {}
            if session_id:
                headers["HTTP_X_SESSION_ID"] = session_id

            response = self.client.post(
                self.url,
                data=json.dumps(self.valid_video_data),
                content_type="application/json",
                **headers,
            )

            self.assertEqual(response.status_code, 200)

            if session_id is None:
                session_id = response.json()["session_id"]

            expected_remaining = 2 - i
            self.assertEqual(response.json()["remaining_limit"], expected_remaining)

        # 4th request should be rate limited
        response4 = self.client.post(
            self.url,
            data=json.dumps(self.valid_video_data),
            content_type="application/json",
            HTTP_X_SESSION_ID=session_id,
        )

        self.assertEqual(response4.status_code, 429)  # Too Many Requests

        data4 = response4.json()
        self.assertEqual(data4["status"], "rate_limited")
        self.assertEqual(data4["remaining_limit"], 0)

    def test_response_format_validation(self):
        """Test that response format matches schema exactly"""
        with patch("api.services.session_service.get_client_ip") as mock_get_ip:
            mock_get_ip.return_value = self.test_ip

            response = self.client.post(
                self.url,
                data=json.dumps(self.valid_video_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)

            data = response.json()

            # Check exact fields (no more, no less)
            expected_fields = {"session_id", "status", "remaining_limit"}
            actual_fields = set(data.keys())
            self.assertEqual(actual_fields, expected_fields)

            # Check field types
            self.assertIsInstance(data["session_id"], str)
            self.assertIsInstance(data["status"], str)
            self.assertIsInstance(data["remaining_limit"], int)

            # Validate UUID format
            uuid.UUID(data["session_id"])  # Should not raise exception

    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request body"""
        response = self.client.post(
            self.url, data="invalid json", content_type="application/json"
        )

        # Should return 400 for invalid JSON
        self.assertEqual(response.status_code, 400)

    def test_empty_request_body(self):
        """Test handling of empty request body"""
        response = self.client.post(self.url, data="", content_type="application/json")

        self.assertEqual(response.status_code, 400)

    @patch("api.services.session_service.get_client_ip")
    def test_malformed_session_id_header(self, mock_get_ip):
        """Test handling malformed X-Session-ID header"""
        mock_get_ip.return_value = self.test_ip

        response = self.client.post(
            self.url,
            data=json.dumps(self.valid_video_data),
            content_type="application/json",
            HTTP_X_SESSION_ID="invalid-uuid-format",
        )

        # Should create new session (ignore invalid session ID)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["remaining_limit"], 2)  # New session

    @patch("api.services.session_service.SessionService.check_rate_limit")
    def test_service_error_handling(self, mock_check_rate_limit):
        """Test handling of service layer errors"""
        # Mock service to raise exception
        mock_check_rate_limit.side_effect = Exception("Service error")

        with patch("api.services.session_service.get_client_ip") as mock_get_ip:
            mock_get_ip.return_value = self.test_ip

            response = self.client.post(
                self.url,
                data=json.dumps(self.valid_video_data),
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 500)

            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"], "Internal server error")
