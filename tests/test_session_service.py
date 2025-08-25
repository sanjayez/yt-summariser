"""
Unit tests for SessionService business logic
Tests session management, rate limiting, and IP validation
"""
from django.test import TestCase
from django.http import HttpRequest
from unittest.mock import Mock, patch
from api.models import UnifiedSession
from api.services.session_service import SessionService
import uuid


class SessionServiceTest(TestCase):
    """Test SessionService business logic"""
    
    def setUp(self):
        """Set up test data"""
        self.test_ip = "192.168.1.100"
        self.test_session_id = str(uuid.uuid4())
        
        # Create a mock request
        self.mock_request = Mock(spec=HttpRequest)
        
        # Create test session
        self.test_session = UnifiedSession.objects.create(
            session_id=self.test_session_id,
            user_ip=self.test_ip
        )
    
    @patch('api.services.session_service.get_client_ip')
    def test_get_or_create_session_new_session(self, mock_get_ip):
        """Test creating new session when no session_id provided"""
        mock_get_ip.return_value = "10.0.0.1"  # Different IP from setUp
        
        session, is_new = SessionService.get_or_create_session(self.mock_request)
        
        self.assertTrue(is_new)
        self.assertIsInstance(session, UnifiedSession)
        self.assertEqual(session.user_ip, "10.0.0.1")
        self.assertEqual(session.video_requests, 0)
        self.assertEqual(session.playlist_requests, 0)
        self.assertEqual(session.topic_requests, 0)
    
    @patch('api.services.session_service.get_client_ip')
    def test_get_or_create_session_existing_session(self, mock_get_ip):
        """Test retrieving existing session with valid session_id"""
        mock_get_ip.return_value = self.test_ip
        
        session, is_new = SessionService.get_or_create_session(
            self.mock_request, 
            session_id=self.test_session_id
        )
        
        self.assertFalse(is_new)
        self.assertEqual(session.session_id, uuid.UUID(self.test_session_id))
        self.assertEqual(session.user_ip, self.test_ip)
    
    @patch('api.services.session_service.get_client_ip')
    def test_get_or_create_session_ip_mismatch(self, mock_get_ip):
        """Test session retrieval with IP mismatch creates new session"""
        mock_get_ip.return_value = "10.0.0.1"  # Different IP
        
        session, is_new = SessionService.get_or_create_session(
            self.mock_request,
            session_id=self.test_session_id
        )
        
        # Should create new session due to IP mismatch
        self.assertTrue(is_new)
        self.assertNotEqual(session.session_id, uuid.UUID(self.test_session_id))
        self.assertEqual(session.user_ip, "10.0.0.1")
    
    @patch('api.services.session_service.get_client_ip')
    def test_get_or_create_session_invalid_session_id(self, mock_get_ip):
        """Test handling invalid session_id format"""
        mock_get_ip.return_value = "172.16.0.1"  # Different IP from setUp
        
        session, is_new = SessionService.get_or_create_session(
            self.mock_request,
            session_id="invalid-uuid-format"
        )
        
        # Should create new session
        self.assertTrue(is_new)
        self.assertEqual(session.user_ip, "172.16.0.1")
    
    @patch('api.services.session_service.get_client_ip')
    def test_get_or_create_session_nonexistent_session_id(self, mock_get_ip):
        """Test handling non-existent but valid session_id"""
        mock_get_ip.return_value = "203.0.113.1"  # Different IP from setUp
        fake_session_id = str(uuid.uuid4())
        
        session, is_new = SessionService.get_or_create_session(
            self.mock_request,
            session_id=fake_session_id
        )
        
        # Should create new session
        self.assertTrue(is_new)
        self.assertEqual(session.user_ip, "203.0.113.1")
    
    def test_check_rate_limit_valid_request_type_video(self):
        """Test rate limit check with valid video request type"""
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'video')
        
        self.assertTrue(allowed)
        self.assertEqual(response_data['status'], 'processing')
        self.assertEqual(response_data['remaining_limit'], 2)  # 3 - 1 = 2
        
        # Verify session was updated
        self.test_session.refresh_from_db()
        self.assertEqual(self.test_session.video_requests, 1)
    
    def test_check_rate_limit_valid_request_type_playlist(self):
        """Test rate limit check with valid playlist request type"""
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'playlist')
        
        self.assertTrue(allowed)
        self.assertEqual(response_data['status'], 'processing')
        self.assertEqual(response_data['remaining_limit'], 2)
        
        # Verify session was updated
        self.test_session.refresh_from_db()
        self.assertEqual(self.test_session.playlist_requests, 1)
    
    def test_check_rate_limit_valid_request_type_topic(self):
        """Test rate limit check with valid topic request type"""
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'topic')
        
        self.assertTrue(allowed)
        self.assertEqual(response_data['status'], 'processing')
        self.assertEqual(response_data['remaining_limit'], 2)
        
        # Verify session was updated
        self.test_session.refresh_from_db()
        self.assertEqual(self.test_session.topic_requests, 1)
    
    def test_check_rate_limit_invalid_request_type(self):
        """Test rate limit check with invalid request type"""
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'invalid')
        
        self.assertFalse(allowed)
        self.assertEqual(response_data['status'], 'error')
        self.assertIn('Invalid request type', response_data['message'])
        
        # Verify session was not updated
        self.test_session.refresh_from_db()
        self.assertEqual(self.test_session.video_requests, 0)
        self.assertEqual(self.test_session.playlist_requests, 0)
        self.assertEqual(self.test_session.topic_requests, 0)
    
    def test_check_rate_limit_at_limit(self):
        """Test rate limit check when at daily limit"""
        # Set session to limit (3 requests)
        self.test_session.video_requests = 2
        self.test_session.playlist_requests = 1
        self.test_session.save()
        
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'video')
        
        self.assertFalse(allowed)
        self.assertEqual(response_data['status'], 'rate_limited')
        self.assertEqual(response_data['remaining_limit'], 0)
        self.assertIn('Daily limit reached', response_data['message'])
        
        # Verify session was not updated further
        self.test_session.refresh_from_db()
        self.assertEqual(self.test_session.video_requests, 2)  # Unchanged
    
    def test_check_rate_limit_over_limit(self):
        """Test rate limit check when over daily limit"""
        # Set session over limit
        self.test_session.video_requests = 3
        self.test_session.playlist_requests = 2
        self.test_session.save()
        
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'topic')
        
        self.assertFalse(allowed)
        self.assertEqual(response_data['status'], 'rate_limited')
        self.assertEqual(response_data['remaining_limit'], 0)
    
    def test_check_rate_limit_sequence(self):
        """Test rate limit sequence: 3 allowed, 4th denied"""
        request_types = ['video', 'playlist', 'topic', 'video']
        expected_allowed = [True, True, True, False]
        expected_remaining = [2, 1, 0, 0]
        
        for i, (req_type, exp_allowed, exp_remaining) in enumerate(zip(request_types, expected_allowed, expected_remaining)):
            allowed, response_data = SessionService.check_rate_limit(self.test_session, req_type)
            
            self.assertEqual(allowed, exp_allowed, f"Request {i+1} allowed status")
            self.assertEqual(response_data['remaining_limit'], exp_remaining, f"Request {i+1} remaining limit")
            
            if exp_allowed:
                self.assertEqual(response_data['status'], 'processing')
            else:
                self.assertEqual(response_data['status'], 'rate_limited')
    
    def test_check_rate_limit_response_format(self):
        """Test that rate limit response has correct format"""
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'video')
        
        # Check required fields
        required_fields = ['session_id', 'status', 'remaining_limit', 'message']
        for field in required_fields:
            self.assertIn(field, response_data, f"Missing field: {field}")
        
        # Check field types
        self.assertIsInstance(response_data['session_id'], str)
        self.assertIsInstance(response_data['status'], str)
        self.assertIsInstance(response_data['remaining_limit'], int)
        self.assertIsInstance(response_data['message'], str)
        
        # Check session_id format
        uuid.UUID(response_data['session_id'])  # Should not raise exception
    
    def test_get_session_info(self):
        """Test get_session_info method"""
        # Add some request counts
        self.test_session.video_requests = 2
        self.test_session.playlist_requests = 1
        self.test_session.save()
        
        info = SessionService.get_session_info(self.test_session)
        
        # Check required fields
        required_fields = [
            'session_id', 'user_ip', 'total_requests', 'video_requests',
            'playlist_requests', 'topic_requests', 'remaining_limit',
            'created_at', 'last_request_at', 'has_account'
        ]
        
        for field in required_fields:
            self.assertIn(field, info, f"Missing field: {field}")
        
        # Check values
        self.assertEqual(info['session_id'], str(self.test_session.session_id))
        self.assertEqual(info['user_ip'], self.test_session.user_ip)
        self.assertEqual(info['total_requests'], 3)  # 2 + 1 + 0
        self.assertEqual(info['video_requests'], 2)
        self.assertEqual(info['playlist_requests'], 1)
        self.assertEqual(info['topic_requests'], 0)
        self.assertEqual(info['remaining_limit'], 0)  # 3 - 3 = 0
        self.assertFalse(info['has_account'])
    
    def test_get_session_info_with_account(self):
        """Test get_session_info with associated user account"""
        from django.contrib.auth.models import User
        
        user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        
        self.test_session.user_account = user
        self.test_session.save()
        
        info = SessionService.get_session_info(self.test_session)
        
        self.assertTrue(info['has_account'])
    
    def test_daily_request_limit_constant(self):
        """Test that DAILY_REQUEST_LIMIT is properly defined"""
        self.assertEqual(SessionService.DAILY_REQUEST_LIMIT, 3)
    
    @patch('api.services.session_service.get_client_ip')
    def test_session_creation_with_different_ips(self, mock_get_ip):
        """Test creating sessions with different IP addresses"""
        ips = ["192.168.1.1", "10.0.0.1", "172.16.0.1"]
        sessions = []
        
        for ip in ips:
            mock_get_ip.return_value = ip
            session, is_new = SessionService.get_or_create_session(self.mock_request)
            
            self.assertTrue(is_new)
            self.assertEqual(session.user_ip, ip)
            sessions.append(session)
        
        # All sessions should be different
        session_ids = [s.session_id for s in sessions]
        self.assertEqual(len(session_ids), len(set(session_ids)))
    
    def test_rate_limit_message_content(self):
        """Test that rate limit messages contain useful information"""
        # Test successful request message
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'video')
        
        self.assertIn('Request accepted', response_data['message'])
        self.assertIn('remaining today', response_data['message'])
        
        # Test rate limited message
        self.test_session.video_requests = 3
        self.test_session.save()
        
        allowed, response_data = SessionService.check_rate_limit(self.test_session, 'playlist')
        
        self.assertIn('Daily limit reached', response_data['message'])
        self.assertIn('3 requests', response_data['message'])
    
    def test_concurrent_session_updates(self):
        """Test handling concurrent session updates"""
        # Get the session created in setUp
        session = UnifiedSession.objects.get(session_id=self.test_session_id)
        
        # Verify initial state
        self.assertEqual(session.video_requests, 0)
        self.assertEqual(session.playlist_requests, 0)
        
        # Simulate concurrent updates by calling check_rate_limit multiple times
        # This should increment the counters properly
        allowed1, _ = SessionService.check_rate_limit(session, 'video')
        self.assertTrue(allowed1)
        
        # Refresh session from database to get updated state
        session.refresh_from_db()
        allowed2, _ = SessionService.check_rate_limit(session, 'playlist')
        self.assertTrue(allowed2)
        
        # Refresh and check final state
        final_session = UnifiedSession.objects.get(session_id=self.test_session_id)
        
        # Should have both updates
        self.assertEqual(final_session.video_requests, 1)
        self.assertEqual(final_session.playlist_requests, 1)
