"""
Unit tests for UnifiedSession model
Tests model methods, properties, and business logic
"""
from django.test import TestCase
from django.contrib.auth import get_user_model
from api.models import UnifiedSession
import uuid


class UnifiedSessionModelTest(TestCase):
    """Test UnifiedSession model functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_ip = "192.168.1.100"
        self.session = UnifiedSession.objects.create(user_ip=self.test_ip)
        
        # Create a user for account integration tests
        User = get_user_model()
        self.test_user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
    
    def test_session_creation(self):
        """Test basic session creation"""
        session = UnifiedSession.objects.create(user_ip="10.0.0.1")
        
        # Check default values
        self.assertEqual(session.video_requests, 0)
        self.assertEqual(session.playlist_requests, 0)
        self.assertEqual(session.topic_requests, 0)
        self.assertIsNone(session.user_account)
        self.assertIsNotNone(session.session_id)
        self.assertIsInstance(session.session_id, uuid.UUID)
    
    def test_str_representation(self):
        """Test string representation of session"""
        expected = f"Session {str(self.session.session_id)[:8]} - {self.test_ip}"
        self.assertEqual(str(self.session), expected)
    
    def test_total_requests_property(self):
        """Test total_requests property calculation"""
        # Initially should be 0
        self.assertEqual(self.session.total_requests, 0)
        
        # Add some requests
        self.session.video_requests = 2
        self.session.playlist_requests = 1
        self.session.topic_requests = 1
        self.session.save()
        
        # Should sum to 4
        self.assertEqual(self.session.total_requests, 4)
    
    def test_can_make_request_under_limit(self):
        """Test can_make_request when under limit"""
        # Fresh session should allow requests
        self.assertTrue(self.session.can_make_request())
        
        # With 2 total requests should still allow
        self.session.video_requests = 1
        self.session.playlist_requests = 1
        self.session.save()
        self.assertTrue(self.session.can_make_request())
    
    def test_can_make_request_at_limit(self):
        """Test can_make_request when at limit"""
        # Set to exactly 3 requests (the limit)
        self.session.video_requests = 2
        self.session.topic_requests = 1
        self.session.save()
        
        # Should not allow more requests
        self.assertFalse(self.session.can_make_request())
    
    def test_can_make_request_over_limit(self):
        """Test can_make_request when over limit"""
        # Set to over 3 requests
        self.session.video_requests = 3
        self.session.playlist_requests = 2
        self.session.save()
        
        # Should not allow more requests
        self.assertFalse(self.session.can_make_request())
    
    def test_increment_request_count_video(self):
        """Test incrementing video request count"""
        initial_count = self.session.video_requests
        
        self.session.increment_request_count('video')
        
        self.assertEqual(self.session.video_requests, initial_count + 1)
        self.assertEqual(self.session.playlist_requests, 0)  # Others unchanged
        self.assertEqual(self.session.topic_requests, 0)
    
    def test_increment_request_count_playlist(self):
        """Test incrementing playlist request count"""
        initial_count = self.session.playlist_requests
        
        self.session.increment_request_count('playlist')
        
        self.assertEqual(self.session.playlist_requests, initial_count + 1)
        self.assertEqual(self.session.video_requests, 0)  # Others unchanged
        self.assertEqual(self.session.topic_requests, 0)
    
    def test_increment_request_count_topic(self):
        """Test incrementing topic request count"""
        initial_count = self.session.topic_requests
        
        self.session.increment_request_count('topic')
        
        self.assertEqual(self.session.topic_requests, initial_count + 1)
        self.assertEqual(self.session.video_requests, 0)  # Others unchanged
        self.assertEqual(self.session.playlist_requests, 0)
    
    def test_increment_request_count_invalid_type(self):
        """Test incrementing with invalid request type"""
        with self.assertRaises(ValueError) as context:
            self.session.increment_request_count('invalid_type')
        
        self.assertIn("Invalid request type", str(context.exception))
        
        # Counts should remain unchanged
        self.assertEqual(self.session.video_requests, 0)
        self.assertEqual(self.session.playlist_requests, 0)
        self.assertEqual(self.session.topic_requests, 0)
    
    def test_increment_request_count_updates_timestamp(self):
        """Test that incrementing updates last_request_at"""
        original_timestamp = self.session.last_request_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        self.session.increment_request_count('video')
        
        # Timestamp should be updated
        self.assertGreater(self.session.last_request_at, original_timestamp)
    
    def test_get_remaining_requests_fresh_session(self):
        """Test get_remaining_requests for fresh session"""
        self.assertEqual(self.session.get_remaining_requests(), 3)
    
    def test_get_remaining_requests_partial_usage(self):
        """Test get_remaining_requests with partial usage"""
        self.session.video_requests = 1
        self.session.playlist_requests = 1
        self.session.save()
        
        self.assertEqual(self.session.get_remaining_requests(), 1)
    
    def test_get_remaining_requests_at_limit(self):
        """Test get_remaining_requests when at limit"""
        self.session.video_requests = 2
        self.session.topic_requests = 1
        self.session.save()
        
        self.assertEqual(self.session.get_remaining_requests(), 0)
    
    def test_get_remaining_requests_over_limit(self):
        """Test get_remaining_requests when over limit"""
        self.session.video_requests = 3
        self.session.playlist_requests = 2
        self.session.save()
        
        # Should return 0, not negative
        self.assertEqual(self.session.get_remaining_requests(), 0)
    
    def test_user_account_integration(self):
        """Test user account association"""
        # Initially no account
        self.assertIsNone(self.session.user_account)
        
        # Associate with user
        self.session.user_account = self.test_user
        self.session.save()
        
        # Verify association
        self.assertEqual(self.session.user_account, self.test_user)
        
        # Test cascade behavior (SET_NULL on user deletion)
        user_id = self.test_user.id
        self.test_user.delete()
        
        # Refresh from database
        self.session.refresh_from_db()
        self.assertIsNone(self.session.user_account)
    
    def test_session_ordering(self):
        """Test that sessions are ordered by last_request_at descending"""
        # Create multiple sessions with different timestamps
        session1 = UnifiedSession.objects.create(user_ip="10.0.0.1")
        session2 = UnifiedSession.objects.create(user_ip="10.0.0.2")
        
        # Update timestamps by making requests
        session1.increment_request_count('video')
        import time
        time.sleep(0.01)
        session2.increment_request_count('playlist')
        
        # Get all sessions
        sessions = list(UnifiedSession.objects.all())
        
        # Should be ordered by last_request_at descending
        self.assertEqual(sessions[0], session2)  # Most recent first
    
    def test_database_indexes(self):
        """Test that database indexes are properly configured"""
        # This test verifies the Meta configuration
        meta = UnifiedSession._meta
        
        # Check that indexes are defined
        index_fields = []
        for index in meta.indexes:
            index_fields.extend(index.fields)
        
        # Note: session_id is the primary key, so it has an implicit index
        # We only check for explicit indexes defined in Meta.indexes
        expected_indexed_fields = ['user_ip', 'created_at', 'last_request_at']
        
        for field in expected_indexed_fields:
            self.assertIn(field, index_fields, f"Index missing for field: {field}")
        
        # Verify session_id is the primary key (which automatically has an index)
        self.assertEqual(meta.pk.name, 'session_id', "session_id should be the primary key")
    
    def test_application_level_session_prevention(self):
        """Test that application logic prevents multiple sessions per IP per day"""
        from api.services.session_service import SessionService
        from unittest.mock import Mock
        from django.utils import timezone
        
        # Create a mock request with the same IP
        mock_request = Mock()
        mock_request.META = {'REMOTE_ADDR': self.test_ip}
        
        # First call should return existing session
        session1, created1 = SessionService.get_or_create_session(mock_request)
        self.assertEqual(session1.session_id, self.session.session_id)
        self.assertFalse(created1)  # Should find existing session
        
        # Second call should also return the same existing session
        session2, created2 = SessionService.get_or_create_session(mock_request)
        self.assertEqual(session2.session_id, self.session.session_id)
        self.assertFalse(created2)  # Should find existing session
        
        # Verify only one session exists for this IP today
        sessions_today = UnifiedSession.objects.filter(
            user_ip=self.test_ip,
            created_at__date=timezone.now().date()
        )
        self.assertEqual(sessions_today.count(), 1)
    
    def test_session_id_uniqueness(self):
        """Test that session IDs are unique"""
        sessions = []
        
        # Create multiple sessions
        for i in range(10):
            session = UnifiedSession.objects.create(user_ip=f"10.0.0.{i}")
            sessions.append(session.session_id)
        
        # All session IDs should be unique
        self.assertEqual(len(sessions), len(set(sessions)))
