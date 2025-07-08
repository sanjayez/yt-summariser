from django.test import TestCase, RequestFactory
from django.core.exceptions import ValidationError
from topic.models import SearchSession
from topic.utils.session_utils import get_or_create_session, update_session_status


class SessionUtilsTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        
    def test_get_or_create_session_new_session(self):
        """Test creating a new session for a new IP"""
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        session = get_or_create_session(request)
        
        self.assertIsInstance(session, SearchSession)
        self.assertEqual(session.user_ip, '192.168.1.100')
        self.assertEqual(session.status, 'processing')
        
    def test_get_or_create_session_existing_session(self):
        """Test retrieving existing session for same IP"""
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        # Create first session
        session1 = get_or_create_session(request)
        
        # Get session again - should return the same one
        session2 = get_or_create_session(request)
        
        self.assertEqual(session1.session_id, session2.session_id)
        
    def test_get_or_create_session_with_forwarded_ip(self):
        """Test IP extraction from X-Forwarded-For header"""
        request = self.factory.get('/')
        request.META['HTTP_X_FORWARDED_FOR'] = '203.0.113.1, 192.168.1.100'
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        session = get_or_create_session(request)
        
        self.assertEqual(session.user_ip, '203.0.113.1')
        
    def test_update_session_status_valid(self):
        """Test updating session status with valid status"""
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        session = get_or_create_session(request)
        updated_session = update_session_status(session, 'success')
        
        self.assertEqual(updated_session.status, 'success')
        
    def test_update_session_status_invalid(self):
        """Test updating session status with invalid status"""
        request = self.factory.get('/')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        session = get_or_create_session(request)
        
        with self.assertRaises(ValueError):
            update_session_status(session, 'invalid_status')
