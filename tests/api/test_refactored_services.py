"""
Tests for refactored API services.
Ensures the extracted services work correctly after refactoring.
"""
import pytest
from django.test import TransactionTestCase
from django.core.cache import cache

from api.services.response_service import ResponseService
from api.services.cache_service import CacheService
from api.services.service_container import ServiceContainer


class RefactoredServicesTests(TransactionTestCase):
    """Test refactored service functionality"""
    
    def setUp(self):
        """Set up test services"""
        self.response_service = ResponseService()
        self.cache_service = CacheService()
        self.service_container = ServiceContainer()
        
        # Clear cache between tests
        cache.clear()
    
    def test_compress_context_function_refactored(self):
        """Test that compress_context works in ResponseService"""
        
        # Test data similar to what the function currently handles
        test_results = [
            {
                'text': 'This is a long text segment that should be compressed properly when it exceeds the character limit',
                'score': 0.95,
                'type': 'segment',
                'metadata': {'start_time': 90}  # 1:30
            },
            {
                'text': 'This is another segment with lower relevance score',
                'score': 0.75,
                'type': 'segment', 
                'metadata': {'start_time': 150}  # 2:30
            },
            {
                'text': 'Short segment',
                'score': 0.65,
                'type': 'segment',
                'metadata': {'start_time': 30}  # 0:30
            }
        ]
        
        # Test with default max_chars
        result = self.response_service.compress_context(test_results)
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), 600)
        
        # Verify sorting by score (highest first)
        lines = result.split('\n')
        self.assertIn('[01:30]', lines[0])  # Highest score segment first
        
        # Test with custom max_chars
        result_short = self.response_service.compress_context(test_results, max_chars=100)
        self.assertLessEqual(len(result_short), 100)
    
    def test_cache_functions_refactored(self):
        """Test that caching functions work in CacheService"""
        
        test_question = "What is machine learning?"
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test cache miss
        result = self.cache_service.get_cached_embedding(test_question)
        self.assertIsNone(result)
        
        # Test cache set
        self.cache_service.cache_embedding(test_question, test_embedding)
        
        # Test cache hit
        cached_result = self.cache_service.get_cached_embedding(test_question)
        self.assertEqual(cached_result, test_embedding)
    
    def test_service_container_initialization(self):
        """Test that ServiceContainer can initialize services"""
        
        # Test that container can be initialized
        self.assertFalse(self.service_container.is_initialized())
        
        # Note: We can't fully test initialization without mocking AI services
        # But we can test the basic container functionality
        
        # Test service registration
        test_service = "test_service_instance"
        self.service_container.register_service("test", test_service)
        
        # Test service retrieval
        retrieved = self.service_container.get_service("test")
        self.assertEqual(retrieved, test_service)
        
        # Test error on missing service
        with self.assertRaises(ValueError):
            self.service_container.get_service("nonexistent")
    
    def test_response_service_formatting(self):
        """Test response formatting functions"""
        
        # Test confidence calculation
        results = [
            {'score': 0.9},
            {'score': 0.8},
            {'score': 0.7}
        ]
        confidence = self.response_service.calculate_confidence_score(results)
        self.assertAlmostEqual(confidence, 0.8, places=1)
        
        # Test empty results
        empty_confidence = self.response_service.calculate_confidence_score([])
        self.assertEqual(empty_confidence, 0.0)
        
        # Test error response formatting
        error_response = self.response_service.format_error_response(
            "Test error", "Test message", "test_status"
        )
        self.assertEqual(error_response['error'], "Test error")
        self.assertEqual(error_response['message'], "Test message")
        self.assertEqual(error_response['status'], "test_status")
    
    def test_cache_service_additional_functions(self):
        """Test additional cache service functions"""
        
        # Test response caching
        test_response = {"answer": "test answer", "confidence": 0.8}
        cache_key = "test_response_key"
        
        # Cache miss
        result = self.cache_service.get_cached_response(cache_key)
        self.assertIsNone(result)
        
        # Cache set
        self.cache_service.cache_response(cache_key, test_response)
        
        # Cache hit
        cached_response = self.cache_service.get_cached_response(cache_key)
        self.assertEqual(cached_response, test_response)
        
        # Test cache key generation
        question = "Test question"
        video_id = "test_video_id"
        generated_key = self.cache_service.generate_response_cache_key(question, video_id)
        self.assertIsInstance(generated_key, str)
        self.assertIn(video_id, generated_key)