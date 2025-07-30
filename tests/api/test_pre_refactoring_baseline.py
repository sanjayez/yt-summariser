"""
Pre-refactoring baseline tests for API functionality.
These tests establish the current behavior before refactoring to ensure no regressions.
"""
import pytest
import json
import asyncio
from django.test import TransactionTestCase, Client
from django.urls import reverse
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

from api.models import URLRequestTable
from video_processor.models import VideoMetadata, VideoTranscript


class APIPreRefactoringBaselineTests(TransactionTestCase):
    """Baseline tests to capture current API behavior before refactoring"""
    
    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.test_ip = "192.168.1.1"
        
    def test_process_single_video_endpoint_baseline(self):
        """Test current video processing endpoint behavior"""
        
        # Mock the Celery task to avoid actual video processing
        with patch('video_processor.processors.workflow.process_youtube_video') as mock_task:
            mock_task.delay.return_value = MagicMock(id='test-task-id')
            
            response = self.client.post(
                reverse('process_single_video'),
                data=json.dumps({'url': self.test_url}),
                content_type='application/json',
                HTTP_X_FORWARDED_FOR=self.test_ip
            )
            
            # Verify response structure
            self.assertEqual(response.status_code, 201)
            data = response.json()
            
            # Document current response format
            required_fields = ['request_id', 'url', 'status', 'message']
            for field in required_fields:
                self.assertIn(field, data)
            
            self.assertEqual(data['url'], self.test_url)
            self.assertEqual(data['status'], 'processing')
            
            # Verify database record created
            url_request = URLRequestTable.objects.get(request_id=data['request_id'])
            self.assertEqual(url_request.url, self.test_url)
            self.assertEqual(url_request.status, 'processing')
            self.assertEqual(url_request.ip_address, self.test_ip)
    
    def test_get_video_summary_endpoint_baseline(self):
        """Test current video summary endpoint behavior"""
        
        # Create test data
        url_request = URLRequestTable.objects.create(
            url=self.test_url,
            ip_address=self.test_ip,
            status='success'
        )
        
        video_metadata = VideoMetadata.objects.create(
            url_request=url_request,
            video_id='dQw4w9WgXcQ',
            title='Test Video',
            description='Test Description',
            duration=180,
            duration_string='3:00',
            channel_name='Test Channel',
            view_count=1000,
            like_count=100,
            upload_date='2023-01-01',
            language='en',
            webpage_url=self.test_url,
            thumbnail='https://example.com/thumb.jpg',
            status='success'
        )
        
        video_transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            transcript_text='This is a test transcript',
            summary='This is a test summary',
            key_points=['Point 1', 'Point 2'],
            status='success'
        )
        
        response = self.client.get(
            reverse('get_video_summary', kwargs={'request_id': url_request.request_id})
        )
        
        # Verify response structure
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Document current response format
        required_fields = ['summary', 'key_points', 'video_metadata', 'status']
        for field in required_fields:
            self.assertIn(field, data)
        
        self.assertEqual(data['summary'], 'This is a test summary')
        self.assertEqual(data['key_points'], ['Point 1', 'Point 2'])
        self.assertEqual(data['status'], 'completed')
    
    def test_ask_video_question_endpoint_baseline(self):
        """Test current video question endpoint behavior"""
        
        # Create test data with embedded video
        url_request = URLRequestTable.objects.create(
            url=self.test_url,
            ip_address=self.test_ip,
            status='success'
        )
        
        video_metadata = VideoMetadata.objects.create(
            url_request=url_request,
            video_id='dQw4w9WgXcQ',
            title='Test Video',
            webpage_url=self.test_url,
            is_embedded=True,  # Mark as embedded for vector search path
            status='success'
        )
        
        video_transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            transcript_text='This is a test transcript about AI and machine learning',
            status='success'
        )
        
        # Mock the RAG services to avoid actual AI calls
        with patch('api.views.get_rag_services') as mock_services, \
             patch('api.views.search_video_content_async') as mock_search:
            
            mock_services.return_value = {
                'vector': MagicMock(),
                'llm': MagicMock(),
                'config': MagicMock()
            }
            
            mock_search.return_value = {
                'answer': 'This is a test answer about AI',
                'sources': [{
                    'type': 'segment',
                    'timestamp': '01:30',
                    'text': 'AI and machine learning content',
                    'youtube_url': self.test_url,
                    'confidence': 0.85
                }],
                'confidence': 0.85,
                'search_method': 'vector_search',
                'results_count': 1
            }
            
            response = self.client.post(
                reverse('ask_video_question', kwargs={'request_id': url_request.request_id}),
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )
            
            # Verify response structure
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Document current response format
            required_fields = ['question', 'answer', 'sources', 'confidence', 'search_method', 'video_metadata']
            for field in required_fields:
                self.assertIn(field, data)
            
            self.assertEqual(data['question'], 'What is AI?')
            self.assertIn('answer', data)
            self.assertIsInstance(data['sources'], list)
            self.assertIsInstance(data['confidence'], (int, float))
    
    def test_video_status_stream_endpoint_baseline(self):
        """Test current video status streaming endpoint behavior"""
        
        # Create test data
        url_request = URLRequestTable.objects.create(
            url=self.test_url,
            ip_address=self.test_ip,
            status='processing'
        )
        
        response = self.client.get(
            reverse('video_status_stream', kwargs={'request_id': url_request.request_id})
        )
        
        # Verify SSE response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/event-stream')
        self.assertEqual(response['Cache-Control'], 'no-cache')
        self.assertEqual(response['X-Accel-Buffering'], 'no')
    
    def test_invalid_request_handling_baseline(self):
        """Test current error handling behavior"""
        
        # Test missing URL
        response = self.client.post(
            reverse('process_single_video'),
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        # Test invalid request ID for summary
        invalid_uuid = uuid4()
        response = self.client.get(
            reverse('get_video_summary', kwargs={'request_id': invalid_uuid})
        )
        self.assertEqual(response.status_code, 404)
        
        # Test invalid request ID for question
        response = self.client.post(
            reverse('ask_video_question', kwargs={'request_id': invalid_uuid}),
            data=json.dumps({'question': 'Test question?'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 404)
    
    def test_search_transcript_fallback_baseline(self):
        """Test current transcript fallback behavior"""
        
        # Create test data without embedding (should trigger fallback)
        url_request = URLRequestTable.objects.create(
            url=self.test_url,
            ip_address=self.test_ip,
            status='success'
        )
        
        video_metadata = VideoMetadata.objects.create(
            url_request=url_request,
            video_id='dQw4w9WgXcQ',
            title='Test Video',
            webpage_url=self.test_url,
            is_embedded=False,  # No embedding, should use fallback
            status='success'
        )
        
        video_transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            transcript_text='This is a test transcript about AI and machine learning. It contains important information about neural networks.',
            status='success'
        )
        
        # Mock fallback search
        with patch('api.views.search_transcript_fallback') as mock_fallback:
            mock_fallback.return_value = {
                'answer': 'Based on the video transcript, here are the most relevant parts:\n\nThis is a test transcript about AI and machine learning.',
                'sources': [{
                    'type': 'transcript',
                    'timestamp': 'Unknown',
                    'text': 'This is a test transcript about AI and machine learning',
                    'youtube_url': self.test_url,
                    'confidence': 0.5
                }],
                'confidence': 0.5,
                'search_method': 'transcript_fallback',
                'results_count': 1
            }
            
            response = self.client.post(
                reverse('ask_video_question', kwargs={'request_id': url_request.request_id}),
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['search_method'], 'transcript_fallback')


class APIFunctionBehaviorTests(TransactionTestCase):
    """Test individual function behaviors that will be extracted"""
    
    def test_compress_context_function_baseline(self):
        """Test current compress_context function behavior"""
        from api.views import compress_context
        
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
        result = compress_context(test_results)
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), 600)
        
        # Verify sorting by score (highest first)
        lines = result.split('\n')
        self.assertIn('[01:30]', lines[0])  # Highest score segment first
        
        # Test with custom max_chars
        result_short = compress_context(test_results, max_chars=100)
        self.assertLessEqual(len(result_short), 100)
    
    def test_cache_functions_baseline(self):
        """Test current caching function behaviors"""
        from api.views import get_cached_embedding, cache_embedding
        from django.core.cache import cache
        
        # Clear cache
        cache.clear()
        
        test_question = "What is machine learning?"
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test cache miss
        result = get_cached_embedding(test_question)
        self.assertIsNone(result)
        
        # Test cache set
        cache_embedding(test_question, test_embedding)
        
        # Test cache hit
        cached_result = get_cached_embedding(test_question)
        self.assertEqual(cached_result, test_embedding)
    
    def test_get_rag_services_function_baseline(self):
        """Test current get_rag_services function behavior with mocking"""
        
        # Mock the external dependencies
        with patch('api.views.VectorService') as mock_vector, \
             patch('api.views.GeminiLLMProvider') as mock_llm, \
             patch('api.views.WeaviateVectorStoreProvider') as mock_store, \
             patch('api.views.LLMService') as mock_llm_service, \
             patch('api.views.get_config') as mock_config:
            
            mock_config.return_value = {'test': 'config'}
            
            from api.views import get_rag_services
            
            # First call should initialize services
            services = get_rag_services()
            
            self.assertIn('vector', services)
            self.assertIn('llm', services)
            self.assertIn('config', services)
            
            # Second call should return cached services (due to @lru_cache)
            services2 = get_rag_services()
            self.assertEqual(services, services2)


if __name__ == '__main__':
    pytest.main([__file__])