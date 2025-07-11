"""
Management command to test parallel processing integration
"""

import time
import json
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from topic.models import SearchSession, SearchRequest
from topic.parallel_tasks import process_search_results, get_search_processing_status
from topic.tasks import process_search_query, process_search_with_videos


class Command(BaseCommand):
    help = 'Test parallel processing integration for search-to-process workflow'

    def add_arguments(self, parser):
        parser.add_argument(
            '--query',
            type=str,
            default='machine learning tutorial',
            help='Search query to test with'
        )
        parser.add_argument(
            '--test-mode',
            choices=['search-only', 'parallel-processing', 'integrated'],
            default='integrated',
            help='Test mode to run'
        )
        parser.add_argument(
            '--wait-for-completion',
            action='store_true',
            help='Wait for all tasks to complete'
        )
        parser.add_argument(
            '--timeout',
            type=int,
            default=300,
            help='Timeout in seconds for waiting'
        )

    def handle(self, *args, **options):
        query = options['query']
        test_mode = options['test_mode']
        wait_for_completion = options['wait_for_completion']
        timeout = options['timeout']
        
        self.stdout.write(self.style.SUCCESS(f'Starting parallel processing test'))
        self.stdout.write(f'Query: {query}')
        self.stdout.write(f'Test mode: {test_mode}')
        
        try:
            # Create test session and search request
            session, search_request = self._create_test_data(query)
            
            if test_mode == 'search-only':
                result = self._test_search_only(search_request)
            elif test_mode == 'parallel-processing':
                result = self._test_parallel_processing(search_request)
            elif test_mode == 'integrated':
                result = self._test_integrated_workflow(search_request)
            
            self.stdout.write(self.style.SUCCESS(f'Task initiated: {result}'))
            
            if wait_for_completion:
                self._wait_for_completion(search_request, timeout)
            
        except Exception as e:
            raise CommandError(f'Test failed: {str(e)}')

    def _create_test_data(self, query):
        """Create test session and search request"""
        with transaction.atomic():
            session = SearchSession.objects.create(
                user_ip='127.0.0.1',
                status='processing'
            )
            
            search_request = SearchRequest.objects.create(
                search_session=session,
                original_query=query,
                status='processing'
            )
            
            self.stdout.write(f'Created test data - Session: {session.session_id}, Request: {search_request.request_id}')
            return session, search_request

    def _test_search_only(self, search_request):
        """Test search functionality only"""
        self.stdout.write(self.style.WARNING('Testing search-only mode'))
        
        from topic.tasks import process_search_query
        result = process_search_query.apply_async(args=[str(search_request.request_id)])
        
        return {
            'task_id': result.id,
            'search_request_id': str(search_request.request_id),
            'mode': 'search-only'
        }

    def _test_parallel_processing(self, search_request):
        """Test parallel processing assuming search is already complete"""
        self.stdout.write(self.style.WARNING('Testing parallel processing mode'))
        
        # First complete the search
        from topic.tasks import process_search_query
        search_result = process_search_query.apply_async(args=[str(search_request.request_id)])
        search_result = search_result.get()  # Wait for completion
        
        if search_result['status'] != 'success':
            raise CommandError(f'Search failed: {search_result}')
        
        # Then start parallel processing
        result = process_search_results.apply_async(args=[str(search_request.request_id)])
        
        return {
            'task_id': result.id,
            'search_request_id': str(search_request.request_id),
            'mode': 'parallel-processing',
            'search_result': search_result
        }

    def _test_integrated_workflow(self, search_request):
        """Test integrated search and video processing workflow"""
        self.stdout.write(self.style.WARNING('Testing integrated workflow'))
        
        result = process_search_with_videos.apply_async(
            args=[str(search_request.request_id)],
            kwargs={'start_video_processing': True}
        )
        
        return {
            'task_id': result.id,
            'search_request_id': str(search_request.request_id),
            'mode': 'integrated'
        }

    def _wait_for_completion(self, search_request, timeout):
        """Wait for processing to complete"""
        self.stdout.write(self.style.WARNING(f'Waiting for completion (timeout: {timeout}s)'))
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check status
            status_result = get_search_processing_status.apply_async(
                args=[str(search_request.request_id)]
            )
            status = status_result.get()
            
            self.stdout.write(f'Status check: {json.dumps(status, indent=2)}')
            
            # Check if processing is complete
            if status.get('status') == 'success':
                processing_stats = status.get('processing_stats', {})
                if processing_stats.get('processing_videos', 0) == 0:
                    self.stdout.write(self.style.SUCCESS('Processing completed successfully!'))
                    self._print_final_results(search_request)
                    return
            
            # Wait before next check
            time.sleep(10)
        
        self.stdout.write(self.style.ERROR(f'Timeout reached after {timeout} seconds'))

    def _print_final_results(self, search_request):
        """Print final results"""
        self.stdout.write(self.style.SUCCESS('Final Results:'))
        
        # Refresh from database
        search_request.refresh_from_db()
        
        self.stdout.write(f'Search Request Status: {search_request.status}')
        self.stdout.write(f'Total Videos Found: {search_request.total_videos}')
        self.stdout.write(f'Video URLs: {search_request.video_urls}')
        
        # Get video processing results
        from video_processor.processors.search_adapter import get_search_video_results
        video_results = get_search_video_results.apply_async(
            args=[str(search_request.request_id)]
        )
        video_results = video_results.get()
        
        if video_results.get('status') == 'success':
            summary = video_results.get('summary', {})
            self.stdout.write(f'Video Processing Summary: {json.dumps(summary, indent=2)}')
        
        self.stdout.write(self.style.SUCCESS('Test completed!'))