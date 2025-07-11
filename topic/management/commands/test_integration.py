"""
Management command to test the search-to-process integration functionality.
"""

import json
import time
from django.core.management.base import BaseCommand
from django.test import RequestFactory
from django.contrib.auth.models import AnonymousUser

from topic.views import IntegratedSearchProcessAPIView, SearchStatusAPIView
from topic.models import SearchRequest


class Command(BaseCommand):
    help = 'Test the search-to-process integration functionality'

    def add_arguments(self, parser):
        parser.add_argument(
            '--query',
            default='python tutorial',
            help='Search query to test with'
        )
        parser.add_argument(
            '--max-videos',
            type=int,
            default=2,
            help='Maximum number of videos to process'
        )
        parser.add_argument(
            '--wait',
            action='store_true',
            help='Wait for processing to complete'
        )

    def handle(self, *args, **options):
        query = options['query']
        max_videos = options['max_videos']
        wait_for_completion = options['wait']

        self.stdout.write(
            self.style.SUCCESS(f'Testing search-to-process integration with query: "{query}"')
        )

        # Create a mock request
        factory = RequestFactory()
        request_data = {
            'query': query,
            'max_videos': max_videos,
            'process_videos': True
        }

        # Test the integrated API
        request = factory.post(
            '/api/topic/search/integrated/',
            data=request_data,
            content_type='application/json'
        )
        request.user = AnonymousUser()
        
        # Add remote address for session creation
        request.META['REMOTE_ADDR'] = '127.0.0.1'
        request.META['HTTP_X_FORWARDED_FOR'] = '127.0.0.1'
        
        # Add the data attribute that DRF expects
        request.data = request_data

        # Call the API view
        view = IntegratedSearchProcessAPIView()
        try:
            response = view.post(request)
            
            self.stdout.write(f'Response status: {response.status_code}')
            self.stdout.write(f'Response data: {json.dumps(response.data, indent=2)}')
            
            if response.status_code == 202:  # HTTP_202_ACCEPTED
                search_request_id = response.data.get('search_request_id')
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully started processing. Search ID: {search_request_id}')
                )
                
                if wait_for_completion and search_request_id:
                    self.stdout.write('Waiting for processing to complete...')
                    self.wait_for_completion(search_request_id)
            else:
                self.stdout.write(
                    self.style.ERROR(f'API call failed with status {response.status_code}')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error testing API: {str(e)}')
            )
            import traceback
            traceback.print_exc()

    def wait_for_completion(self, search_request_id, max_wait_time=600):
        """Wait for the search processing to complete"""
        factory = RequestFactory()
        status_view = SearchStatusAPIView()
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                # Create status check request
                request = factory.get(f'/api/topic/search/status/{search_request_id}/')
                request.user = AnonymousUser()
                request.META['REMOTE_ADDR'] = '127.0.0.1'
                
                # Get status
                response = status_view.get(request, search_request_id)
                
                if response.status_code == 200:
                    data = response.data
                    status = data.get('status', 'unknown')
                    progress = data.get('progress_percentage', 0)
                    total_videos = data.get('total_videos', 0)
                    completed_videos = data.get('completed_videos', 0)
                    
                    self.stdout.write(f'Status: {status}, Progress: {progress}%, Videos: {completed_videos}/{total_videos}')
                    
                    if status in ['success', 'failed']:
                        self.stdout.write(
                            self.style.SUCCESS(f'Processing completed with status: {status}')
                        )
                        self.stdout.write(f'Final results: {json.dumps(data, indent=2)}')
                        return
                    elif status == 'partial_success':
                        self.stdout.write(
                            self.style.WARNING(f'Processing completed with partial success')
                        )
                        self.stdout.write(f'Final results: {json.dumps(data, indent=2)}')
                        return
                        
                else:
                    self.stdout.write(f'Status check failed: {response.status_code}')
                    
            except Exception as e:
                self.stdout.write(f'Error checking status: {str(e)}')
            
            time.sleep(10)  # Wait 10 seconds before checking again
        
        self.stdout.write(
            self.style.WARNING(f'Timeout waiting for completion after {max_wait_time} seconds')
        )