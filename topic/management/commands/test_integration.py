"""
Management command to test the search-to-process integration functionality.
"""

import json
import time

from django.contrib.auth.models import AnonymousUser
from django.core.management.base import BaseCommand
from django.test import RequestFactory

from topic.models import SearchRequest
from topic.views import IntegratedSearchProcessAPIView


class Command(BaseCommand):
    help = "Test the search-to-process integration functionality"

    def add_arguments(self, parser):
        parser.add_argument(
            "--query", default="python tutorial", help="Search query to test with"
        )
        parser.add_argument(
            "--max-videos",
            type=int,
            default=2,
            help="Maximum number of videos to process",
        )
        parser.add_argument(
            "--wait", action="store_true", help="Wait for processing to complete"
        )

    def handle(self, *args, **options):
        query = options["query"]
        max_videos = options["max_videos"]
        wait_for_completion = options["wait"]

        self.stdout.write(
            self.style.SUCCESS(
                f'Testing search-to-process integration with query: "{query}"'
            )
        )

        # Create a mock request
        factory = RequestFactory()
        request_data = {
            "query": query,
            "max_videos": max_videos,
            "process_videos": True,
        }

        # Test the integrated API
        request = factory.post(
            "/api/topic/search/integrated/",
            data=request_data,
            content_type="application/json",
        )
        request.user = AnonymousUser()

        # Add remote address for session creation
        request.META["REMOTE_ADDR"] = "127.0.0.1"
        request.META["HTTP_X_FORWARDED_FOR"] = "127.0.0.1"

        # Add the data attribute that DRF expects
        request.data = request_data

        # Call the API view
        view = IntegratedSearchProcessAPIView()
        try:
            response = view.post(request)

            self.stdout.write(f"Response status: {response.status_code}")
            self.stdout.write(f"Response data: {json.dumps(response.data, indent=2)}")

            if response.status_code == 202:  # HTTP_202_ACCEPTED
                # Update the response data access
                search_id = response.data.get("search_id")
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully started processing. Search ID: {search_id}"
                    )
                )

                if wait_for_completion and search_id:
                    self.stdout.write("Waiting for processing to complete...")
                    self.wait_for_completion(search_id)
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"API call failed with status {response.status_code}"
                    )
                )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error testing API: {str(e)}"))
            import traceback

            traceback.print_exc()

    def wait_for_completion(self, search_id, max_wait_time=600):
        """Wait for the search processing to complete"""
        # Note: SearchStatusAPIView doesn't exist; use DB polling against the SearchRequest model

        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                # Check status directly from the database
                search_request = SearchRequest.objects.get(search_id=search_id)

                # Get status from the model
                status = getattr(search_request, "status", "unknown")

                self.stdout.write(f"Status: {status}")

                if status in ["success", "completed", "failed"]:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Processing completed with status: {status}"
                        )
                    )
                    return
                elif status == "partial_success":
                    self.stdout.write(
                        self.style.WARNING("Processing completed with partial success")
                    )
                    return

                # If status is still processing, wait and continue

            except Exception as e:
                self.stdout.write(f"Error checking status: {str(e)}")

            time.sleep(10)  # Wait 10 seconds before checking again

        self.stdout.write(
            self.style.WARNING(
                f"Timeout waiting for completion after {max_wait_time} seconds"
            )
        )
