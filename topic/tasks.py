"""
Celery Tasks for Topic Search Processing
Handles asynchronous YouTube search and AI processing operations
"""

import asyncio
import logging

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction

from ai_utils.config import get_config
from ai_utils.services.registry import get_gemini_llm_service
from video_processor.config import BUSINESS_LOGIC_CONFIG, YOUTUBE_CONFIG
from video_processor.utils import handle_dead_letter_task

from .models import SearchRequest as SearchRequestModel
from .models import SearchSession
from .services.providers.scrapetube_provider import ScrapeTubeProvider
from .services.query_processor import QueryProcessor
from .services.search_service import SearchRequest, YouTubeSearchService
from .utils.explorer_progress import ExplorerProgressTracker
from .utils.session_utils import update_session_status

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name="topic.process_search_query",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["search_soft_limit"],
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["search_hard_limit"],
    max_retries=YOUTUBE_CONFIG["RETRY_CONFIG"]["search"]["max_retries"],
    default_retry_delay=YOUTUBE_CONFIG["RETRY_CONFIG"]["search"]["countdown"],
    autoretry_for=(Exception,),
    retry_backoff=YOUTUBE_CONFIG["RETRY_CONFIG"]["search"]["backoff"],
    retry_jitter=YOUTUBE_CONFIG["RETRY_CONFIG"]["search"]["jitter"],
)
def process_search_query(self, search_id: str, max_videos: int = 5):
    """
    Process a search query asynchronously with real-time progress tracking

    This task handles:
    1. AI services initialization (MAPPING stage)
    2. LLM query enhancement and planning (MAPPING stage)
    3. YouTube search (EXPLORING stage)
    4. Database updates with results

    Args:
        search_id: UUID of the SearchRequest to process

    Returns:
        dict: Processing result with status and details
    """
    # Initialize progress tracker for real-time updates
    progress = ExplorerProgressTracker(search_id)
    progress.begin_expedition()

    search_request = None
    session = None

    try:
        # 🗺️ MAPPING STAGE: Understanding and Planning
        progress.start_stage("MAPPING")

        # Get search request from database
        try:
            search_request = SearchRequestModel.objects.select_related(
                "search_session"
            ).get(search_id=search_id)
            session = search_request.search_session
            original_query = search_request.original_query

            logger.debug(
                f"Processing search request: {search_id} for query: '{original_query}'"
            )

        except SearchRequestModel.DoesNotExist:
            error_msg = f"Search request {search_id} not found"
            logger.error(error_msg)
            progress.send_error("Search request not found")
            return {
                "status": "failed",
                "error": "Search request not found",
                "search_id": search_id,
            }

        # Initialize AI services
        try:
            # Strategic delay for realistic progress timing
            import time

            time.sleep(1)

            config = get_config()
            config.validate()

            # Use shared Gemini LLM service for query processing
            llm_service = get_gemini_llm_service()

            # get clients for LLM and YouTube search
            query_processor = QueryProcessor(llm_service=llm_service)

            # Configure YouTube search service with English-only and shorts filtering
            youtube_search_provider = ScrapeTubeProvider(
                max_results=max_videos,
                timeout=30,
                filter_shorts=True,  # Filter out YouTube shorts
                english_only=True,  # Only English videos
                min_duration_seconds=BUSINESS_LOGIC_CONFIG["DURATION_LIMITS"][
                    "minimum_seconds"
                ],
                max_duration_seconds=BUSINESS_LOGIC_CONFIG["DURATION_LIMITS"][
                    "maximum_seconds"
                ],
            )
            youtube_search_service = YouTubeSearchService(youtube_search_provider)

            logger.debug("AI services initialized with Gemini LLM provider")

        except Exception as e:
            error_msg = f"Failed to initialize AI services: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"Failed to initialize AI services: {str(e)}")

            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)

            return {
                "status": "failed",
                "error": "AI services initialization failed",
                "details": str(e),
                "search_id": search_id,
            }

        # Process query with LLM enhancement (async operation)
        try:
            # Strategic delay before LLM processing
            time.sleep(1.5)

            logger.debug("Starting LLM query processing")

            # Create a new event loop for Celery task
            # This is safer than trying to get existing loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Process query asynchronously with timeout
                # Use asyncio.wait_for to add a timeout
                enhancement_task = query_processor.enhance_query(original_query)
                enhancement_result = loop.run_until_complete(
                    asyncio.wait_for(
                        enhancement_task, timeout=10.0
                    )  # 10 second timeout
                )
            except TimeoutError:
                logger.error(
                    f"LLM query enhancement timed out after 10 seconds for query: '{original_query}'"
                )
                # Use smart fallback on timeout
                if " and " in original_query.lower():
                    parts = original_query.lower().split(" and ")
                    concepts = [part.strip() for part in parts if part.strip()]
                    enhanced_queries = [
                        f"{concept} tutorial english" for concept in concepts
                    ]
                    is_complex = True
                else:
                    concepts = [original_query]
                    enhanced_queries = [original_query]
                    is_complex = False
                enhancement_result = {
                    "status": "failed",
                    "error": "Timeout",
                    "concepts": concepts,
                    "enhanced_queries": enhanced_queries,
                    "intent_type": "",
                    "is_complex": is_complex,
                }
            finally:
                # Clean up the loop
                loop.close()

            # Extract enhanced queries, concepts and intent from result
            if enhancement_result.get("status") == "completed":
                enhanced_queries = enhancement_result.get(
                    "enhanced_queries", [original_query]
                )
                concepts = enhancement_result.get("concepts", [original_query])
                intent_type = enhancement_result.get("intent_type", "")
                is_complex = enhancement_result.get("is_complex", False)

                logger.info(
                    f"Enhanced query: '{original_query}' → {len(enhanced_queries)} queries (Intent: {intent_type}, Complex: {is_complex})"
                )
                logger.debug(f"Concepts: {concepts}")
                logger.debug(f"Enhanced queries: {enhanced_queries}")
            else:
                enhanced_queries = [original_query]
                concepts = [original_query]
                intent_type = ""
                is_complex = False
                logger.warning(
                    f"Query enhancement failed: {enhancement_result.get('error', 'Unknown error')}, using original query"
                )

        except Exception as e:
            error_msg = f"Query enhancement failed: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Add full traceback

            # Better fallback handling
            if " and " in original_query.lower():
                parts = original_query.lower().split(" and ")
                concepts = [part.strip() for part in parts if part.strip()]
                enhanced_queries = [
                    f"{concept} tutorial english" for concept in concepts
                ]
                is_complex = True
                logger.info(f"Using smart fallback: extracted {len(concepts)} concepts")
            else:
                enhanced_queries = [original_query]
                concepts = [original_query]
                is_complex = False
                logger.info("Using original query due to enhancement failure")
            intent_type = ""

        # Strategic delay for stage transition
        time.sleep(1)

        # ⛏️ EXPLORING STAGE: Discovery and Search
        progress.start_stage("EXPLORING")

        # Strategic delay before starting searches
        time.sleep(0.5)

        # Perform YouTube search for all enhanced queries (PARALLEL EXECUTION)
        try:
            all_video_urls = []
            seen_urls = set()  # For deduplication

            # Use ThreadPoolExecutor for parallel execution of synchronous YouTube searches
            import concurrent.futures

            logger.info(
                f"🚀 Starting {len(enhanced_queries)} YouTube searches in PARALLEL (instead of sequential)"
            )

            def execute_search(query_info):
                """Execute a single YouTube search"""
                query, search_idx = query_info
                try:
                    logger.debug(
                        f"Starting parallel YouTube search {search_idx + 1}/{len(enhanced_queries)} for: '{query}'"
                    )

                    # Strategic delay between searches
                    time.sleep(0.8)

                    search_request_obj = SearchRequest(
                        query=query, max_results=max_videos
                    )

                    search_response = youtube_search_service.search(search_request_obj)
                    query_video_urls = search_response.results

                    logger.info(
                        f"Parallel search {search_idx + 1} completed: Found {len(query_video_urls)} videos for '{query}'"
                    )
                    return query_video_urls

                except Exception as e:
                    logger.error(
                        f"Parallel search failed for query '{query}': {str(e)}"
                    )
                    return []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(enhanced_queries), 5)
            ) as executor:
                # Prepare query info for parallel execution
                query_tasks = [(query, i) for i, query in enumerate(enhanced_queries)]

                # Submit all search tasks and collect results
                future_to_query = {
                    executor.submit(execute_search, task): task for task in query_tasks
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(
                    future_to_query, timeout=30
                ):
                    try:
                        query_video_urls = future.result()

                        # Add unique URLs to the combined result
                        for url in query_video_urls:
                            if url not in seen_urls:
                                seen_urls.add(url)
                                all_video_urls.append(url)

                    except Exception as e:
                        logger.error(
                            f"Failed to get result from parallel search: {str(e)}"
                        )

            video_urls = all_video_urls
            logger.info(
                f"✅ Parallel search completed! Total unique videos found: {len(video_urls)}"
            )

            if not video_urls:
                logger.warning("No videos found for any of the enhanced queries")

        except Exception as e:
            error_msg = f"YouTube search failed: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"YouTube search failed: {str(e)}")

            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)

            return {
                "status": "failed",
                "error": "YouTube search failed",
                "details": str(e),
                "search_id": search_id,
            }

        # Strategic delay after search completion
        time.sleep(1)

        # Update database with results
        try:
            logger.info("About to save to database:")
            logger.info(f"  - concepts type: {type(concepts)}, value: {concepts}")
            logger.info(
                f"  - enhanced_queries type: {type(enhanced_queries)}, value: {enhanced_queries}"
            )
            logger.info(f"  - intent_type: {intent_type}")
            logger.info(f"  - video_urls count: {len(video_urls)}")

            with transaction.atomic():
                search_request.concepts = concepts
                search_request.enhanced_queries = enhanced_queries
                search_request.intent_type = intent_type
                search_request.video_urls = video_urls
                search_request.total_videos = len(video_urls)
                search_request.status = "success"
                search_request.save()

                # Verify what was saved
                search_request.refresh_from_db()
                logger.info(f"After save - concepts from DB: {search_request.concepts}")

                # Update session status to success
                update_session_status(session, "success")

                logger.info(
                    f"Search request {search_id} completed successfully with {len(video_urls)} videos"
                )

        except Exception as e:
            error_msg = f"Failed to update search results: {str(e)}"
            logger.error(error_msg)
            progress.send_error(f"Failed to save results: {str(e)}")

            _update_search_request_error(search_request, error_msg)
            _update_session_error(session)

            return {
                "status": "failed",
                "error": "Failed to update search results",
                "details": str(e),
                "search_id": search_id,
            }

        # Search phase completed successfully
        logger.info(f"Search phase completed for {search_id}")

        return {
            "status": "success",
            "search_id": search_id,
            "enhanced_queries": enhanced_queries,
            "concepts": concepts,
            "video_urls": video_urls,
            "total_videos": len(video_urls),
        }

    except SoftTimeLimitExceeded:
        # Search processing is approaching timeout
        logger.warning(f"Search processing soft timeout reached for search {search_id}")
        progress.send_error("Search processing timeout - query may be too complex")

        try:
            # Mark search as failed due to timeout
            if search_request:
                _update_search_request_error(
                    search_request,
                    "Search processing timed out - query may be too complex",
                )
            if session:
                _update_session_error(session)
            logger.error(
                f"Marked search processing as failed due to timeout: {search_id}"
            )

        except Exception as cleanup_error:
            logger.error(
                f"Failed to update search status during timeout cleanup: {cleanup_error}"
            )

        # Re-raise to mark task as failed
        raise Exception(f"Search processing timeout for search {search_id}")

    except Exception as e:
        logger.error(f"Unexpected error in search processing: {e}")
        progress.send_error(f"Unexpected error: {str(e)}")

        # Handle dead letter queue on final failure
        if self.request.retries >= self.max_retries:
            handle_dead_letter_task(
                "process_search_query", self.request.id, [search_id], {}, e
            )

        try:
            if search_request:
                _update_search_request_error(
                    search_request, f"Unexpected error: {str(e)}"
                )
            if session:
                _update_session_error(session)
        except:
            pass

        return {
            "status": "failed",
            "error": "Unexpected error in search processing",
            "details": str(e),
            "search_id": search_id,
        }


@shared_task(
    bind=True,
    name="topic.process_search_with_videos",
    soft_time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["parallel_soft_limit"],
    time_limit=YOUTUBE_CONFIG["TASK_TIMEOUTS"]["parallel_hard_limit"],
    autoretry_for=(Exception,),
    retry_backoff=YOUTUBE_CONFIG["RETRY_CONFIG"]["parallel"]["backoff"],
    retry_jitter=YOUTUBE_CONFIG["RETRY_CONFIG"]["parallel"]["jitter"],
)
def process_search_with_videos(
    self, search_id: str, max_videos: int = 5, start_video_processing: bool = False
):
    """
    Combined search and video processing workflow.

    Args:
        search_id: UUID of the SearchRequest to process
        max_videos: Maximum number of videos to find and process
        start_video_processing: Whether to immediately start video processing

    Returns:
        dict: Combined search and processing result
    """
    logger.info(f"Starting combined search and video processing for {search_id}")

    search_request = None

    try:
        # First, run the search
        search_result = process_search_query(search_id, max_videos)

        if search_result["status"] != "success":
            return search_result

        # If video processing is requested, start it
        if start_video_processing and search_result.get("video_urls"):
            logger.info(
                f"Starting video processing for {len(search_result['video_urls'])} videos"
            )

            try:
                # Import here to avoid circular imports
                from topic.parallel_tasks import process_search_results

                # Start video processing
                processing_task = process_search_results.delay(search_id)

                search_result["video_processing_task_id"] = processing_task.id
                search_result["video_processing_started"] = True

            except Exception as e:
                logger.error(f"Failed to start video processing: {e}")
                search_result["video_processing_error"] = str(e)
                search_result["video_processing_started"] = False

        return search_result

    except SoftTimeLimitExceeded:
        # Combined processing is approaching timeout
        logger.warning(
            f"Combined search and video processing soft timeout reached for search {search_id}"
        )

        try:
            # Mark search as failed due to timeout
            if search_request:
                _update_search_request_error(
                    search_request, "Combined processing timed out"
                )
            logger.error(
                f"Marked combined processing as failed due to timeout: {search_id}"
            )

        except Exception as cleanup_error:
            logger.error(
                f"Failed to update status during combined processing timeout cleanup: {cleanup_error}"
            )

        # Re-raise to mark task as failed
        raise Exception(
            f"Combined search and video processing timeout for search {search_id}"
        )

    except Exception as e:
        logger.error(f"Error in combined search and video processing: {e}")
        return {
            "status": "failed",
            "error": "Combined processing failed",
            "details": str(e),
            "search_id": search_id,
        }


def _update_search_request_error(
    search_request: SearchRequestModel, error_message: str
):
    """Helper function to update search request with error"""
    try:
        search_request.status = "failed"
        search_request.error_message = error_message
        search_request.save()
    except Exception as e:
        logger.error(f"Failed to update search request with error: {e}")


def _update_session_error(session: SearchSession):
    """Helper function to update session with error status"""
    try:
        update_session_status(session, "failed")
    except Exception as e:
        logger.error(f"Failed to update session with error: {e}")
