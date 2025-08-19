"""
SearchProgressAggregator - Event-driven stage progress tracking for topic search processing.

Aggregates individual video completion events from Redis to trigger accurate stage transitions
based on actual processing completion rather than arbitrary time delays.
"""

import json
import time
import logging
from typing import List, Set, Dict, Optional
import redis
from django.conf import settings

from .explorer_progress import ExplorerProgressTracker

logger = logging.getLogger(__name__)

# Stage transition configuration
STAGE_THRESHOLDS = {
    'ANALYZING': {
        'trigger_stage': 'TRANSCRIPT',
        'threshold_pct': 60,
        'description': 'Majority of videos have transcripts extracted'
    },
    'TREASURE_READY': {
        'trigger_stage': 'EMBEDDING', 
        'threshold_pct': 80,
        'description': 'Most videos are embedded and analyzed'
    }
}

# Stage processing order
STAGE_ORDER = ['METADATA', 'TRANSCRIPT', 'SUMMARY', 'EMBEDDING', 'COMPLETE']

# Monitoring configuration
COMPLETION_THRESHOLD = 95  # Consider search complete at 95% to allow for failures
MONITORING_TIMEOUT = 900   # 15 minutes max monitoring duration
REDIS_MESSAGE_TIMEOUT = 1.0  # 1 second timeout for Redis message polling


class SearchProgressAggregator:
    """
    Aggregates video completion events to trigger accurate stage transitions.
    
    Listens to Redis pattern 'video.*.progress' and filters events for videos
    belonging to the current search, then triggers stage transitions when
    completion thresholds are reached.
    """
    
    def __init__(self, search_id: str, url_request_ids: List[str], progress_tracker: ExplorerProgressTracker):
        """
        Initialize progress aggregator for a search.
        
        Args:
            search_id: UUID of the search request
            url_request_ids: List of URLRequestTable UUIDs for this search
            progress_tracker: ExplorerProgressTracker instance for SSE emission
        """
        self.search_id = search_id
        # Convert UUID objects to strings for Redis event comparison
        self.url_request_ids = set(str(uid) for uid in url_request_ids)  # O(1) lookup for filtering
        self.total_videos = len(url_request_ids)
        self.progress = progress_tracker
        
        # Track completion counts by stage
        self.stage_completions: Dict[str, Set[str]] = {
            'METADATA': set(),
            'TRANSCRIPT': set(), 
            'SUMMARY': set(),
            'EMBEDDING': set(),
            'COMPLETE': set()
        }
        
        # Track which stages we've already triggered
        self.triggered_stages: Set[str] = set()
        
        # Redis connection setup
        redis_config = {
            'host': getattr(settings, 'REDIS_HOST', 'localhost'),
            'port': getattr(settings, 'REDIS_PORT', 6379),
            'db': 3,  # Same database as ExplorerProgressTracker
            'decode_responses': True
        }
        
        try:
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()  # Test connection
            logger.info(f"SearchProgressAggregator initialized for search {search_id} with {self.total_videos} videos")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed for SearchProgressAggregator: {e}")
            raise Exception(f"Redis connection required for progress tracking: {e}")
    
    def monitor_video_stages(self) -> None:
        """
        Main monitoring loop - subscribes to Redis events and processes until completion.
        
        This method blocks until all videos complete or timeout is reached.
        """
        logger.info(f"Starting video stage monitoring for search {self.search_id}")
        
        # CRITICAL FIX: Check for already-processed videos and mark them as complete
        all_videos_processed = self._initialize_existing_video_completions()
        
        # If all videos are already processed, no need to monitor Redis events
        if all_videos_processed:
            return
        
        # Setup Redis subscription for remaining videos
        pubsub = self._setup_redis_subscription()
        if not pubsub:
            raise Exception("Failed to setup Redis subscription")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < MONITORING_TIMEOUT:
                # Get message with timeout
                message = pubsub.get_message(timeout=REDIS_MESSAGE_TIMEOUT)
                
                if message and message['type'] == 'pmessage':
                    logger.error(f"🔴 AGGREGATOR RECEIVED: {message}")
                    self._process_video_event(message)
                elif message:
                    logger.debug(f"SearchProgressAggregator received non-pmessage: {message['type']}")
                
                # Check if search is complete
                completion_pct = (len(self.stage_completions['COMPLETE']) / self.total_videos) * 100
                if completion_pct >= COMPLETION_THRESHOLD:
                    logger.info(f"Search {self.search_id} complete: {len(self.stage_completions['COMPLETE'])}/{self.total_videos} videos finished")
                    self.progress.expedition_complete()
                    break
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
            
            # Handle timeout case
            if time.time() - start_time >= MONITORING_TIMEOUT:
                logger.warning(f"Monitoring timeout reached for search {self.search_id} - forcing completion")
                self._force_completion_with_available_data()
                
        except Exception as e:
            logger.error(f"Error during video stage monitoring: {e}")
            raise
        finally:
            try:
                pubsub.close()
            except Exception as e:
                logger.warning(f"Error closing Redis subscription: {e}")
    
    def _initialize_existing_video_completions(self) -> bool:
        """
        Check for already-processed videos and mark them as complete in all stages.
        This handles the case where videos were processed in previous searches.
        
        Returns:
            bool: True if all videos are already processed, False otherwise
        """
        try:
            from api.models import URLRequestTable
            from video_processor.models import VideoMetadata
            from topic.models import SearchRequest as SearchRequestModel
            
            # Get all URLRequestTable entries for this search
            search_request = SearchRequestModel.objects.get(search_id=self.search_id)
            url_requests = URLRequestTable.objects.filter(search_request=search_request)
            
            already_processed_count = 0
            
            for url_request in url_requests:
                try:
                    # Check if VideoMetadata exists and is successful
                    video_metadata = VideoMetadata.objects.get(url_request=url_request)
                    if video_metadata.status == 'success':
                        request_id = str(url_request.request_id)
                        
                        # Mark this video as complete in all stages
                        for stage in STAGE_ORDER:
                            if request_id not in self.stage_completions[stage]:
                                self.stage_completions[stage].add(request_id)
                        
                        already_processed_count += 1
                        logger.info(f"Marked already-processed video {request_id[:8]} as complete in all stages")
                        
                except VideoMetadata.DoesNotExist:
                    # Video not yet processed, will wait for Redis events
                    pass
            
            if already_processed_count > 0:
                logger.info(f"Found {already_processed_count}/{self.total_videos} already-processed videos")
                
                # Trigger immediate stage transitions based on current completion rates
                self._check_stage_transitions()
                
                # If all videos are already processed, complete immediately
                if already_processed_count == self.total_videos:
                    logger.info(f"All {self.total_videos} videos already processed - completing search immediately")
                    self.progress.expedition_complete()
                    return True
            else:
                logger.info(f"No already-processed videos found - waiting for Redis events")
                
            return False
                
        except Exception as e:
            logger.error(f"Error initializing existing video completions: {e}")
            # Don't fail the monitoring - continue with Redis events
            return False
    
    def _setup_redis_subscription(self) -> Optional[redis.client.PubSub]:
        """
        Setup Redis pattern subscription for video events.
        
        Returns:
            PubSub instance or None if setup fails
        """
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.psubscribe('video.*.progress')
            
            logger.info(f"Subscribed to Redis pattern 'video.*.progress' for search {self.search_id}")
            return pubsub
            
        except Exception as e:
            logger.error(f"Failed to setup Redis subscription: {e}")
            return None
    
    def _process_video_event(self, message) -> None:
        """
        Process individual video completion events from Redis.
        
        Args:
            message: Redis pub/sub message
        """
        try:
            # Parse message data
            data = json.loads(message['data'])
            request_id = data.get('request_id')
            
            # Validate message structure
            if not self._validate_event_data(data):
                return
            
            # Filter: Only process events from our search's videos
            if request_id not in self.url_request_ids:
                logger.error(f"🔴 FILTERED OUT: Video {request_id[:8]} not in search {self.search_id}")
                return  # Different search's video, ignore
            
            event_type = data.get('type')
            stage = data.get('stage')
            
            logger.error(f"🔴 PROCESSING EVENT: {event_type} for video {request_id[:8]} in search {self.search_id}")
            if stage:
                stage = stage.upper()
            
            # Track stage completions
            if event_type == 'stage_complete' and stage in self.stage_completions:
                self.stage_completions[stage].add(request_id)
                logger.info(f"Video {request_id[:8]} completed {stage} stage ({len(self.stage_completions[stage])}/{self.total_videos})")
                
            elif event_type == 'complete':
                self.stage_completions['COMPLETE'].add(request_id)
                logger.info(f"Video {request_id[:8]} fully complete ({len(self.stage_completions['COMPLETE'])}/{self.total_videos})")
            
            # Check for stage transitions
            self._check_stage_transitions()
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in Redis message: {e}")
        except Exception as e:
            logger.error(f"Error processing video event: {e}")
    
    def _validate_event_data(self, data: dict) -> bool:
        """
        Validate Redis event data structure.
        
        Args:
            data: Parsed JSON data from Redis message
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['request_id', 'type', 'timestamp']
        return all(field in data for field in required_fields)
    
    def _check_stage_transitions(self) -> None:
        """
        Check if thresholds are met and emit stage transitions.
        
        This is called after each video event to see if we should trigger
        the next stage based on completion percentages.
        """
        for stage_name, config in STAGE_THRESHOLDS.items():
            # Skip if already triggered
            if stage_name in self.triggered_stages:
                continue
            
            # Calculate completion percentage for the trigger stage
            trigger_stage = config['trigger_stage']
            completed_count = len(self.stage_completions[trigger_stage])
            completion_pct = (completed_count / self.total_videos) * 100
            
            # Check threshold
            if completion_pct >= config['threshold_pct']:
                logger.info(f"Triggering {stage_name} stage for search {self.search_id}: "
                           f"{completed_count}/{self.total_videos} videos completed {trigger_stage} "
                           f"({completion_pct:.1f}% >= {config['threshold_pct']}%)")
                
                # Emit stage transition
                self.progress.start_stage(stage_name)
                self.triggered_stages.add(stage_name)
    
    def _force_completion_with_available_data(self) -> None:
        """
        Force completion based on available data when timeout is reached.
        
        This ensures the search doesn't hang indefinitely if some videos
        fail to complete or send events.
        """
        completed_videos = len(self.stage_completions['COMPLETE'])
        total_videos = self.total_videos
        
        logger.warning(f"Forcing completion for search {self.search_id}: "
                      f"{completed_videos}/{total_videos} videos completed")
        
        # Trigger any missing stages based on available data
        for stage_name, config in STAGE_THRESHOLDS.items():
            if stage_name not in self.triggered_stages:
                trigger_stage = config['trigger_stage']
                completed_count = len(self.stage_completions[trigger_stage])
                
                # Use a lower threshold for timeout scenarios
                timeout_threshold = max(50, config['threshold_pct'] - 20)  # At least 50%, or 20% lower than normal
                completion_pct = (completed_count / total_videos) * 100
                
                if completion_pct >= timeout_threshold:
                    logger.info(f"Timeout trigger: {stage_name} stage with {completion_pct:.1f}% completion")
                    self.progress.start_stage(stage_name)
                    self.triggered_stages.add(stage_name)
        
        # Force expedition complete
        self.progress.expedition_complete()
    
    def get_completion_stats(self) -> Dict[str, any]:
        """
        Get current completion statistics for debugging/monitoring.
        
        Returns:
            Dictionary with completion stats
        """
        return {
            'search_id': self.search_id,
            'total_videos': self.total_videos,
            'stage_completions': {
                stage: len(completed_set) for stage, completed_set in self.stage_completions.items()
            },
            'triggered_stages': list(self.triggered_stages),
            'completion_percentages': {
                stage: (len(completed_set) / self.total_videos) * 100 
                for stage, completed_set in self.stage_completions.items()
            }
        }
