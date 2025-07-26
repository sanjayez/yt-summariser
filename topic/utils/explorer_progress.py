"""
Explorer-themed progress tracking for YouTube summarizer pipeline.
Sends real-time progress updates via Redis pub/sub for SSE delivery.
"""

import redis
import json
import time
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class ExplorerProgressTracker:
    """
    progress tracking with Redis pub/sub for real-time updates.
    
    Safely uses Redis database 3 to avoid conflicts with:
    - Database 0: Celery broker
    - Database 1: Celery results  
    - Database 2: Celery cache
    """
    
    # Explorer adventure stage definitions
    STAGES = {
        'MAPPING': {
            'message': '🗺️ Mapping the knowledge terrain...',
            'progress_range': (0, 15),
            'completion': '🗺️ Navigation route planned!'
        },
        'EXPLORING': {
            'message': '⛏️ Venturing into uncharted territories...',
            'progress_range': (15, 35),
            'completion': '⛏️ Found {count} promising expedition sites!'
        },
        'EXCAVATING': {
            'message': '🔍 Excavating valuable insights...',
            'progress_range': (35, 75),
            'completion': '🔍 Archaeological dig complete!'
        },
        'ANALYZING': {  # FUTURE FEATURE
            'message': '🔬 Analyzing discoveries in the lab...',
            'progress_range': (75, 90),
            'completion': '🔬 Analysis reveals fascinating patterns!'
        },
        'TREASURE_READY': {
            'message': '💎 Your treasure trove of answers is ready!',
            'progress_range': (90, 100),
            'completion': '💎 Adventure complete! Enjoy your discoveries!'
        }
    }
    
    def __init__(self, search_id):
        """
        Initialize progress tracker for a specific search expedition.
        
        Args:
            search_id: Unique identifier for the search request
        """
        self.search_id = str(search_id)
        
        # Use dedicated Redis database (3) for progress tracking
        # This avoids conflicts with existing Celery Redis usage
        redis_config = {
            'host': getattr(settings, 'REDIS_HOST', 'localhost'),
            'port': getattr(settings, 'REDIS_PORT', 6379),
            'db': 3,  # Dedicated database for progress tracking
            'decode_responses': True
        }
        
        try:
            self.redis_client = redis.Redis(**redis_config)
            # Test connection
            self.redis_client.ping()
            self.channel = f'search.{self.search_id}.progress'
            logger.debug(f"ExplorerProgressTracker initialized for search {search_id}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis for progress tracking: {e}")
            self.redis_client = None
    
    def _publish(self, message_type, data):
        """
        Publish progress update to Redis channel.
        
        Args:
            message_type: Type of progress message
            data: Message data dictionary
        """
        if not self.redis_client:
            logger.warning("Redis client not available, skipping progress update")
            return
            
        try:
            payload = {
                'type': message_type,
                'search_id': self.search_id,
                'timestamp': time.time(),
                **data
            }
            
            self.redis_client.publish(self.channel, json.dumps(payload))
            logger.debug(f"Published {message_type} to {self.channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish progress update: {e}")
    
    def begin_expedition(self):
        """Start the knowledge exploration adventure"""
        self._publish('expedition_start', {
            'message': '🚀 Explorer expedition starting...'
        })
    
    def start_stage(self, stage_code):
        """
        Begin a new exploration stage.
        
        Args:
            stage_code: Stage identifier from STAGES dict
        """
        if stage_code not in self.STAGES:
            logger.error(f"Unknown stage: {stage_code}")
            return
            
        stage = self.STAGES[stage_code]
        progress = stage['progress_range'][0]
        
        self._publish('stage_start', {
            'stage': stage_code,
            'message': stage['message'],
            'progress': progress
        })
    
    def update_progress(self, stage_code, stage_progress, custom_message=None):
        """
        Update progress within current stage.
        
        Args:
            stage_code: Current stage identifier
            stage_progress: Progress within stage (0-100)
            custom_message: Optional custom progress message
        """
        if stage_code not in self.STAGES:
            logger.error(f"Unknown stage: {stage_code}")
            return
            
        stage = self.STAGES[stage_code]
        min_prog, max_prog = stage['progress_range']
        
        # Map stage progress to overall progress
        overall_progress = min_prog + (stage_progress * (max_prog - min_prog) / 100)
        
        self._publish('stage_progress', {
            'stage': stage_code,
            'message': custom_message or stage['message'],
            'progress': round(overall_progress, 1),
            'stage_progress': stage_progress
        })
    
    def complete_stage(self, stage_code, **format_vars):
        """
        Complete current stage with celebration message.
        
        Args:
            stage_code: Stage identifier to complete
            **format_vars: Variables for formatting completion message
        """
        if stage_code not in self.STAGES:
            logger.error(f"Unknown stage: {stage_code}")
            return
            
        stage = self.STAGES[stage_code]
        progress = stage['progress_range'][1]
        
        # Format completion message with provided variables
        try:
            completion_msg = stage['completion'].format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing format variable for completion message: {e}")
            completion_msg = stage['completion']
        
        self._publish('stage_complete', {
            'stage': stage_code,
            'message': completion_msg,
            'progress': progress
        })
    
    def send_error(self, error_message, stage=None):
        """
        Send error notification with adventure theming.
        
        Args:
            error_message: Error description
            stage: Optional current stage where error occurred
        """
        self._publish('error', {
            'message': f"🚨 Expedition encountered difficulty: {error_message}",
            'stage': stage
        })
    
    def expedition_complete(self):
        """Adventure successfully completed!"""
        self._publish('complete', {
            'message': '🎉 Knowledge expedition completed successfully!'
        })
    
    def get_connection_status(self):
        """
        Check Redis connection status.
        
        Returns:
            bool: True if Redis is connected, False otherwise
        """
        if not self.redis_client:
            return False
            
        try:
            self.redis_client.ping()
            return True
        except:
            return False 