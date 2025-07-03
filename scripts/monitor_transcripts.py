#!/usr/bin/env python
"""
Transcript monitoring and auto-repair script
Run this periodically to ensure transcript reliability
"""

import os
import sys
import django
import requests
import logging
from datetime import datetime

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yt_summariser.settings')
django.setup()

from video_processor.models import VideoTranscript
from django.core.management import call_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_transcript_health():
    """Check the health of transcript processing"""
    try:
        total = VideoTranscript.objects.count()
        successful = VideoTranscript.objects.filter(status='success').count()
        with_timestamps = VideoTranscript.objects.filter(
            status='success',
            transcript_data__isnull=False
        ).count()
        
        coverage = (with_timestamps / successful * 100) if successful > 0 else 0
        
        logger.info(f"Transcript Health Check:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  With Timestamps: {with_timestamps}")
        logger.info(f"  Coverage: {coverage:.1f}%")
        
        return {
            'total': total,
            'successful': successful,
            'with_timestamps': with_timestamps,
            'coverage': coverage,
            'healthy': coverage >= 95.0  # Consider healthy if 95%+ have timestamps
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {'healthy': False, 'error': str(e)}

def repair_transcripts():
    """Repair transcripts missing timestamp data"""
    try:
        broken_transcripts = VideoTranscript.objects.filter(
            status='success',
            transcript_data__isnull=True
        )
        
        if broken_transcripts.exists():
            logger.info(f"Found {broken_transcripts.count()} transcripts needing repair")
            
            # Run the reprocess command
            call_command('reprocess_transcripts', '--all', verbosity=0)
            logger.info("Transcript repair completed")
            return True
        else:
            logger.info("No transcripts need repair")
            return True
            
    except Exception as e:
        logger.error(f"Transcript repair failed: {e}")
        return False

def validate_transcript_structure():
    """Validate that all transcripts have proper structure"""
    invalid_count = 0
    
    for transcript in VideoTranscript.objects.filter(status='success'):
        try:
            if transcript.transcript_data:
                # Check if it's a list
                if not isinstance(transcript.transcript_data, list):
                    logger.warning(f"Transcript {transcript.id} has invalid data structure (not a list)")
                    invalid_count += 1
                    continue
                
                # Check each segment
                for i, segment in enumerate(transcript.transcript_data):
                    if not isinstance(segment, dict):
                        logger.warning(f"Transcript {transcript.id} segment {i} is not a dict")
                        invalid_count += 1
                        break
                    
                    if 'text' not in segment or 'start' not in segment:
                        logger.warning(f"Transcript {transcript.id} segment {i} missing required fields")
                        invalid_count += 1
                        break
                        
        except Exception as e:
            logger.error(f"Error validating transcript {transcript.id}: {e}")
            invalid_count += 1
    
    logger.info(f"Validation complete. Found {invalid_count} invalid transcripts")
    return invalid_count == 0

def main():
    """Main monitoring function"""
    logger.info(f"Starting transcript monitoring at {datetime.now()}")
    
    # 1. Check health
    health = check_transcript_health()
    
    if not health.get('healthy', False):
        logger.warning("Transcript system is not healthy - attempting repair")
        
        # 2. Attempt repair
        if repair_transcripts():
            # 3. Re-check health
            health = check_transcript_health()
            if health.get('healthy', False):
                logger.info("Repair successful - system is now healthy")
            else:
                logger.error("Repair completed but system still unhealthy")
        else:
            logger.error("Repair failed")
    
    # 4. Validate structure
    if not validate_transcript_structure():
        logger.warning("Some transcripts have invalid structure")
    
    logger.info("Monitoring complete")
    return health.get('healthy', False)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 