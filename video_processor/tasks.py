from celery import shared_task
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from api.models import URLRequestTable
from .models import VideoMetadata, VideoTranscript, update_url_request_status
from .serializer import VideoMetadataSerializer, VideoTranscriptSerializer
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_youtube_video(self, url_request_id):
    url_request = None
    metadata_obj = None
    transcript_obj = None
    
    try:
        url_request = URLRequestTable.objects.get(id=url_request_id)
        logger.info(f"Processing video for request {url_request_id}")
        
        # Configure yt-dlp with timeouts and error handling
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 2,
        }
        
        # Step 1: Extract video metadata
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_request.url, download=False)
                
                # Create metadata using serializer
                metadata_serializer = VideoMetadataSerializer(data={
                    'url_request': url_request.id,
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'duration': info.get('duration'),
                    'channel_name': info.get('uploader', ''),
                    'view_count': info.get('view_count'),
                    'status': 'processing'
                })
                
                if metadata_serializer.is_valid():
                    metadata_obj = metadata_serializer.save()
                    # Mark metadata as successful
                    metadata_obj.status = 'success'
                    metadata_obj.save()
                    logger.info(f"Successfully extracted metadata for video: {info.get('title')}")
                else:
                    logger.error(f"Metadata validation failed: {metadata_serializer.errors}")
                    raise ValueError(f"Invalid metadata: {metadata_serializer.errors}")
                
                video_id = info.get('id')
                
        except Exception as e:
            logger.error(f"Metadata extraction failed for request {url_request_id}: {str(e)}")
            if metadata_obj:
                metadata_obj.status = 'failed'
                metadata_obj.save()
            else:
                # Create failed metadata record using serializer
                failed_metadata_serializer = VideoMetadataSerializer(data={
                    'url_request': url_request.id,
                    'status': 'failed'
                })
                if failed_metadata_serializer.is_valid():
                    failed_metadata_serializer.save()
            update_url_request_status(url_request)
            raise
        
        # Step 2: Extract video transcript
        try:
            if video_id:
                # Create transcript using serializer
                transcript_serializer = VideoTranscriptSerializer(data={
                    'url_request': url_request.id,
                    'transcript_text': '',
                    'status': 'processing'
                })
                
                if transcript_serializer.is_valid():
                    transcript_obj = transcript_serializer.save()
                    
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = ' '.join([item['text'] for item in transcript])
                    
                    # Update transcript with actual text
                    updated_transcript_serializer = VideoTranscriptSerializer(
                        transcript_obj,
                        data={
                            'url_request': url_request.id,
                            'transcript_text': transcript_text,
                            'status': 'success'
                        },
                        partial=True
                    )
                    
                    if updated_transcript_serializer.is_valid():
                        transcript_obj = updated_transcript_serializer.save()
                        logger.info(f"Successfully extracted transcript for video {video_id}")
                    else:
                        logger.error(f"Transcript update validation failed: {updated_transcript_serializer.errors}")
                        raise ValueError(f"Invalid transcript update: {updated_transcript_serializer.errors}")
                else:
                    logger.error(f"Transcript validation failed: {transcript_serializer.errors}")
                    raise ValueError(f"Invalid transcript: {transcript_serializer.errors}")
            else:
                raise Exception("No video ID found for transcript extraction")
                
        except Exception as e:
            logger.warning(f"Transcript extraction failed for video {video_id}: {e}")
            if transcript_obj:
                transcript_obj.status = 'failed'
                transcript_obj.save()
            else:
                # Create failed transcript record using serializer
                failed_transcript_serializer = VideoTranscriptSerializer(data={
                    'url_request': url_request.id,
                    'transcript_text': '',
                    'status': 'failed'
                })
                if failed_transcript_serializer.is_valid():
                    failed_transcript_serializer.save()
        
        # Step 3: Update overall status based on both components
        update_url_request_status(url_request)
        
        # Refresh to get updated status
        url_request.refresh_from_db()
        logger.info(f"Final status for request {url_request_id}: {url_request.status}")
        
        return f"Processed video: {info.get('title', 'Unknown')} - Status: {url_request.status}"
        
    except Exception as e:
        logger.error(f"Critical error processing video for request {url_request_id}: {str(e)}")
        
        # On final failure, ensure everything is marked as failed
        if self.request.retries >= self.max_retries:
            if url_request:
                # Mark any existing records as failed
                if hasattr(url_request, 'video_metadata') and url_request.video_metadata.status == 'processing':
                    url_request.video_metadata.status = 'failed'
                    url_request.video_metadata.save()
                    
                if hasattr(url_request, 'video_transcript') and url_request.video_transcript.status == 'processing':
                    url_request.video_transcript.status = 'failed'
                    url_request.video_transcript.save()
                
                update_url_request_status(url_request)
                logger.error(f"Final failure for request {url_request_id} after {self.request.retries} retries")
        
        # Re-raise to trigger retry
        raise