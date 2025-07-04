from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.models import URLRequestTable
from .models import VideoTranscript
from .serializers import VideoTranscriptSerializer

# Create your views here.

@api_view(['GET'])
def transcript_health_check(request):
    """
    Health check for transcript processing
    """
    try:
        total_transcripts = VideoTranscript.objects.count()
        successful_transcripts = VideoTranscript.objects.filter(status='success').count()
        transcripts_with_data = VideoTranscript.objects.filter(
            status='success',
            transcript_data__isnull=False
        ).count()
        
        health_status = {
            'total_transcripts': total_transcripts,
            'successful_transcripts': successful_transcripts,
            'transcripts_with_timestamps': transcripts_with_data,
            'timestamp_coverage': round(
                (transcripts_with_data / successful_transcripts * 100) if successful_transcripts > 0 else 0, 2
            ),
            'status': 'healthy' if transcripts_with_data == successful_transcripts else 'degraded'
        }
        
        return Response(health_status, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e), 'status': 'unhealthy'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_transcript_with_timestamps(request, request_id):
    """
    Get transcript data with timestamps for interactive UI
    """
    try:
        url_request = URLRequestTable.objects.get(request_id=request_id)
        
        # Check if VideoMetadata exists
        if not hasattr(url_request, 'video_metadata'):
            return Response(
                {'error': 'Video metadata not found for this request'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if VideoTranscript exists through VideoMetadata
        if not hasattr(url_request.video_metadata, 'video_transcript'):
            return Response(
                {'error': 'Transcript not found for this request'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        transcript = url_request.video_metadata.video_transcript
        
        if transcript.status != 'success':
            return Response(
                {'error': f'Transcript status: {transcript.status}'}, 
                status=status.HTTP_200_OK
            )
        
        serializer = VideoTranscriptSerializer(transcript)
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    except URLRequestTable.DoesNotExist:
        return Response(
            {'error': 'Request not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
