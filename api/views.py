from .serializers import URLRequestTableSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .utils.get_client_ip import get_client_ip
from django.http import StreamingHttpResponse
from .models import URLRequestTable
from video_processor.tasks import process_youtube_video
from video_processor.config import validate_youtube_url
import json
import time

@api_view(['POST'])
def summarise_single(request):
    # Get client IP address
    ip_address = get_client_ip(request)
    
    # Get URL from request body
    url = request.data.get('url')

    if not url:
        return Response({'error': 'URL is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate YouTube URL format
    try:
        validate_youtube_url(url)
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    # Prepare data for serializer
    data = {
        'url': url,
        'ip_address': ip_address,
        'status': 'processing'
    }
    
    # Use serializer to save data
    serializer = URLRequestTableSerializer(data=data)
    if serializer.is_valid():
        url_request = serializer.save()
        
        # Trigger video processing
        process_youtube_video.delay(url_request.id)
        
        # Return response in requested format with request_id
        return Response({
            'request_id': str(url_request.request_id),
            'url': url,
            'status': 'processing'
        }, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Stream the status of the request
def status_stream(request, request_id):
    def event_stream():
        max_attempts = 60  
        attempts = 0
        
        # Maximum 2 minutes (60 * 2 seconds)
        while attempts < max_attempts:
            try:
                url_request = URLRequestTable.objects.get(request_id=request_id)
                
                # Build status data
                data = {
                    'overall_status': url_request.status,
                    'timestamp': time.time(),
                    'metadata_status': None,
                    'transcript_status': None,
                }
                
                # Add metadata details if exists
                if hasattr(url_request, 'video_metadata'):
                    metadata = url_request.video_metadata
                    data['metadata_status'] = metadata.status
                
                # Add transcript details if exists
                if hasattr(url_request, 'video_transcript'):
                    transcript = url_request.video_transcript
                    data['transcript_status'] = transcript.status
                
                # Send data
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop streaming if complete
                if url_request.status in ['success', 'failed']:
                    break
                    
                time.sleep(2)  # Wait 2 seconds
                attempts += 1
                
            except URLRequestTable.DoesNotExist:
                yield f"data: {json.dumps({'error': 'Request not found'})}\n\n"
                break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response