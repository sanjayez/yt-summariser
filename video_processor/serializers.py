from rest_framework import serializers
from .models import VideoMetadata, VideoTranscript

class VideoMetadataSerializer(serializers.ModelSerializer):
    # Add computed properties
    duration_string = serializers.ReadOnlyField()
    webpage_url = serializers.ReadOnlyField()
    
    class Meta:
        model = VideoMetadata
        fields = [
            'id',
            'url_request',
            'video_id',
            'title',
            'description',
            'duration',
            'duration_string',
            'channel_name',
            'view_count',
            'upload_date',
            'language',
            'like_count',
            'channel_id',
            'tags',
            'categories',
            'thumbnail',
            'channel_follower_count',
            'channel_is_verified',
            'uploader_id',
            'webpage_url',
            'status',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'duration_string', 'webpage_url']

    def validate_duration(self, value):
        """
        Validate that duration is a positive integer
        """
        if value and value < 0:
            raise serializers.ValidationError("Duration must be a positive integer")
        return value

    def validate_view_count(self, value):
        """
        Validate that view_count is a positive integer
        """
        if value and value < 0:
            raise serializers.ValidationError("View count must be a positive integer")
        return value

class VideoTranscriptSerializer(serializers.ModelSerializer):
    formatted_transcript = serializers.SerializerMethodField()
    # Include segments data for backward compatibility
    transcript_data = serializers.SerializerMethodField()
    # Get language from video metadata
    language = serializers.SerializerMethodField()
    
    class Meta:
        model = VideoTranscript
        fields = [
            'id',
            'url_request',
            'transcript_text',
            'transcript_data',  # Now computed from segments
            'language',  # Now from video_metadata
            'status',
            'created_at',
            'formatted_transcript'
        ]
        read_only_fields = ['id', 'created_at', 'formatted_transcript', 'transcript_data', 'language']
    
    def get_formatted_transcript(self, obj):
        """Return formatted transcript for UI consumption"""
        return obj.get_formatted_transcript()
    
    def get_transcript_data(self, obj):
        """Generate transcript data from segments for backward compatibility"""
        if not hasattr(obj, 'segments') or not obj.segments.exists():
            return []
        
        segments_data = []
        for segment in obj.segments.all():
            segments_data.append({
                'start': segment.start_time,
                'text': segment.text,
                'duration': segment.duration
            })
        return segments_data
    
    def get_language(self, obj):
        """Get language from related VideoMetadata"""
        if obj.video_metadata and obj.video_metadata.language:
            return obj.video_metadata.language
        return 'en'  # Default fallback