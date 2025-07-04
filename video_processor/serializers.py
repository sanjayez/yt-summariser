from rest_framework import serializers
from .models import VideoMetadata, VideoTranscript

class VideoMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoMetadata
        fields = [
            'id',
            'url_request',
            'title',
            'description',
            'duration',
            'channel_name',
            'view_count',
            'status',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']

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
    
    class Meta:
        model = VideoTranscript
        fields = [
            'id',
            'url_request',
            'transcript_text',
            'transcript_data',  # Now computed from segments
            'language',
            'status',
            'created_at',
            'formatted_transcript'
        ]
        read_only_fields = ['id', 'created_at', 'formatted_transcript', 'transcript_data']
    
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

    def validate_language(self, value):
        """
        Validate language code (basic check)
        """
        if value and len(value) > 10:  # Most language codes are 2-5 chars
            raise serializers.ValidationError("Invalid language code format")
        return value.lower()  # Store language codes in lowercase

 