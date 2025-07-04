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
    
    class Meta:
        model = VideoTranscript
        fields = [
            'id',
            'url_request',
            'transcript_text',
            'transcript_data',
            'language',
            'status',
            'created_at',
            'formatted_transcript'
        ]
        read_only_fields = ['id', 'created_at', 'formatted_transcript']
    
    def get_formatted_transcript(self, obj):
        """Return formatted transcript for UI consumption"""
        return obj.get_formatted_transcript()

    def validate_language(self, value):
        """
        Validate language code (basic check)
        """
        if value and len(value) > 10:  # Most language codes are 2-5 chars
            raise serializers.ValidationError("Invalid language code format")
        return value.lower()  # Store language codes in lowercase

    def validate_transcript_data(self, value):
        """
        Validate transcript data structure
        """
        if value is not None:
            if not isinstance(value, list):
                raise serializers.ValidationError("Transcript data must be a list")
            
            # Validate each segment has required fields
            for i, segment in enumerate(value):
                if not isinstance(segment, dict):
                    raise serializers.ValidationError(f"Segment {i} must be a dictionary")
                
                if 'text' not in segment:
                    raise serializers.ValidationError(f"Segment {i} missing 'text' field")
                
                if 'start' not in segment:
                    raise serializers.ValidationError(f"Segment {i} missing 'start' field")
                
                try:
                    float(segment['start'])
                except (ValueError, TypeError):
                    raise serializers.ValidationError(f"Segment {i} 'start' must be a number")
        
        return value 