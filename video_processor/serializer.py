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
    class Meta:
        model = VideoTranscript
        fields = [
            'id',
            'url_request',
            'transcript_text',
            'language',
            'status',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']

    def validate_language(self, value):
        """
        Validate language code (basic check)
        """
        if value and len(value) > 10:  # Most language codes are 2-5 chars
            raise serializers.ValidationError("Invalid language code format")
        return value.lower()  # Store language codes in lowercase
