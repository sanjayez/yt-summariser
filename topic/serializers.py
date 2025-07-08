"""
API Serializers for Topic Search Functionality
Handles request/response serialization and validation for YouTube topic search
"""

from rest_framework import serializers
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError as DjangoValidationError


class TopicSearchRequestSerializer(serializers.Serializer):
    """
    Serializer for topic search request validation
    
    Validates user input for YouTube topic search queries
    """
    query = serializers.CharField(
        min_length=1,
        max_length=500,
        trim_whitespace=True,
        help_text="Search query for YouTube videos",
        error_messages={
            'blank': 'Search query cannot be empty',
            'min_length': 'Search query must be at least 1 character long',
            'max_length': 'Search query cannot exceed 500 characters',
            'required': 'Search query is required',
        }
    )
    
    def validate_query(self, value):
        """
        Custom validation for search query
        
        Args:
            value: Query string to validate
            
        Returns:
            Cleaned query string
            
        Raises:
            ValidationError: If query is invalid
        """
        if not value or not value.strip():
            raise serializers.ValidationError('Search query cannot be empty or whitespace only')
        
        # Strip and validate length after trimming
        cleaned_query = value.strip()
        if len(cleaned_query) == 0:
            raise serializers.ValidationError('Search query cannot be empty or whitespace only')
        
        return cleaned_query


class TopicSearchResponseSerializer(serializers.Serializer):
    """
    Serializer for topic search response formatting
    
    Ensures consistent response structure for search results
    """
    search_request_id = serializers.UUIDField(
        help_text="Unique identifier for this search request"
    )
    session_id = serializers.UUIDField(
        help_text="Session identifier for the user"
    )
    original_query = serializers.CharField(
        help_text="User's original search query"
    )
    processed_query = serializers.CharField(
        help_text="LLM-enhanced search query used for YouTube search"
    )
    video_urls = serializers.ListField(
        child=serializers.URLField(),
        help_text="List of YouTube video URLs found"
    )
    total_videos = serializers.IntegerField(
        min_value=0,
        help_text="Total number of videos found"
    )
    processing_time_ms = serializers.FloatField(
        min_value=0,
        help_text="Time taken to process the search request in milliseconds"
    )
    status = serializers.ChoiceField(
        choices=['success', 'processing', 'failed'],
        help_text="Status of the search request"
    )
    
    def validate_video_urls(self, value):
        """
        Custom validation for video URLs
        
        Args:
            value: List of URLs to validate
            
        Returns:
            Validated list of URLs
            
        Raises:
            ValidationError: If any URL is invalid or not a YouTube URL
        """
        url_validator = URLValidator()
        validated_urls = []
        
        for url in value:
            # Validate URL format
            try:
                url_validator(url)
            except DjangoValidationError:
                raise serializers.ValidationError(f"Invalid URL format: {url}")
            
            # Check if it's a YouTube URL
            if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']):
                raise serializers.ValidationError(f"URL must be a YouTube URL: {url}")
            
            validated_urls.append(url)
        
        return validated_urls
    
    def validate(self, attrs):
        """
        Object-level validation for response consistency
        
        Args:
            attrs: Dictionary of all field values
            
        Returns:
            Validated attributes dictionary
            
        Raises:
            ValidationError: If data is inconsistent
        """
        video_urls = attrs.get('video_urls', [])
        total_videos = attrs.get('total_videos', 0)
        
        # Ensure total_videos matches actual video_urls count
        if len(video_urls) != total_videos:
            attrs['total_videos'] = len(video_urls)
        
        return attrs


class ErrorResponseSerializer(serializers.Serializer):
    """
    Serializer for error responses to maintain consistent error format
    """
    error = serializers.CharField(help_text="Error type or category")
    message = serializers.CharField(help_text="Detailed error message")
    details = serializers.DictField(
        required=False,
        help_text="Additional error details"
    ) 