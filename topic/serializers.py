"""
API Serializers for Topic Search Functionality
Handles request/response serialization and validation for YouTube topic search
"""

from rest_framework import serializers


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
    Serializer for initial topic search response (processing state)
    
    Used when returning immediate response while processing happens in background
    """
    session_id = serializers.UUIDField(
        help_text="Session identifier for the user"
    )
    original_query = serializers.CharField(
        help_text="User's original search query"
    )
    status = serializers.ChoiceField(
        choices=['processing', 'success', 'failed'],
        help_text="Status of the search request"
    )


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