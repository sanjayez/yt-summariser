"""
Enhanced API Serializers with proper validation and security.
Replaces the insecure 'fields = __all__' pattern with explicit field control.
"""
from rest_framework import serializers
from .models import URLRequestTable
from .schemas import VideoURLValidator


class URLRequestTableSerializer(serializers.ModelSerializer):
    """
    Enhanced URLRequestTable serializer with proper field control and validation.
    
    This replaces the insecure 'fields = __all__' pattern with explicit field
    definitions and custom validation logic.
    """
    
    # Explicit field definitions (security best practice)
    url = serializers.URLField(max_length=500, required=True)
    ip_address = serializers.IPAddressField(required=True)
    status = serializers.ChoiceField(
        choices=URLRequestTable.STATUS_CHOICES,
        default='processing',
        read_only=True
    )
    request_id = serializers.UUIDField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    
    # Optional fields for task tracking
    celery_task_id = serializers.CharField(max_length=255, read_only=True, allow_null=True)
    chain_task_id = serializers.CharField(max_length=255, read_only=True, allow_null=True)
    failure_reason = serializers.ChoiceField(
        choices=URLRequestTable.FAILURE_REASON_CHOICES,
        read_only=True,
        allow_null=True
    )
    
    class Meta:
        model = URLRequestTable
        fields = [
            'request_id',
            'url', 
            'ip_address',
            'status',
            'celery_task_id',
            'chain_task_id', 
            'failure_reason',
            'created_at'
        ]
        read_only_fields = [
            'request_id',
            'status', 
            'celery_task_id',
            'chain_task_id',
            'failure_reason',
            'created_at'
        ]
    
    def validate_url(self, value):
        """
        Validate that the URL is a proper YouTube URL.
        
        Args:
            value: URL string to validate
            
        Returns:
            Validated URL string
            
        Raises:
            serializers.ValidationError: If URL is not a valid YouTube URL
        """
        if not VideoURLValidator.is_valid_youtube_url(value):
            raise serializers.ValidationError(
                "Must be a valid YouTube URL (youtube.com or youtu.be)"
            )
        return value
    
    def validate_ip_address(self, value):
        """
        Validate IP address format and apply any necessary restrictions.
        
        Args:
            value: IP address string to validate
            
        Returns:
            Validated IP address string
        """
        # Additional IP validation logic could be added here
        # (e.g., blocking certain IP ranges, etc.)
        return value
    
    def create(self, validated_data):
        """
        Create a new URLRequestTable instance with proper defaults.
        
        Args:
            validated_data: Validated data dictionary
            
        Returns:
            Created URLRequestTable instance
        """
        # Ensure status is set to processing for new requests
        validated_data['status'] = 'processing'
        return super().create(validated_data)
    
    def to_representation(self, instance):
        """
        Customize the serialized representation.
        
        Args:
            instance: URLRequestTable instance
            
        Returns:
            Dictionary representation of the instance
        """
        data = super().to_representation(instance)
        
        # Add computed fields if needed
        if instance.created_at:
            data['created_at_formatted'] = instance.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Add processing time if available
        if instance.status in ['success', 'failed'] and instance.created_at:
            from django.utils import timezone
            processing_time = timezone.now() - instance.created_at
            data['processing_time_seconds'] = processing_time.total_seconds()
        
        return data


class URLRequestStatusSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for status-only responses.
    Used for status streaming and quick status checks.
    """
    
    class Meta:
        model = URLRequestTable
        fields = [
            'request_id',
            'status',
            'failure_reason',
            'created_at'
        ]
        read_only_fields = ['request_id', 'status', 'failure_reason', 'created_at']