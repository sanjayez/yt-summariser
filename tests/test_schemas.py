"""
Unit tests for Pydantic schemas
Tests validation rules, field validation, and schema behavior
"""
from django.test import TestCase
from pydantic import ValidationError
from api.schemas import UnifiedProcessRequest, UnifiedProcessResponse


class UnifiedProcessRequestSchemaTest(TestCase):
    """Test UnifiedProcessRequest schema validation"""
    
    def test_valid_video_request(self):
        """Test valid video request validation"""
        data = {
            "content": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "type": "video"
        }
        
        schema = UnifiedProcessRequest(**data)
        
        self.assertEqual(schema.content, data["content"])
        self.assertEqual(schema.type, data["type"])
    
    def test_valid_playlist_request(self):
        """Test valid playlist request validation"""
        data = {
            "content": "https://www.youtube.com/playlist?list=PLrAXtmRdnEQy4Qth_4wQi_Q4",
            "type": "playlist"
        }
        
        schema = UnifiedProcessRequest(**data)
        
        self.assertEqual(schema.content, data["content"])
        self.assertEqual(schema.type, data["type"])
    
    def test_valid_topic_request(self):
        """Test valid topic request validation"""
        data = {
            "content": "python machine learning tutorial",
            "type": "topic"
        }
        
        schema = UnifiedProcessRequest(**data)
        
        self.assertEqual(schema.content, data["content"])
        self.assertEqual(schema.type, data["type"])
    
    def test_missing_content_field(self):
        """Test validation error when content field is missing"""
        data = {"type": "video"}
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        self.assertTrue(any(error['loc'] == ('content',) for error in errors))
    
    def test_missing_type_field(self):
        """Test validation error when type field is missing"""
        data = {"content": "https://www.youtube.com/watch?v=test"}
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        self.assertTrue(any(error['loc'] == ('type',) for error in errors))
    
    def test_invalid_type_value(self):
        """Test validation error for invalid type value"""
        data = {
            "content": "https://www.youtube.com/watch?v=test",
            "type": "invalid_type"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        # Should have validation error for type field
        type_errors = [error for error in errors if error['loc'] == ('type',)]
        self.assertTrue(len(type_errors) > 0)
    
    def test_video_url_validation_non_youtube(self):
        """Test validation error for non-YouTube URL with video type"""
        data = {
            "content": "https://www.google.com",
            "type": "video"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        # Should have validation error mentioning YouTube (model-level error)
        model_errors = [error for error in errors if error['loc'] == ()]
        self.assertTrue(len(model_errors) > 0)
        self.assertIn('YouTube', str(model_errors[0]['msg']))
    
    def test_playlist_url_validation_non_youtube(self):
        """Test validation error for non-YouTube URL with playlist type"""
        data = {
            "content": "https://www.example.com/playlist",
            "type": "playlist"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        model_errors = [error for error in errors if error['loc'] == ()]
        self.assertTrue(len(model_errors) > 0)
    
    def test_video_url_validation_non_http(self):
        """Test validation error for non-HTTP URL with video type"""
        data = {
            "content": "ftp://example.com/video",
            "type": "video"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        model_errors = [error for error in errors if error['loc'] == ()]
        self.assertTrue(len(model_errors) > 0)
    
    def test_topic_query_too_short(self):
        """Test validation error for topic query that's too short"""
        data = {
            "content": "ai",  # Only 2 characters
            "type": "topic"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        model_errors = [error for error in errors if error['loc'] == ()]
        self.assertTrue(len(model_errors) > 0)
        self.assertIn('4 characters', str(model_errors[0]['msg']))
    
    def test_topic_query_minimum_length(self):
        """Test topic query at minimum valid length (4 characters)"""
        data = {
            "content": "abcd",  # Exactly 4 characters (minimum required)
            "type": "topic"
        }
        
        # Should not raise validation error
        schema = UnifiedProcessRequest(**data)
        self.assertEqual(schema.content, "abcd")
        self.assertEqual(schema.type, "topic")
    
    def test_content_whitespace_stripping(self):
        """Test that content whitespace is properly stripped"""
        data = {
            "content": "  python machine learning  ",
            "type": "topic"
        }
        
        schema = UnifiedProcessRequest(**data)
        self.assertEqual(schema.content, "python machine learning")
    
    def test_youtube_url_variations(self):
        """Test various valid YouTube URL formats"""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/playlist?list=PLtest123"
        ]
        
        for url in valid_urls:
            data = {"content": url, "type": "video"}
            
            # Should not raise validation error
            schema = UnifiedProcessRequest(**data)
            self.assertEqual(schema.content, url)
    
    def test_empty_content_string(self):
        """Test validation error for empty content string"""
        data = {
            "content": "",
            "type": "video"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        # Should have validation error for content
        content_errors = [error for error in errors if error['loc'] == ('content',)]
        self.assertTrue(len(content_errors) > 0)
    
    def test_whitespace_only_content(self):
        """Test validation error for whitespace-only content"""
        data = {
            "content": "   ",
            "type": "topic"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessRequest(**data)
        
        errors = context.exception.errors()
        content_errors = [error for error in errors if error['loc'] == ('content',)]
        self.assertTrue(len(content_errors) > 0)
    
    def test_schema_json_examples(self):
        """Test that schema examples are valid"""
        # Get examples from schema config
        examples = UnifiedProcessRequest.model_config.get('json_schema_extra', {}).get('examples', [])
        
        self.assertTrue(len(examples) > 0, "Schema should have examples")
        
        # Validate each example
        for example in examples:
            schema = UnifiedProcessRequest(**example)
            self.assertIsNotNone(schema.content)
            self.assertIsNotNone(schema.type)


class UnifiedProcessResponseSchemaTest(TestCase):
    """Test UnifiedProcessResponse schema validation"""
    
    def test_valid_response_creation(self):
        """Test creating valid response schema"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "remaining_limit": 2
        }
        
        schema = UnifiedProcessResponse(**data)
        
        self.assertEqual(schema.session_id, data["session_id"])
        self.assertEqual(schema.status, data["status"])
        self.assertEqual(schema.remaining_limit, data["remaining_limit"])
    
    def test_missing_session_id(self):
        """Test validation error when session_id is missing"""
        data = {
            "status": "processing",
            "remaining_limit": 2
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessResponse(**data)
        
        errors = context.exception.errors()
        self.assertTrue(any(error['loc'] == ('session_id',) for error in errors))
    
    def test_missing_status(self):
        """Test validation error when status is missing"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "remaining_limit": 2
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessResponse(**data)
        
        errors = context.exception.errors()
        self.assertTrue(any(error['loc'] == ('status',) for error in errors))
    
    def test_missing_remaining_limit(self):
        """Test validation error when remaining_limit is missing"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing"
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessResponse(**data)
        
        errors = context.exception.errors()
        self.assertTrue(any(error['loc'] == ('remaining_limit',) for error in errors))
    
    def test_invalid_remaining_limit_type(self):
        """Test validation error for non-integer remaining_limit"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "remaining_limit": "invalid"  # Non-coercible string
        }
        
        with self.assertRaises(ValidationError) as context:
            UnifiedProcessResponse(**data)
        
        errors = context.exception.errors()
        remaining_limit_errors = [error for error in errors if error['loc'] == ('remaining_limit',)]
        self.assertTrue(len(remaining_limit_errors) > 0)
    
    def test_negative_remaining_limit(self):
        """Test that negative remaining_limit is allowed (edge case)"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "rate_limited",
            "remaining_limit": -1
        }
        
        # Should not raise validation error (business logic handles this)
        schema = UnifiedProcessResponse(**data)
        self.assertEqual(schema.remaining_limit, -1)
    
    def test_zero_remaining_limit(self):
        """Test zero remaining_limit"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "rate_limited",
            "remaining_limit": 0
        }
        
        schema = UnifiedProcessResponse(**data)
        self.assertEqual(schema.remaining_limit, 0)
    
    def test_various_status_values(self):
        """Test various status values"""
        valid_statuses = ["processing", "rate_limited", "error", "completed"]
        
        for status in valid_statuses:
            data = {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": status,
                "remaining_limit": 1
            }
            
            schema = UnifiedProcessResponse(**data)
            self.assertEqual(schema.status, status)
    
    def test_empty_status_string(self):
        """Test empty status string"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "",
            "remaining_limit": 2
        }
        
        # Empty string should be allowed (business logic validates specific values)
        schema = UnifiedProcessResponse(**data)
        self.assertEqual(schema.status, "")
    
    def test_dict_conversion(self):
        """Test converting schema to dictionary"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "remaining_limit": 2
        }
        
        schema = UnifiedProcessResponse(**data)
        result_dict = schema.model_dump()
        
        self.assertEqual(result_dict, data)
        self.assertIsInstance(result_dict, dict)
    
    def test_json_serialization(self):
        """Test JSON serialization of schema"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "remaining_limit": 2
        }
        
        schema = UnifiedProcessResponse(**data)
        json_str = schema.model_dump_json()
        
        import json
        parsed = json.loads(json_str)
        self.assertEqual(parsed, data)
    
    def test_schema_json_examples(self):
        """Test that response schema examples are valid"""
        # Get examples from schema config
        examples = UnifiedProcessResponse.model_config.get('json_schema_extra', {}).get('examples', [])
        
        self.assertTrue(len(examples) > 0, "Response schema should have examples")
        
        # Validate each example
        for example in examples:
            schema = UnifiedProcessResponse(**example)
            self.assertIsNotNone(schema.session_id)
            self.assertIsNotNone(schema.status)
            self.assertIsNotNone(schema.remaining_limit)
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored during validation"""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "remaining_limit": 2,
            "extra_field": "should_be_ignored"
        }
        
        schema = UnifiedProcessResponse(**data)
        
        # Extra field should not be included in the schema
        self.assertFalse(hasattr(schema, 'extra_field'))
        
        # Dict conversion should not include extra field
        result_dict = schema.model_dump()
        self.assertNotIn('extra_field', result_dict)
