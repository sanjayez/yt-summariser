"""
Pydantic schemas for API request/response validation and type safety.
This provides comprehensive type checking and validation for all API endpoints.
"""
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime, date
from enum import Enum
from decimal import Decimal


class VideoProcessingStatus(str, Enum):
    """Valid video processing status values"""
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class SearchMethod(str, Enum):
    """Valid search method types"""
    VECTOR_SEARCH = "vector_search"
    TRANSCRIPT_FALLBACK = "transcript_fallback"


class SourceType(str, Enum):
    """Valid source types for search results"""
    SEGMENT = "segment"
    TRANSCRIPT = "transcript"


# Request Schemas
class VideoProcessRequest(BaseModel):
    """Request schema for video processing endpoint"""
    url: HttpUrl = Field(..., description="Valid YouTube video URL")
    
    @field_validator('url')
    @classmethod
    def validate_youtube_url(cls, v):
        """Validate that URL is from YouTube"""
        url_str = str(v)
        if 'youtube.com' not in url_str and 'youtu.be' not in url_str:
            raise ValueError('Must be a valid YouTube URL (youtube.com or youtu.be)')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
        }


class VideoQuestionRequest(BaseModel):
    """Request schema for video question endpoint"""
    question: str = Field(
        ..., 
        min_length=3, 
        max_length=500,
        description="Question about the video content"
    )
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate question format"""
        if not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main topics discussed in this video?"
            }
        }


# Response Schemas
class SearchSource(BaseModel):
    """Schema for search result sources"""
    type: SourceType = Field(..., description="Type of source (segment or transcript)")
    timestamp: str = Field(..., description="Timestamp in MM:SS format or 'Unknown'")
    text: str = Field(..., description="Content text from the source")
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for this source"
    )
    
    @field_validator('youtube_url', mode='before')
    @classmethod
    def serialize_youtube_url(cls, v):
        """Convert URL objects to strings for proper HttpUrl validation"""
        if v is None:
            return None
        return str(v)
    
    @field_validator('confidence', mode='before')
    @classmethod
    def serialize_confidence(cls, v):
        """Ensure confidence is a proper float between 0.0 and 1.0"""
        if v is None:
            return 0.0
        try:
            conf = float(v)
            return max(0.0, min(1.0, conf))  # Clamp between 0.0 and 1.0
        except (ValueError, TypeError):
            return 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "segment",
                "timestamp": "02:30",
                "text": "This segment discusses the main concepts of machine learning",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "confidence": 0.85
            }
        }


class VideoMetadataResponse(BaseModel):
    """Schema for video metadata in responses"""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description (truncated)")
    duration: Optional[int] = Field(None, description="Duration in seconds")
    duration_string: Optional[str] = Field(None, description="Duration in MM:SS format")
    channel_name: Optional[str] = Field(None, description="Channel name")
    view_count: Optional[int] = Field(None, description="View count")
    like_count: Optional[int] = Field(None, description="Like count")
    upload_date: Optional[str] = Field(None, description="Upload date in YYYY-MM-DD format")
    language: Optional[str] = Field(None, description="Video language")
    tags: Optional[List[str]] = Field(None, description="Video tags (limited)")
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    thumbnail: Optional[HttpUrl] = Field(None, description="Thumbnail URL")
    
    @field_validator('upload_date', mode='before')
    @classmethod
    def serialize_upload_date(cls, v):
        """Convert date/datetime objects to string format"""
        if isinstance(v, date):
            return v.isoformat()  # Returns YYYY-MM-DD format
        elif isinstance(v, datetime):
            return v.date().isoformat()  # Extract date part and format
        return v
    
    @field_validator('view_count', 'like_count', 'duration', mode='before')
    @classmethod
    def serialize_numeric_fields(cls, v):
        """Convert numeric types to proper integers"""
        if v is None:
            return None
        if isinstance(v, (int, float, Decimal)):
            return int(v) if v >= 0 else 0  # Ensure non-negative integers
        try:
            return int(float(str(v))) if str(v).strip() else None
        except (ValueError, TypeError):
            return None
    
    @field_validator('youtube_url', 'thumbnail', mode='before')
    @classmethod
    def serialize_url_fields(cls, v):
        """Convert URL objects to strings for proper HttpUrl validation"""
        if v is None:
            return None
        return str(v)


class ProcessingStages(BaseModel):
    """Schema for processing stage status"""
    metadata_extracted: bool = Field(..., description="Metadata extraction completed")
    transcript_extracted: bool = Field(..., description="Transcript extraction completed")
    summary_generated: bool = Field(..., description="Summary generation completed")
    content_embedded: bool = Field(..., description="Content embedding completed")
    processing_complete: bool = Field(..., description="Overall processing completed")


class TimingInfo(BaseModel):
    """Schema for operation timing information"""
    total_time_ms: Optional[float] = Field(None, description="Total time in milliseconds")
    stage_timings: Optional[Dict[str, float]] = Field(None, description="Individual stage timings")
    measured_stages: Optional[int] = Field(None, description="Number of measured stages")
    note: Optional[str] = Field(None, description="Additional timing notes")


class VideoProcessResponse(BaseModel):
    """Response schema for video processing initiation"""
    request_id: str = Field(..., description="Unique request identifier")
    url: HttpUrl = Field(..., description="Processed video URL")
    status: VideoProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "status": "processing",
                "message": "Video processing started. Use the request_id to check status and retrieve results."
            }
        }


class VideoSummaryResponse(BaseModel):
    """Response schema for video summary endpoint"""
    summary: str = Field(..., description="AI-generated video summary")
    key_points: List[str] = Field(..., description="Key points from the video")
    chapters: Optional[List[Dict[str, Any]]] = Field(None, description="Chapter-wise structured content")
    video_metadata: VideoMetadataResponse = Field(..., description="Video metadata")
    status: str = Field(..., description="Processing status")
    generated_at: Optional[str] = Field(None, description="Summary generation timestamp in ISO format")
    summary_length: int = Field(..., description="Length of summary in characters")
    key_points_count: int = Field(..., description="Number of key points")
    
    @field_validator('summary', mode='before')
    @classmethod
    def validate_summary(cls, v):
        """Validate and clean summary text"""
        if v is None:
            return ""
        summary_text = str(v).strip()
        # Ensure summary is not too long (reasonable limit)
        if len(summary_text) > 10000:
            return summary_text[:9997] + "..."
        return summary_text
    
    @field_validator('key_points', mode='before')
    @classmethod 
    def validate_key_points(cls, v):
        """Ensure key_points is always a list of strings
        
        TODO: Simplify this validation once LLM returns consistent JSON format
        """
        if v is None:
            return []
        if not isinstance(v, list):
            # Try to convert to list
            try:
                if isinstance(v, str):
                    import json
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        v = parsed
                    else:
                        return [str(v)]
                else:
                    return [str(v)]
            except:
                return [str(v)]
        
        # Clean and validate each item
        cleaned_points = []
        for item in v:
            if item is not None:
                str_item = str(item).strip()
                if str_item and str_item != "None":
                    # Limit length of individual key points
                    if len(str_item) > 500:
                        str_item = str_item[:497] + "..."
                    cleaned_points.append(str_item)
        
        return cleaned_points
    
    @field_validator('chapters', mode='before')
    @classmethod
    def validate_chapters(cls, v):
        """Validate chapters structure
        
        TODO: Remove complex validation after fixing chapter response format
        """
        if v is None or not v:
            return None
        
        if not isinstance(v, list):
            return None
        
        validated_chapters = []
        for i, chapter in enumerate(v):
            if not isinstance(chapter, dict):
                continue
                
            # Ensure required fields exist with reasonable defaults
            validated_chapter = {
                "chapter": chapter.get("chapter", i + 1),
                "title": str(chapter.get("title", f"Chapter {i + 1}")).strip(),
                "summary": str(chapter.get("summary", "")).strip()
            }
            
            # Validate types and lengths
            if not validated_chapter["title"]:
                validated_chapter["title"] = f"Chapter {i + 1}"
            elif len(validated_chapter["title"]) > 300:
                validated_chapter["title"] = validated_chapter["title"][:297] + "..."
                
            if not validated_chapter["summary"]:
                validated_chapter["summary"] = "No summary available"
            elif len(validated_chapter["summary"]) > 2000:
                validated_chapter["summary"] = validated_chapter["summary"][:1997] + "..."
                
            # Ensure chapter number is valid
            try:
                validated_chapter["chapter"] = int(validated_chapter["chapter"])
            except (ValueError, TypeError):
                validated_chapter["chapter"] = i + 1
                
            validated_chapters.append(validated_chapter)
        
        return validated_chapters if validated_chapters else None
    
    @field_validator('generated_at', mode='before')
    @classmethod
    def serialize_generated_at(cls, v):
        """Convert datetime objects to ISO string format"""
        if isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, date):
            return datetime.combine(v, datetime.min.time()).isoformat()
        return v
    
    @field_validator('summary_length', 'key_points_count', mode='before')
    @classmethod
    def serialize_count_fields(cls, v):
        """Ensure count fields are proper integers"""
        if v is None:
            return 0
        if isinstance(v, (int, float, Decimal)):
            return int(v) if v >= 0 else 0
        try:
            return int(float(str(v))) if str(v).strip() else 0
        except (ValueError, TypeError):
            return 0


class VideoSearchResponse(BaseModel):
    """Response schema for video question endpoint"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="AI-generated answer")
    sources: List[SearchSource] = Field(..., description="Supporting sources")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in the answer"
    )
    search_method: SearchMethod = Field(..., description="Search method used")
    results_count: int = Field(..., description="Number of results found")
    video_metadata: Optional[Dict[str, Any]] = Field(None, description="Video metadata")
    timing: Optional[TimingInfo] = Field(None, description="Performance timing information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    {
                        "type": "segment",
                        "timestamp": "02:30",
                        "text": "Machine learning is discussed here",
                        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "confidence": 0.85
                    }
                ],
                "confidence": 0.85,
                "search_method": "vector_search",
                "results_count": 1
            }
        }


class VideoStatusResponse(BaseModel):
    """Response schema for video status streaming"""
    overall_status: VideoProcessingStatus = Field(..., description="Overall processing status")
    timestamp: float = Field(..., description="Unix timestamp")
    metadata_status: Optional[str] = Field(None, description="Metadata extraction status")
    transcript_status: Optional[str] = Field(None, description="Transcript extraction status")
    summary_status: Optional[str] = Field(None, description="Summary generation status")
    embedding_status: Optional[str] = Field(None, description="Embedding generation status")
    stages: ProcessingStages = Field(..., description="Processing stage completion status")
    progress_percentage: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall progress percentage"
    )


class APIErrorResponse(BaseModel):
    """Standard error response schema"""
    error: str = Field(..., description="Error type or category")
    status: str = Field(..., description="Error status code")
    message: str = Field(..., description="Detailed error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Video not found",
                "status": "not_found", 
                "message": "No video processing request found with the provided ID",
                "details": {"request_id": "123e4567-e89b-12d3-a456-426614174000"}
            }
        }


# Utility function to validate UUID strings
def validate_uuid_string(uuid_string: str) -> bool:
    """Validate that a string is a valid UUID format"""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


# Common validators
class VideoURLValidator:
    """Utility class for YouTube URL validation"""
    
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Check if URL is a valid YouTube URL"""
        return 'youtube.com' in url or 'youtu.be' in url
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None