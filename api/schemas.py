"""
Pydantic schemas for API request/response validation and type safety.
This provides comprehensive type checking and validation for all API endpoints.
"""
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime
from enum import Enum


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
    
    @validator('url')
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
    
    @validator('question')
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
    upload_date: Optional[str] = Field(None, description="Upload date")
    language: Optional[str] = Field(None, description="Video language")
    tags: Optional[List[str]] = Field(None, description="Video tags (limited)")
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    thumbnail: Optional[HttpUrl] = Field(None, description="Thumbnail URL")


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
    video_metadata: VideoMetadataResponse = Field(..., description="Video metadata")
    status: str = Field(..., description="Processing status")
    generated_at: Optional[str] = Field(None, description="Summary generation timestamp")
    summary_length: int = Field(..., description="Length of summary in characters")
    key_points_count: int = Field(..., description="Number of key points")


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