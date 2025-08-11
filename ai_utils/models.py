"""
Pydantic models for type-safe data structures in AI utilities.
These models handle data validation, serialization, and type safety.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator

class ProcessingStatus(str, Enum):
    """Status of processing operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EmbeddingRequest(BaseModel):
    """Request model for text embedding"""
    text: str = Field(..., description="Text to embed", min_length=1)
    model: str = Field(default="text-embedding-3-small", description="Embedding model to use")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class EmbeddingResponse(BaseModel):
    """Response model for text embedding"""
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for embedding")
    usage: Dict[str, Any] = Field(..., description="Usage statistics")
    request_id: Optional[str] = Field(None, description="Request identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding"""
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)
    model: str = Field(default="text-embedding-3-small", description="Embedding model to use")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v or not all(text.strip() for text in v):
            raise ValueError("All texts must be non-empty")
        return [text.strip() for text in v]

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding"""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model used for embedding")
    usage: Dict[str, Any] = Field(..., description="Usage statistics")
    request_id: Optional[str] = Field(None, description="Request identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class VectorDocument(BaseModel):
    """Model for vector store documents"""
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector (optional with native vectorization)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Document creation timestamp")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()

class VectorQuery(BaseModel):
    """Query model for vector similarity search"""
    query: str = Field(..., description="Query text")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed query embedding")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    include_embeddings: bool = Field(default=False, description="Include embeddings in results")

class VectorSearchResult(BaseModel):
    """Result model for vector similarity search"""
    id: str = Field(..., description="Document identifier")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding (if requested)")

class VectorSearchResponse(BaseModel):
    """Response model for vector similarity search"""
    results: List[VectorSearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")

class RAGQuery(BaseModel):
    """Query model for RAG operations"""
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar documents to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    response_mode: str = Field(default="compact", description="Response generation mode")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum response tokens")

class RAGResponse(BaseModel):
    """Response model for RAG operations"""
    answer: str = Field(..., description="Generated answer")
    sources: List[VectorSearchResult] = Field(..., description="Source documents used")
    query: str = Field(..., description="Original query")
    response_time_ms: float = Field(..., description="Response generation time in milliseconds")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage statistics")

class ProcessingJob(BaseModel):
    """Model for tracking processing jobs"""
    job_id: str = Field(..., description="Unique job identifier")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
    operation: str = Field(..., description="Type of operation (embedding, indexing, etc.)")
    total_items: int = Field(..., description="Total items to process")
    processed_items: int = Field(default=0, description="Number of items processed")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job metadata")

class IndexConfig(BaseModel):
    """Configuration model for vector index"""
    name: str = Field(..., description="Index name")
    dimension: int = Field(..., description="Vector dimension")
    metric: str = Field(default="cosine", description="Distance metric")
    metadata_config: Optional[Dict[str, Any]] = Field(None, description="Metadata configuration")
    replicas: int = Field(default=1, ge=1, description="Number of replicas")

class IndexStats(BaseModel):
    """Statistics model for vector index"""
    total_vector_count: int = Field(..., description="Total number of vectors")
    dimension: int = Field(..., description="Vector dimension")
    index_fullness: float = Field(..., description="Index fullness percentage")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

# LLM Models

class ChatRole(str, Enum):
    """Chat message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class ChatMessage(BaseModel):
    """Chat message model"""
    role: ChatRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Message author name")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call data")

class ChatRequest(BaseModel):
    """Chat completion request model"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages", min_length=1)
    model: str = Field(default="gemini-2.5-flash", description="Chat model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response")
    user: Optional[str] = Field(None, description="User identifier")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        return v

class ChatChoice(BaseModel):
    """Chat completion choice model"""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Response message")
    finish_reason: Optional[str] = Field(None, description="Finish reason")

class ChatUsage(BaseModel):
    """Chat completion usage model"""
    prompt_tokens: int = Field(..., description="Prompt tokens used")
    completion_tokens: int = Field(..., description="Completion tokens used")
    total_tokens: int = Field(..., description="Total tokens used")

class ChatResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(..., description="Response ID")
    object: str = Field(..., description="Response object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatChoice] = Field(..., description="Response choices")
    usage: ChatUsage = Field(..., description="Token usage")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")

class TextGenerationRequest(BaseModel):
    """Simple text generation request model"""
    prompt: str = Field(..., description="Input prompt", min_length=1)
    model: str = Field(default="gpt-3.5-turbo", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

class TextGenerationResponse(BaseModel):
    """Simple text generation response model"""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    usage: ChatUsage = Field(..., description="Token usage")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp") 