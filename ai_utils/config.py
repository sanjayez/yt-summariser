"""
Configuration management for AI utilities.
Supports environment variables and config files with fallback defaults.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIConfig(BaseModel):
    """OpenAI configuration settings"""
    api_key: str = Field(..., description="OpenAI API key")
    
    # Embedding model settings
    embedding_model: str = Field(default="text-embedding-3-large", description="Default embedding model")
    
    # Chat/LLM model settings
    chat_model: str = Field(default="gpt-3.5-turbo", description="Default chat completion model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Default max tokens for generation")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Default top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Default frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Default presence penalty")
    
    # API settings
    base_url: Optional[str] = Field(None, description="OpenAI API base URL (for custom endpoints)")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    user: Optional[str] = Field(None, description="User identifier for API requests")

class GeminiConfig(BaseModel):
    """Google Gemini configuration settings"""
    api_key: str = Field(..., description="Google Gemini API key")
    
    # Model settings
    model: str = Field(default="gemini-2.5-flash", description="Default Gemini model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Default generation temperature")
    max_tokens: Optional[int] = Field(default=4000, ge=1, description="Default max tokens for generation")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Default top-p sampling")
    top_k: Optional[int] = Field(default=40, ge=1, description="Default top-k sampling")
    
    # API settings
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

class PineconeConfig(BaseModel):
    """Pinecone configuration settings"""
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment (e.g., 'aws-starter')")
    index_name: str = Field(default="youtube-transcripts", description="Default index name")
    dimension: int = Field(default=3072, description="Vector dimension (for text-embedding-3-large)")
    metric: str = Field(default="cosine", description="Distance metric for similarity")
    cloud: str = Field(default="aws", description="Cloud provider (aws or gcp)")
    region: str = Field(default="us-east-1", description="Cloud region")

class LlamaIndexConfig(BaseModel):
    """LlamaIndex configuration settings"""
    chunk_size: int = Field(default=512, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    similarity_top_k: int = Field(default=5, description="Number of similar documents to retrieve")
    response_mode: str = Field(default="compact", description="Response generation mode")

class AIConfig(BaseModel):
    """Main AI configuration container"""
    openai: OpenAIConfig
    gemini: GeminiConfig
    pinecone: PineconeConfig
    llamaindex: LlamaIndexConfig = Field(default_factory=LlamaIndexConfig)
    
    # Processing settings
    batch_size: int = Field(default=100, description="Batch size for embedding operations")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")
    
    # Feature flags
    enable_async: bool = Field(default=True, description="Enable async operations")
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    
    @classmethod
    def from_env(cls) -> "AIConfig":
        """Create configuration from environment variables"""
        return cls(
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
                chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS")) if os.getenv("OPENAI_MAX_TOKENS") else None,
                top_p=float(os.getenv("OPENAI_TOP_P", "1.0")),
                frequency_penalty=float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.0")),
                presence_penalty=float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.0")),
                base_url=os.getenv("OPENAI_BASE_URL"),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                user=os.getenv("OPENAI_USER")
            ),
            gemini=GeminiConfig(
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4000")),
                top_p=float(os.getenv("GEMINI_TOP_P", "1.0")),
                top_k=int(os.getenv("GEMINI_TOP_K")) if os.getenv("GEMINI_TOP_K") else 40,
                timeout=int(os.getenv("GEMINI_TIMEOUT", "30")),
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3"))
            ),
            pinecone=PineconeConfig(
                api_key=os.getenv("PINECONE_API_KEY", ""),
                environment=os.getenv("PINECONE_ENVIRONMENT", "aws-starter"),
                index_name=os.getenv("PINECONE_INDEX_NAME", "youtube-transcripts"),
                dimension=int(os.getenv("PINECONE_DIMENSION", "3072")),
                metric=os.getenv("PINECONE_METRIC", "cosine"),
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            ),
            llamaindex=LlamaIndexConfig(
                chunk_size=int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("LLAMAINDEX_CHUNK_OVERLAP", "50")),
                similarity_top_k=int(os.getenv("LLAMAINDEX_TOP_K", "5")),
                response_mode=os.getenv("LLAMAINDEX_RESPONSE_MODE", "compact")
            ),
            batch_size=int(os.getenv("AI_BATCH_SIZE", "100")),
            max_concurrent_requests=int(os.getenv("AI_MAX_CONCURRENT", "10")),
            enable_async=os.getenv("AI_ENABLE_ASYNC", "true").lower() == "true",
            enable_logging=os.getenv("AI_ENABLE_LOGGING", "true").lower() == "true"
        )
    
    def validate(self) -> None:
        """Validate configuration and raise errors for missing required fields"""
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.gemini.api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        if not self.pinecone.api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.pinecone.environment:
            raise ValueError("PINECONE_ENVIRONMENT is required")

# Global configuration instance
config = AIConfig.from_env()

def get_config() -> AIConfig:
    """Get the global AI configuration"""
    return config

def update_config(**kwargs) -> None:
    """Update global configuration with new values"""
    global config
    config = config.model_copy(update=kwargs) 