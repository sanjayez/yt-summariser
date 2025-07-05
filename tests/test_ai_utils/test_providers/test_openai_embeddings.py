"""
Unit tests for OpenAI embeddings provider.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from ai_utils.providers.openai_embeddings import OpenAIEmbeddingProvider
from ai_utils.models import EmbeddingRequest, BatchEmbeddingRequest
from ai_utils.config import AIConfig, OpenAIConfig, PineconeConfig

class TestOpenAIEmbeddingProvider:
    """Test OpenAI embeddings provider"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return AIConfig(
            openai=OpenAIConfig(
                api_key="test_key",
                model="text-embedding-3-small",
                timeout=30
            ),
            pinecone=PineconeConfig(api_key="test", environment="test"),  # Required field
            batch_size=100
        )
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock_client = AsyncMock()
        
        # Mock single embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_embedding_response.usage.prompt_tokens = 10
        mock_embedding_response.usage.total_tokens = 10
        mock_embedding_response.id = "test_request_id"
        
        # Mock batch embedding response
        mock_batch_response = MagicMock()
        mock_batch_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_batch_response.usage.prompt_tokens = 20
        mock_batch_response.usage.total_tokens = 20
        mock_batch_response.id = "test_batch_request_id"
        
        async def mock_create(*args, **kwargs):
            input_val = kwargs.get("input", None)
            if isinstance(input_val, list):
                return mock_batch_response
            else:
                return mock_embedding_response
        
        mock_client.embeddings.create = mock_create
        
        return mock_client
    
    @pytest.fixture
    def provider(self, mock_config, mock_openai_client):
        """Create provider with mocked dependencies"""
        with patch('ai_utils.providers.openai_embeddings.AsyncOpenAI', return_value=mock_openai_client):
            return OpenAIEmbeddingProvider(config=mock_config)
    
    @pytest.mark.asyncio
    async def test_embed_text_success(self, provider):
        """Test successful single text embedding"""
        request = EmbeddingRequest(
            text="Hello world",
            model="text-embedding-3-small",
            user="test_user"
        )
        
        response = await provider.embed_text(request)
        
        assert len(response.embedding) == 3
        assert response.model == "text-embedding-3-small"
        assert response.usage["prompt_tokens"] == 10
        assert response.request_id == "test_request_id"
    
    @pytest.mark.asyncio
    async def test_embed_text_empty_after_cleaning(self, provider):
        """Test embedding with text that becomes empty after cleaning"""
        # The model validation will catch this before it reaches the provider
        with pytest.raises(ValueError, match="Text cannot be empty"):
            request = EmbeddingRequest(text="   ")
            await provider.embed_text(request)
    
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, provider):
        """Test successful batch embedding"""
        request = BatchEmbeddingRequest(
            texts=["Hello", "World"],
            model="text-embedding-3-small",
            user="test_user"
        )
        
        response = await provider.embed_batch(request)
        
        print(f"DEBUG: Request texts: {request.texts}")
        print(f"DEBUG: Response embeddings: {response.embeddings}")
        print(f"DEBUG: Response usage: {response.usage}")
        print(f"DEBUG: Response request_id: {response.request_id}")
        
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert len(response.embeddings[1]) == 3
        assert response.model == "text-embedding-3-small"
        assert response.usage["prompt_tokens"] == 20
    
    @pytest.mark.asyncio
    async def test_embed_batch_no_valid_texts(self, provider):
        """Test batch embedding with no valid texts after cleaning"""
        # The model validation will catch this before it reaches the provider
        with pytest.raises(ValueError, match="All texts must be non-empty"):
            request = BatchEmbeddingRequest(
                texts=["   ", "  "],
                model="text-embedding-3-small"
            )
            await provider.embed_batch(request)
    
    def test_get_supported_models(self, provider):
        """Test getting supported models"""
        models = provider.get_supported_models()
        
        assert "text-embedding-3-small" in models
        assert "text-embedding-3-large" in models
        assert "text-embedding-ada-002" in models
        assert len(models) == 3
    
    def test_get_model_dimensions(self, provider):
        """Test getting model dimensions"""
        assert provider.get_model_dimensions("text-embedding-3-small") == 1536
        assert provider.get_model_dimensions("text-embedding-3-large") == 3072
        assert provider.get_model_dimensions("text-embedding-ada-002") == 1536
    
    def test_get_model_dimensions_unsupported(self, provider):
        """Test getting dimensions for unsupported model"""
        with pytest.raises(ValueError, match="Unsupported model"):
            provider.get_model_dimensions("unsupported-model")
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check"""
        # Mock successful models.list call
        provider.client.models.list = AsyncMock()
        
        result = await provider.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check"""
        # Mock failed models.list call
        provider.client.models.list = AsyncMock(side_effect=Exception("API Error"))
        
        result = await provider.health_check()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_embedding_with_retry_success(self, provider):
        """Test successful embedding with retry"""
        result = await provider.get_embedding_with_retry("Hello world", model="text-embedding-3-small")
        
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_get_embedding_with_retry_failure(self, provider):
        """Test embedding with retry that eventually fails"""
        # Mock provider to always fail
        provider.embed_text = AsyncMock(side_effect=Exception("API Error"))
        
        with pytest.raises(Exception, match="API Error"):
            await provider.get_embedding_with_retry("Hello world", model="text-embedding-3-small", max_retries=2)
    
    @pytest.mark.asyncio
    async def test_get_embedding_with_retry_success_after_failure(self, provider):
        """Test embedding with retry that succeeds after initial failure"""
        # Mock provider to fail once then succeed
        call_count = 0
        async def mock_embed_text(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary Error")
            return MagicMock(embedding=[0.1, 0.2, 0.3])
        
        provider.embed_text = mock_embed_text
        
        result = await provider.get_embedding_with_retry("Hello world", model="text-embedding-3-small", max_retries=3)
        
        assert len(result) == 3
        assert call_count == 2  # Should have been called twice (1 failure + 1 success) 