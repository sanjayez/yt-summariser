"""
Unit tests for embedding service.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from ai_utils.services.embedding_service import EmbeddingService
from ai_utils.models import EmbeddingResponse, BatchEmbeddingResponse, ProcessingStatus
from ai_utils.config import AIConfig, OpenAIConfig, PineconeConfig

class TestEmbeddingService:
    """Test embedding service"""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock embedding provider"""
        from ai_utils.models import EmbeddingResponse, BatchEmbeddingResponse
        provider = AsyncMock()

        # Return real EmbeddingResponse for embed_text
        provider.embed_text.return_value = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 10, "total_tokens": 10},
            request_id="test_request_id"
        )

        # Return real BatchEmbeddingResponse for embed_batch
        provider.embed_batch.return_value = BatchEmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 20, "total_tokens": 20},
            request_id="test_batch_request_id"
        )

        # get_supported_models should be a regular function, not async
        provider.get_supported_models = lambda: ["text-embedding-3-small", "text-embedding-3-large"]
        provider.health_check.return_value = True

        return provider
    
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
    def service(self, mock_provider, mock_config):
        """Create service with mocked dependencies"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("ai_utils.services.embedding_service.get_config", lambda: mock_config)
            return EmbeddingService(provider=mock_provider)
    
    @pytest.mark.asyncio
    async def test_embed_text_success(self, service):
        """Test successful single text embedding"""
        response = await service.embed_text("Hello world", user="test_user")
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embedding) == 3
        assert response.model == "text-embedding-3-small"
        assert response.usage["prompt_tokens"] == 10
    
    @pytest.mark.asyncio
    async def test_embed_text_with_custom_model(self, service):
        """Test embedding with custom model"""
        response = await service.embed_text("Hello world", model="text-embedding-3-large")
        
        # Verify the provider was called with correct model
        service.provider.embed_text.assert_called_once()
        call_args = service.provider.embed_text.call_args[0][0]
        assert call_args.model == "text-embedding-3-large"
    
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, service):
        """Test successful batch embedding"""
        texts = ["Hello", "World"]
        response = await service.embed_batch(texts, user="test_user")
        
        assert isinstance(response, BatchEmbeddingResponse)
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert len(response.embeddings[1]) == 3
    
    @pytest.mark.asyncio
    async def test_embed_batch_with_job_tracking(self, service):
        """Test batch embedding with job tracking"""
        texts = ["Hello", "World"]
        job_id = "test_job_123"
        
        response = await service.embed_batch(texts, job_id=job_id)
        
        # Check job tracking
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.operation == "batch_embedding"
        assert job.total_items == 2
        assert job.status == ProcessingStatus.COMPLETED
        assert job.processed_items == 2
    
    @pytest.mark.asyncio
    async def test_embed_batch_failure_tracking(self, service):
        """Test batch embedding failure tracking"""
        # Mock provider to fail
        service.provider.embed_batch.side_effect = Exception("API Error")
        
        texts = ["Hello", "World"]
        job_id = "test_job_fail"
        
        with pytest.raises(Exception, match="API Error"):
            await service.embed_batch(texts, job_id=job_id)
        
        # Check job tracking shows failure
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.status == ProcessingStatus.FAILED
        assert job.error_message == "API Error"
    
    @pytest.mark.asyncio
    async def test_embed_texts_with_batching(self, service):
        """Test embedding texts with batching"""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        
        embeddings = await service.embed_texts_with_batching(texts, batch_size=2)
        
        assert len(embeddings) == 4
        assert all(len(emb) == 3 for emb in embeddings)
        
        # Verify provider was called for each batch
        assert service.provider.embed_batch.call_count == 2  # 2 batches of 2 texts each
    
    @pytest.mark.asyncio
    async def test_embed_texts_with_batching_empty_list(self, service):
        """Test batching with empty text list"""
        embeddings = await service.embed_texts_with_batching([])
        
        assert embeddings == []
        # Provider should not be called
        service.provider.embed_batch.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_embed_texts_with_batching_partial_failure(self, service):
        """Test batching with partial batch failure"""
        from ai_utils.models import BatchEmbeddingResponse
        # Mock provider to fail on second batch
        call_count = 0
        async def mock_embed_batch(request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Batch 2 failed")
            return BatchEmbeddingResponse(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                model="text-embedding-3-small",
                usage={"prompt_tokens": 10, "total_tokens": 10},
                request_id="test_batch_request_id"
            )
        
        service.provider.embed_batch.side_effect = mock_embed_batch
        
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        
        # Should continue processing despite batch failure
        embeddings = await service.embed_texts_with_batching(texts, batch_size=2)
        
        # Should have embeddings from successful batches only
        assert len(embeddings) == 2  # Only first batch succeeded
    
    @pytest.mark.asyncio
    async def test_embed_documents(self, service):
        """Test embedding documents with metadata preservation"""
        documents = [
            {"id": "doc1", "text": "Hello world", "metadata": "test1"},
            {"id": "doc2", "text": "Goodbye world", "metadata": "test2"}
        ]
        
        result = await service.embed_documents(documents)
        
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["text"] == "Hello world"
        assert result[0]["metadata"] == "test1"
        assert "embedding" in result[0]
        assert "embedding_model" in result[0]
        
        assert result[1]["id"] == "doc2"
        assert result[1]["text"] == "Goodbye world"
        assert result[1]["metadata"] == "test2"
        assert "embedding" in result[1]
    
    @pytest.mark.asyncio
    async def test_embed_documents_custom_fields(self, service):
        """Test embedding documents with custom field names"""
        documents = [
            {"doc_id": "doc1", "content": "Hello world"},
            {"doc_id": "doc2", "content": "Goodbye world"}
        ]
        
        result = await service.embed_documents(
            documents, 
            text_field="content", 
            id_field="doc_id"
        )
        
        assert len(result) == 2
        assert result[0]["doc_id"] == "doc1"
        assert result[0]["content"] == "Hello world"
        assert "embedding" in result[0]
    
    def test_get_job_status(self, service):
        """Test getting job status"""
        # Create a test job
        job_id = "test_job"
        service._active_jobs[job_id] = MagicMock(job_id=job_id)
        
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        
        # Test non-existent job
        assert service.get_job_status("non_existent") is None
    
    def test_get_active_jobs(self, service):
        """Test getting active jobs"""
        # Add some test jobs
        service._active_jobs["job1"] = MagicMock(job_id="job1")
        service._active_jobs["job2"] = MagicMock(job_id="job2")
        
        jobs = service.get_active_jobs()
        assert len(jobs) == 2
        assert any(job.job_id == "job1" for job in jobs)
        assert any(job.job_id == "job2" for job in jobs)
    
    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check"""
        result = await service.health_check()
        assert result is True
        
        # Test with unhealthy provider
        service.provider.health_check.return_value = False
        result = await service.health_check()
        assert result is False
    
    def test_get_performance_stats(self, service):
        """Test getting performance statistics"""
        # Add some test jobs
        job1 = MagicMock()
        job1.status = ProcessingStatus.COMPLETED
        job1.processed_items = 10
        
        job2 = MagicMock()
        job2.status = ProcessingStatus.FAILED
        job2.processed_items = 0
        
        service._active_jobs = {"job1": job1, "job2": job2}
        
        stats = service.get_performance_stats()
        
        assert stats["active_jobs"] == 2
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["total_items_processed"] == 10
        assert "text-embedding-3-small" in stats["supported_models"]
        assert "text-embedding-3-large" in stats["supported_models"] 