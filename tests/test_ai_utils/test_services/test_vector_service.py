"""
Unit tests for vector service.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_utils.config import AIConfig, OpenAIConfig, PineconeConfig
from ai_utils.models import (
    IndexConfig,
    IndexStats,
    ProcessingStatus,
    VectorDocument,
    VectorQuery,
    VectorSearchResponse,
)
from ai_utils.services.vector_service import VectorService


class TestVectorService:
    """Test vector service"""

    @pytest.fixture
    def mock_provider(self):
        """Mock vector store provider"""
        from ai_utils.models import (
            IndexConfig,
            IndexStats,
            VectorSearchResponse,
            VectorSearchResult,
        )

        provider = AsyncMock()

        # Mock upsert response
        provider.upsert_documents.return_value = {
            "upserted_count": 2,
            "index_name": "test-index",
            "batch_count": 1,
        }

        # Mock search response
        mock_result1 = VectorSearchResult(
            id="doc1",
            text="Sample text 1",
            score=0.95,
            metadata={"category": "test"},
            embedding=[0.1, 0.2, 0.3],
        )
        mock_result2 = VectorSearchResult(
            id="doc2",
            text="Sample text 2",
            score=0.85,
            metadata={"category": "test"},
            embedding=[0.4, 0.5, 0.6],
        )

        provider.search_similar.return_value = VectorSearchResponse(
            results=[mock_result1, mock_result2],
            query="test query",
            total_results=2,
            search_time_ms=50.0,
        )

        # Mock delete response
        provider.delete_documents.return_value = {
            "deleted_count": 3,
            "index_name": "test-index",
        }

        # Mock get document
        provider.get_document.return_value = VectorDocument(
            id="doc1",
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"category": "test"},
        )

        # Mock list indexes
        provider.list_indexes.return_value = [
            IndexConfig(name="index1", dimension=1536, metric="cosine"),
            IndexConfig(name="index2", dimension=1536, metric="cosine"),
        ]

        # Mock index stats
        provider.get_index_stats.return_value = IndexStats(
            total_vector_count=1000,
            dimension=1536,
            index_fullness=0.75,
            last_updated="2023-01-01T00:00:00Z",
        )

        # Mock delete index
        provider.delete_index.return_value = True

        # Mock health check
        provider.health_check.return_value = True

        # Mock supported metrics
        provider.get_supported_metrics.return_value = [
            "cosine",
            "euclidean",
            "dotproduct",
        ]

        return provider

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return AIConfig(
            openai=OpenAIConfig(
                api_key="test_key", model="text-embedding-3-small", timeout=30
            ),
            pinecone=PineconeConfig(
                api_key="test_pinecone_key",
                environment="us-west-2",
                index_name="test-index",
                dimension=1536,
                metric="cosine",
            ),
            batch_size=100,
        )

    @pytest.fixture
    def service(self, mock_provider, mock_config):
        """Create service with mocked dependencies"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "ai_utils.services.vector_service.get_config", lambda: mock_config
            )
            return VectorService(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_upsert_documents_success(self, service):
        """Test successful document upsert"""
        documents = [
            VectorDocument(
                id="doc1",
                text="Sample text 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"category": "test"},
            ),
            VectorDocument(
                id="doc2",
                text="Sample text 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"category": "test"},
            ),
        ]

        result = await service.upsert_documents(documents)

        assert result["upserted_count"] == 2
        assert result["index_name"] == "test-index"
        assert result["batch_count"] == 1

    @pytest.mark.asyncio
    async def test_upsert_documents_with_job_tracking(self, service):
        """Test document upsert with job tracking"""
        documents = [
            VectorDocument(
                id="doc1",
                text="Sample text 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"category": "test"},
            )
        ]

        job_id = "test_job_123"
        await service.upsert_documents(documents, job_id=job_id)

        # Check job tracking
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.operation == "document_upsert"
        assert job.total_items == 1
        assert job.status == ProcessingStatus.COMPLETED
        assert job.processed_items == 2

    @pytest.mark.asyncio
    async def test_upsert_documents_failure_tracking(self, service):
        """Test document upsert failure tracking"""
        # Mock provider to fail
        service.provider.upsert_documents.side_effect = Exception("API Error")

        documents = [
            VectorDocument(
                id="doc1",
                text="Sample text 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"category": "test"},
            )
        ]

        job_id = "test_job_fail"

        with pytest.raises(Exception, match="API Error"):
            await service.upsert_documents(documents, job_id=job_id)

        # Check job tracking shows failure
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.status == ProcessingStatus.FAILED
        assert job.error_message == "API Error"

    @pytest.mark.asyncio
    async def test_search_similar_success(self, service):
        """Test successful similarity search"""
        query = VectorQuery(
            query="test query",
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            include_metadata=True,
            include_embeddings=True,
        )

        response = await service.search_similar(query)

        assert isinstance(response, VectorSearchResponse)
        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.total_results == 2
        assert response.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_by_text_success(self, service):
        """Test search by text (with embedding service)"""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_text.return_value = MagicMock(
            embedding=[0.1, 0.2, 0.3]
        )

        response = await service.search_by_text(
            text="test query", embedding_service=mock_embedding_service, top_k=5
        )

        assert isinstance(response, VectorSearchResponse)
        assert len(response.results) == 2

        # Verify embedding service was called
        mock_embedding_service.embed_text.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_delete_documents_success(self, service):
        """Test successful document deletion"""
        document_ids = ["doc1", "doc2", "doc3"]

        result = await service.delete_documents(document_ids)

        assert result["deleted_count"] == 3
        assert result["index_name"] == "test-index"

    @pytest.mark.asyncio
    async def test_delete_documents_with_job_tracking(self, service):
        """Test document deletion with job tracking"""
        document_ids = ["doc1", "doc2"]
        job_id = "test_delete_job"

        await service.delete_documents(document_ids, job_id=job_id)

        # Check job tracking
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.operation == "document_delete"
        assert job.total_items == 2
        assert job.status == ProcessingStatus.COMPLETED
        assert job.processed_items == 3

    @pytest.mark.asyncio
    async def test_get_document_success(self, service):
        """Test successful document retrieval"""
        document = await service.get_document("doc1")

        assert document is not None
        assert document.id == "doc1"
        assert document.text == "Sample text"
        assert document.embedding == [0.1, 0.2, 0.3]
        assert document.metadata["category"] == "test"

    @pytest.mark.asyncio
    async def test_bulk_upsert_with_embedding_success(self, service):
        """Test bulk upsert with automatic embedding"""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_texts_with_batching.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        texts = ["Sample text 1", "Sample text 2"]
        metadata_list = [{"category": "test1"}, {"category": "test2"}]

        result = await service.bulk_upsert_with_embedding(
            texts=texts,
            embedding_service=mock_embedding_service,
            metadata_list=metadata_list,
        )

        assert result["upserted_count"] == 2
        assert result["index_name"] == "test-index"

        # Verify embedding service was called
        mock_embedding_service.embed_texts_with_batching.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_bulk_upsert_with_embedding_job_tracking(self, service):
        """Test bulk upsert with job tracking"""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_texts_with_batching.return_value = [
            [0.1, 0.2, 0.3]
        ]

        texts = ["Sample text 1"]
        job_id = "test_bulk_job"

        await service.bulk_upsert_with_embedding(
            texts=texts, embedding_service=mock_embedding_service, job_id=job_id
        )

        # Check job tracking
        job = service.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.operation == "bulk_upsert_with_embedding"
        assert job.total_items == 1
        assert job.status == ProcessingStatus.COMPLETED
        assert job.processed_items == 2

    @pytest.mark.asyncio
    async def test_list_indexes_success(self, service):
        """Test successful index listing"""
        indexes = await service.list_indexes()

        assert len(indexes) == 2
        assert all(isinstance(index, IndexConfig) for index in indexes)
        assert indexes[0].name == "index1"
        assert indexes[1].name == "index2"

    @pytest.mark.asyncio
    async def test_get_index_stats_success(self, service):
        """Test successful index stats retrieval"""
        stats = await service.get_index_stats()

        assert isinstance(stats, IndexStats)
        assert stats.total_vector_count == 1000
        assert stats.dimension == 1536
        assert stats.index_fullness == 0.75

    @pytest.mark.asyncio
    async def test_delete_index_success(self, service):
        """Test successful index deletion"""
        result = await service.delete_index("test-index")

        assert result is True

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
        assert "cosine" in stats["supported_metrics"]
        assert "euclidean" in stats["supported_metrics"]
        assert "dotproduct" in stats["supported_metrics"]

    @pytest.mark.asyncio
    async def test_clear_cache(self, service):
        """Test cache clearing"""
        # Mock provider clear_cache method
        service.provider.clear_cache = AsyncMock()

        await service.clear_cache()

        # Verify provider clear_cache was called
        service.provider.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_no_method(self, service):
        """Test cache clearing when provider doesn't have clear_cache method"""
        # Remove clear_cache method from provider
        if hasattr(service.provider, "clear_cache"):
            delattr(service.provider, "clear_cache")

        # Should not raise an exception
        await service.clear_cache()
