"""
Unit tests for Pinecone vector store provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from ai_utils.config import AIConfig, OpenAIConfig, PineconeConfig
from ai_utils.models import (
    IndexConfig,
    IndexStats,
    VectorDocument,
    VectorQuery,
    VectorSearchResponse,
)
from ai_utils.providers.pinecone_store import PineconeVectorStoreProvider


class TestPineconeVectorStoreProvider:
    """Test Pinecone vector store provider"""

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
    def mock_pinecone_client(self):
        """Mock Pinecone client"""
        mock_client = MagicMock()

        # Mock index list
        mock_indexes = MagicMock()
        mock_indexes.names.return_value = ["test-index", "existing-index"]
        mock_client.list_indexes.return_value = mock_indexes

        # Mock index
        mock_index = MagicMock()
        mock_index.upsert.return_value = MagicMock()
        mock_index.query.return_value = MagicMock()
        mock_index.delete.return_value = MagicMock()
        mock_index.fetch.return_value = MagicMock()
        mock_index.describe_index_stats.return_value = MagicMock()

        # Mock query response
        mock_query_response = MagicMock()
        mock_match1 = MagicMock()
        mock_match1.id = "doc1"
        mock_match1.score = 0.95
        mock_match1.metadata = {"text": "Sample text 1", "category": "test"}
        mock_match1.values = [0.1, 0.2, 0.3]

        mock_match2 = MagicMock()
        mock_match2.id = "doc2"
        mock_match2.score = 0.85
        mock_match2.metadata = {"text": "Sample text 2", "category": "test"}
        mock_match2.values = [0.4, 0.5, 0.6]

        mock_query_response.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_query_response

        # Mock fetch response
        mock_fetch_response = MagicMock()
        mock_vector_data = MagicMock()
        mock_vector_data.id = "doc1"
        mock_vector_data.values = [0.1, 0.2, 0.3]
        mock_vector_data.metadata = {"text": "Sample text", "category": "test"}
        mock_fetch_response.vectors = {"doc1": mock_vector_data}
        mock_index.fetch.return_value = mock_fetch_response

        # Mock index stats
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 1000
        mock_stats.dimension = 1536
        mock_stats.index_fullness = 0.75
        mock_stats.last_updated = "2023-01-01T00:00:00Z"
        mock_index.describe_index_stats.return_value = mock_stats

        mock_client.Index.return_value = mock_index

        return mock_client

    @pytest.fixture
    def provider(self, mock_config, mock_pinecone_client):
        """Create provider with mocked dependencies"""
        with patch(
            "ai_utils.providers.pinecone_store.Pinecone",
            return_value=mock_pinecone_client,
        ):
            return PineconeVectorStoreProvider(config=mock_config)

    @pytest.mark.asyncio
    async def test_upsert_documents_success(self, provider):
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

        result = await provider.upsert_documents(documents)

        assert result["upserted_count"] == 2
        assert result["index_name"] == "test-index"
        assert result["batch_count"] == 1

    @pytest.mark.asyncio
    async def test_search_similar_success(self, provider):
        """Test successful similarity search"""
        query = VectorQuery(
            query="test query",
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            include_metadata=True,
            include_embeddings=True,
        )

        response = await provider.search_similar(query)

        assert isinstance(response, VectorSearchResponse)
        assert len(response.results) == 2
        assert response.query == "test query"
        assert response.total_results == 2
        assert response.search_time_ms > 0

        # Check first result
        result1 = response.results[0]
        assert result1.id == "doc1"
        assert result1.text == "Sample text 1"
        assert result1.score == 0.95
        assert result1.metadata["category"] == "test"
        assert result1.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_search_similar_with_filters(self, provider):
        """Test similarity search with filters"""
        query = VectorQuery(
            query="test query",
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            filters={"category": "test"},
            include_metadata=True,
        )

        response = await provider.search_similar(query)

        assert len(response.results) == 2
        # Verify that the provider called query with filters
        provider.provider._get_index.return_value.query.assert_called_once()
        call_args = provider.provider._get_index.return_value.query.call_args[1]
        assert "filter" in call_args

    @pytest.mark.asyncio
    async def test_delete_documents_success(self, provider):
        """Test successful document deletion"""
        document_ids = ["doc1", "doc2", "doc3"]

        result = await provider.delete_documents(document_ids)

        assert result["deleted_count"] == 3
        assert result["index_name"] == "test-index"

    @pytest.mark.asyncio
    async def test_get_document_success(self, provider):
        """Test successful document retrieval"""
        document = await provider.get_document("doc1")

        assert document is not None
        assert document.id == "doc1"
        assert document.text == "Sample text"
        assert document.embedding == [0.1, 0.2, 0.3]
        assert document.metadata["category"] == "test"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, provider):
        """Test document retrieval when document doesn't exist"""
        # Mock fetch to return empty result
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {}
        provider.provider._get_index.return_value.fetch.return_value = (
            mock_fetch_response
        )

        document = await provider.get_document("nonexistent")

        assert document is None

    @pytest.mark.asyncio
    async def test_list_indexes_success(self, provider):
        """Test successful index listing"""
        indexes = await provider.list_indexes()

        assert len(indexes) == 2
        assert all(isinstance(index, IndexConfig) for index in indexes)

    @pytest.mark.asyncio
    async def test_get_index_stats_success(self, provider):
        """Test successful index stats retrieval"""
        stats = await provider.get_index_stats()

        assert isinstance(stats, IndexStats)
        assert stats.total_vector_count == 1000
        assert stats.dimension == 1536
        assert stats.index_fullness == 0.75

    @pytest.mark.asyncio
    async def test_delete_index_success(self, provider):
        """Test successful index deletion"""
        result = await provider.delete_index("test-index")

        assert result is True
        # Verify that the index was removed from cache
        assert "test-index" not in provider._index_cache

    def test_get_supported_metrics(self, provider):
        """Test getting supported metrics"""
        metrics = provider.get_supported_metrics()

        assert "cosine" in metrics
        assert "euclidean" in metrics
        assert "dotproduct" in metrics
        assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check"""
        result = await provider.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test failed health check"""
        # Mock client to raise exception
        provider.client.list_indexes.side_effect = Exception("API Error")

        result = await provider.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_cache(self, provider):
        """Test cache clearing"""
        # Add some items to cache
        provider._index_cache["test-index"] = MagicMock()
        provider._index_cache["another-index"] = MagicMock()

        assert len(provider._index_cache) == 2

        await provider.clear_cache()

        assert len(provider._index_cache) == 0

    @pytest.mark.asyncio
    async def test_create_index_when_not_exists(self, provider):
        """Test index creation when it doesn't exist"""
        # Mock client to return empty index list
        mock_indexes = MagicMock()
        mock_indexes.names.return_value = []
        provider.client.list_indexes.return_value = mock_indexes

        # Mock create_index method
        provider.client.create_index = MagicMock()

        # This should trigger index creation
        await provider._get_index("new-index")

        # Verify create_index was called
        provider.client.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_documents_batching(self, provider):
        """Test document upsert with batching"""
        # Create more documents than batch size
        documents = []
        for i in range(250):  # More than batch size of 100
            documents.append(
                VectorDocument(
                    id=f"doc{i}",
                    text=f"Sample text {i}",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"category": "test"},
                )
            )

        result = await provider.upsert_documents(documents)

        assert result["upserted_count"] == 250
        assert result["batch_count"] == 3  # 250 / 100 = 3 batches

    @pytest.mark.asyncio
    async def test_delete_documents_batching(self, provider):
        """Test document deletion with batching"""
        # Create more document IDs than batch size
        document_ids = [f"doc{i}" for i in range(250)]

        result = await provider.delete_documents(document_ids)

        assert result["deleted_count"] == 250
