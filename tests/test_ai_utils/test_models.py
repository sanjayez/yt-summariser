"""
Unit tests for Pydantic models.
"""

import pytest
from datetime import datetime
from ai_utils.models import (
    EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse,
    VectorDocument, VectorQuery, VectorSearchResult, VectorSearchResponse,
    RAGQuery, RAGResponse, ProcessingStatus, ProcessingJob
)

class TestEmbeddingModels:
    """Test embedding-related models"""
    
    def test_embedding_request_valid(self):
        """Test valid embedding request"""
        request = EmbeddingRequest(
            text="Hello world",
            model="text-embedding-3-small",
            user="test_user"
        )
        assert request.text == "Hello world"
        assert request.model == "text-embedding-3-small"
        assert request.user == "test_user"
    
    def test_embedding_request_empty_text(self):
        """Test embedding request with empty text"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            EmbeddingRequest(text="   ")
    
    def test_embedding_response_valid(self):
        """Test valid embedding response"""
        response = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 10, "total_tokens": 10}
        )
        assert len(response.embedding) == 3
        assert response.model == "text-embedding-3-small"
        assert response.usage["prompt_tokens"] == 10
    
    def test_batch_embedding_request_valid(self):
        """Test valid batch embedding request"""
        request = BatchEmbeddingRequest(
            texts=["Hello", "World", "Test"],
            model="text-embedding-3-small"
        )
        assert len(request.texts) == 3
        assert request.model == "text-embedding-3-small"
    
    def test_batch_embedding_request_empty_texts(self):
        """Test batch embedding request with empty texts"""
        with pytest.raises(ValueError, match="All texts must be non-empty"):
            BatchEmbeddingRequest(texts=["Hello", "   ", "World"])

class TestVectorModels:
    """Test vector store models"""
    
    def test_vector_document_valid(self):
        """Test valid vector document"""
        doc = VectorDocument(
            id="doc_123",
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        assert doc.id == "doc_123"
        assert doc.text == "Sample text"
        assert len(doc.embedding) == 3
        assert doc.metadata["source"] == "test"
    
    def test_vector_document_empty_id(self):
        """Test vector document with empty ID"""
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            VectorDocument(
                id="   ",
                text="Sample text",
                embedding=[0.1, 0.2, 0.3]
            )
    
    def test_vector_query_valid(self):
        """Test valid vector query"""
        query = VectorQuery(
            query="search term",
            top_k=10,
            filters={"category": "test"},
            include_metadata=True
        )
        assert query.query == "search term"
        assert query.top_k == 10
        assert query.filters["category"] == "test"
        assert query.include_metadata is True
    
    def test_vector_query_top_k_validation(self):
        """Test vector query top_k validation"""
        with pytest.raises(ValueError):
            VectorQuery(query="test", top_k=0)  # Should fail (ge=1)
        
        with pytest.raises(ValueError):
            VectorQuery(query="test", top_k=101)  # Should fail (le=100)

class TestRAGModels:
    """Test RAG models"""
    
    def test_rag_query_valid(self):
        """Test valid RAG query"""
        query = RAGQuery(
            query="What is AI?",
            top_k=5,
            temperature=0.7,
            max_tokens=100
        )
        assert query.query == "What is AI?"
        assert query.top_k == 5
        assert query.temperature == 0.7
        assert query.max_tokens == 100
    
    def test_rag_query_temperature_validation(self):
        """Test RAG query temperature validation"""
        with pytest.raises(ValueError):
            RAGQuery(query="test", temperature=2.5)  # Should fail (le=2.0)
        
        with pytest.raises(ValueError):
            RAGQuery(query="test", temperature=-0.1)  # Should fail (ge=0.0)

class TestProcessingModels:
    """Test processing job models"""
    
    def test_processing_status_enum(self):
        """Test processing status enum values"""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.CANCELLED == "cancelled"
    
    def test_processing_job_valid(self):
        """Test valid processing job"""
        job = ProcessingJob(
            job_id="job_123",
            operation="embedding",
            total_items=100,
            processed_items=50
        )
        assert job.job_id == "job_123"
        assert job.status == ProcessingStatus.PENDING
        assert job.operation == "embedding"
        assert job.total_items == 100
        assert job.processed_items == 50 