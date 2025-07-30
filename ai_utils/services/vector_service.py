"""
High-level vector service that orchestrates vector store operations.
Provides business logic, batching, and performance monitoring.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
from collections import defaultdict
from datetime import datetime
from ..interfaces.vector_store import VectorStoreProvider
from ..models import (
    VectorDocument, VectorQuery, VectorSearchResult, VectorSearchResponse,
    IndexConfig, IndexStats, ProcessingJob, ProcessingStatus
)
from ..config import get_config
from ..utils.batch_operations import batch_process
from ..utils.performance import PerformanceBenchmark

logger = logging.getLogger(__name__)

class VectorService:
    """High-level vector service with CRUD operations and performance monitoring"""
    
    def __init__(self, provider: VectorStoreProvider = None):
        """Initialize vector service with provider"""
        self.config = get_config()
        self.provider = provider
        self._active_jobs: Dict[str, ProcessingJob] = {}
        self.performance_tracker = PerformanceBenchmark()
        self.stats = defaultdict(list)
    
    async def upsert_documents(
        self, 
        documents: List[VectorDocument], 
        index_name: str = None,
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Upsert documents to vector store.
        
        Args:
            documents: List of documents to upsert
            index_name: Index name (default from config)
            job_id: Optional job ID for tracking
            
        Returns:
            Upsert result with statistics
        """
        start_time = time.time()
        
        # Create processing job for tracking
        job = ProcessingJob(
            job_id=job_id or f"upsert_{int(time.time())}",
            operation="document_upsert",
            total_items=len(documents),
            status=ProcessingStatus.PROCESSING
        )
        self._active_jobs[job.job_id] = job
        
        try:
            result = await self.provider.upsert_documents(documents)
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = result.processed_items if hasattr(result, 'processed_items') else len(documents)
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Upserted {job.processed_items} documents in {processing_time:.2f}ms")
            
            # Return a dictionary with the result data
            return {
                "upserted_count": result.processed_items if hasattr(result, 'processed_items') else len(documents),
                "job_id": result.job_id if hasattr(result, 'job_id') else job.job_id,
                "status": result.status.value if hasattr(result, 'status') else "completed"
            }
            
        except Exception as e:
            # Update job status
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to upsert documents in {processing_time:.2f}ms: {str(e)}")
            raise
    
    async def search_similar(
        self, 
        query: VectorQuery, 
        index_name: str = None
    ) -> VectorSearchResponse:
        """
        Search for similar documents.
        
        Args:
            query: Search query with embedding and parameters
            index_name: Index name (default from config)
            
        Returns:
            Search response with results
        """
        start_time = time.time()
        
        try:
            response = await self.provider.search_similar(query)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Search completed in {processing_time:.2f}ms, found {len(response.results)} results")
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to search in {processing_time:.2f}ms: {str(e)}")
            raise
    
    async def search_by_text(
        self, 
        text: str, 
        embedding_service=None,  # Now optional since Weaviate handles embeddings natively
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        include_metadata: bool = True,
        include_embeddings: bool = False,
        index_name: str = None
    ) -> VectorSearchResponse:
        """
        Search by text using Weaviate's native vectorization.
        
        Args:
            text: Text to search for
            embedding_service: Service to embed the text (unused with native vectorization)
            top_k: Number of results to return
            filters: Metadata filters
            include_metadata: Include metadata in results
            include_embeddings: Include embeddings in results
            index_name: Index name (default from config)
            
        Returns:
            Search response with results
        """
        try:
            # Create search query - Weaviate will handle embedding natively
            query = VectorQuery(
                query=text,
                embedding=None,  # No embedding needed - Weaviate handles this
                top_k=top_k,
                filters=filters,
                include_metadata=include_metadata,
                include_embeddings=include_embeddings
            )
            
            # Perform the actual vector search
            result = await self.search_similar(query, index_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to search by text: {str(e)}")
            raise
    
    async def delete_documents(
        self, 
        document_ids: List[str], 
        index_name: str = None,
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Delete documents from vector store.
        
        Args:
            document_ids: List of document IDs to delete
            index_name: Index name (default from config)
            job_id: Optional job ID for tracking
            
        Returns:
            Delete result with statistics
        """
        start_time = time.time()
        
        # Create processing job for tracking
        job = ProcessingJob(
            job_id=job_id or f"delete_{int(time.time())}",
            operation="document_delete",
            total_items=len(document_ids),
            status=ProcessingStatus.PROCESSING
        )
        self._active_jobs[job.job_id] = job
        
        try:
            result = await self.provider.delete_documents(document_ids)
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = len(document_ids)
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Deleted {job.processed_items} documents in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            # Update job status
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to delete documents in {processing_time:.2f}ms: {str(e)}")
            raise
    
    async def get_document(
        self, 
        document_id: str, 
        index_name: str = None
    ) -> Optional[VectorDocument]:
        """
        Get a single document by ID.
        
        Args:
            document_id: Document ID to retrieve
            index_name: Index name (default from config)
            
        Returns:
            Document if found, None otherwise
        """
        try:
            return await self.provider.get_document(document_id)
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise
    
    async def bulk_upsert_with_embedding(
        self, 
        texts: List[str], 
        embedding_service=None,  # Now optional since Weaviate handles embeddings natively
        metadata_list: List[Dict[str, Any]] = None,
        index_name: str = None,
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Bulk upsert texts using Weaviate's native vectorization.
        
        Args:
            texts: List of texts to embed and upsert
            embedding_service: Service to embed the texts (unused with native vectorization)
            metadata_list: List of metadata for each text
            index_name: Index name (default from config)
            job_id: Optional job ID for tracking
            
        Returns:
            Upsert result with statistics
        """
        start_time = time.time()
        
        # Create processing job for tracking
        job = ProcessingJob(
            job_id=job_id or f"bulk_upsert_{int(time.time())}",
            operation="bulk_upsert_with_native_embedding",
            total_items=len(texts),
            status=ProcessingStatus.PROCESSING
        )
        self._active_jobs[job.job_id] = job
        
        try:
            # Create documents without embeddings - Weaviate will generate them natively
            documents = []
            for i, text in enumerate(texts):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                document = VectorDocument(
                    id=f"doc_{int(time.time())}_{i}",
                    text=text,
                    embedding=None,  # No embedding needed - Weaviate handles this
                    metadata=metadata
                )
                documents.append(document)
            
            # Upsert documents
            result = await self.upsert_documents(documents, index_name)
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = result["upserted_count"]
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Bulk upsert with native embedding completed in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            # Update job status
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Failed to bulk upsert with native embedding in {processing_time:.2f}ms: {str(e)}")
            raise
    
    async def list_indexes(self) -> List[IndexConfig]:
        """
        List all available indexes.
        
        Returns:
            List of index configurations
        """
        try:
            return await self.provider.list_indexes()
        except Exception as e:
            logger.error(f"Failed to list indexes: {str(e)}")
            raise
    
    async def get_index_stats(self, index_name: str = None) -> IndexStats:
        """
        Get statistics for an index.
        
        Args:
            index_name: Index name (default from config)
            
        Returns:
            Index statistics
        """
        try:
            return await self.provider.get_index_stats(index_name)
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            raise
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            return await self.provider.delete_index(index_name)
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Get status of a processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Processing job status or None if not found
        """
        return self._active_jobs.get(job_id)
    
    def get_active_jobs(self) -> List[ProcessingJob]:
        """
        Get list of active processing jobs.
        
        Returns:
            List of active jobs
        """
        return list(self._active_jobs.values())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the vector service is healthy.
        
        Returns:
            Health check result dictionary
        """
        try:
            with self.performance_tracker.measure("health_check") as timer:
                is_healthy = await self.provider.health_check()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "provider": type(self.provider).__name__,
                "health_check_time_ms": timer.elapsed_ms
            }
        except Exception as e:
            logger.error(f"Vector service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the vector service.
        
        Returns:
            Performance statistics
        """
        active_jobs = self.get_active_jobs()
        
        return {
            "active_jobs": len(active_jobs),
            "completed_jobs": len([j for j in active_jobs if j.status == ProcessingStatus.COMPLETED]),
            "failed_jobs": len([j for j in active_jobs if j.status == ProcessingStatus.FAILED]),
            "total_items_processed": sum(j.processed_items for j in active_jobs),
            "supported_metrics": self.provider.get_supported_metrics() if self.provider else []
        }
    
    async def clear_cache(self):
        """Clear the provider's cache"""
        if hasattr(self.provider, 'clear_cache'):
            await self.provider.clear_cache() 