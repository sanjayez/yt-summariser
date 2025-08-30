"""
High-level embedding service that orchestrates embedding operations.
Provides business logic, batching, and performance monitoring.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..config import get_config
from ..interfaces.embeddings import EmbeddingProvider
from ..models import (
    BatchEmbeddingRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    ProcessingJob,
    ProcessingStatus,
)
from ..utils.performance import PerformanceBenchmark

logger = logging.getLogger(__name__)


class EmbeddingService:
    """High-level embedding service with batching and performance monitoring"""

    def __init__(self, provider: EmbeddingProvider = None):
        """Initialize embedding service with provider"""
        self.config = get_config()
        self.provider = provider
        self._active_jobs: dict[str, ProcessingJob] = {}
        self.performance_tracker = PerformanceBenchmark()
        self.stats = defaultdict(list)

    async def embed_text(
        self,
        text: str,
        model: str | None = None,
        user: str | None = None,
        job_id: str | None = None,
    ) -> EmbeddingResponse:
        """
        Embed a single text.

        Args:
            text: Text to embed
            model: Model to use (default from config)
            user: User identifier
            job_id: Optional job ID for tracking

        Returns:
            Embedding response
        """
        job_id = job_id or f"embed_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(
            job_id=job_id,
            operation="text_embedding",
            total_items=1,
            status=ProcessingStatus.PROCESSING,
        )
        self._active_jobs[job_id] = job

        try:
            # Create embedding request
            request = EmbeddingRequest(
                text=text,
                model=model or self.config.openai.embedding_model,
                user=user or self.config.openai.user,
            )

            # Track performance
            with self.performance_tracker.measure("text_embedding") as timer:
                response = await self.provider.embed_text(request)

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = 1
            job.updated_at = datetime.now()

            # Track statistics
            self.stats["embedding_time_ms"].append(timer.elapsed_ms)
            self.stats["embedding_tokens"].append(
                response.usage.get("prompt_tokens", 0)
            )

            return response

        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()
            raise

    async def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
        user: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Embed multiple texts in batch.

        Args:
            texts: List of texts to embed
            model: Model to use (default from config)
            user: User identifier
            job_id: Optional job ID for tracking

        Returns:
            Batch embedding response
        """
        job_id = job_id or f"batch_embed_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(
            job_id=job_id,
            operation="batch_embedding",
            total_items=len(texts),
            status=ProcessingStatus.PROCESSING,
        )
        self._active_jobs[job_id] = job

        try:
            # Create batch request
            batch_request = BatchEmbeddingRequest(
                texts=texts,
                model=model or self.config.openai.embedding_model,
                user=user or self.config.openai.user,
            )

            # Track performance
            with self.performance_tracker.measure("batch_embedding") as timer:
                batch_response = await self.provider.embed_batch(batch_request)

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = len(texts)
            job.updated_at = datetime.now()

            # Track statistics
            self.stats["batch_embedding_time_ms"].append(timer.elapsed_ms)
            self.stats["batch_embedding_count"].append(len(texts))
            self.stats["batch_embedding_tokens"].append(
                batch_response.usage.get("prompt_tokens", 0)
            )

            return {
                "embeddings": batch_response.embeddings,
                "model": batch_response.model,
                "usage": batch_response.usage,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    async def embed_texts_with_batching(
        self,
        texts: list[str],
        model: str = None,
        user: str = None,
        batch_size: int = None,
    ) -> list[list[float]]:
        """
        Embed texts using intelligent batching for large datasets.

        Args:
            texts: List of texts to embed
            model: Model to use
            user: User identifier
            batch_size: Size of each batch (default from config)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self.config.batch_size
        all_embeddings = []

        # Process in batches
        failed_batches = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                response = await self.embed_batch(
                    texts=batch_texts, model=model, user=user
                )
                all_embeddings.extend(response["embeddings"])

            except Exception as e:
                failed_batches += 1
                logger.error(f"Failed to embed batch {i // batch_size + 1}: {str(e)}")
                # Continue with next batch instead of failing completely
                continue

        # If all batches failed, raise an exception
        if failed_batches > 0 and len(all_embeddings) == 0:
            raise ValueError(f"All {failed_batches} embedding batches failed")

        return all_embeddings

    async def embed_documents(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id",
        model: str = None,
        user: str = None,
    ) -> list[dict[str, Any]]:
        """
        Embed a list of documents with metadata preservation.

        Args:
            documents: List of documents with text and metadata
            text_field: Field name containing text to embed
            id_field: Field name containing document ID
            model: Model to use
            user: User identifier

        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []

        # Extract texts for embedding
        texts = [doc.get(text_field, "") for doc in documents]

        # Get embeddings
        embeddings = await self.embed_texts_with_batching(
            texts=texts, model=model, user=user
        )

        # Combine documents with embeddings
        result = []
        for i, doc in enumerate(documents):
            if i < len(embeddings):
                result.append(
                    {
                        **doc,
                        "embedding": embeddings[i],
                        "embedding_model": model or self.config.openai.embedding_model,
                    }
                )

        return result

    def get_job_status(self, job_id: str) -> ProcessingJob | None:
        """
        Get status of a processing job.

        Args:
            job_id: Job identifier

        Returns:
            Processing job status or None if not found
        """
        return self._active_jobs.get(job_id)

    def get_active_jobs(self) -> list[ProcessingJob]:
        """
        Get list of active processing jobs.

        Returns:
            List of active jobs
        """
        return list(self._active_jobs.values())

    async def health_check(self) -> dict[str, Any]:
        """
        Check if the embedding service is healthy.

        Returns:
            Health check result
        """
        try:
            with self.performance_tracker.measure("health_check") as timer:
                is_healthy = await self.provider.health_check()

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "provider": type(self.provider).__name__,
                "embedding_model": self.config.openai.embedding_model,
                "supported_models": self.provider.get_supported_models(),
                "health_check_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the embedding service.

        Returns:
            Performance statistics
        """
        active_jobs = self.get_active_jobs()

        return {
            "total_jobs": len(self._active_jobs),
            "active_jobs": len(active_jobs),
            "completed_jobs": len(
                [j for j in active_jobs if j.status == ProcessingStatus.COMPLETED]
            ),
            "failed_jobs": len(
                [j for j in active_jobs if j.status == ProcessingStatus.FAILED]
            ),
            "total_items_processed": sum(j.processed_items for j in active_jobs),
            "performance_metrics": dict(self.stats),
            "benchmark_stats": self.performance_tracker.get_stats(),
        }
