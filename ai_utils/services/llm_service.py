"""
LLM Service Layer
High-level service for LLM operations with job tracking and performance monitoring
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..interfaces.llm import LLMProvider
from ..models import (
    ChatMessage,
    ChatRequest,
    ChatRole,
    ProcessingJob,
    ProcessingStatus,
    RAGQuery,
    TextGenerationRequest,
    VectorSearchResult,
)
from ..utils.performance import PerformanceBenchmark

logger = logging.getLogger(__name__)


class LLMService:
    """High-level LLM service with job tracking and performance monitoring"""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.jobs: dict[str, ProcessingJob] = {}
        self.performance_tracker = PerformanceBenchmark()
        self.stats = defaultdict(list)  # type: ignore

    async def generate_rag_response(
        self,
        query: RAGQuery,
        context_documents: list[VectorSearchResult],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate RAG response with job tracking"""
        job_id = job_id or f"rag_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(  # type: ignore
            job_id=job_id,
            operation="rag_generation",
            total_items=1,
            status=ProcessingStatus.PROCESSING,
        )
        self.jobs[job_id] = job

        try:
            # Track performance
            with self.performance_tracker.measure("rag_generation") as timer:
                response = await self.provider.generate_rag_response(
                    query, context_documents
                )

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = 1
            job.updated_at = datetime.now()

            # Track stats
            self.stats["rag_generation_time_ms"].append(timer.elapsed_ms)
            self.stats["rag_response_length"].append(len(response.answer))
            self.stats["rag_context_documents"].append(len(context_documents))

            return {
                "response": response,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate text with job tracking"""
        job_id = job_id or f"text_gen_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(  # type: ignore
            job_id=job_id,
            operation="text_generation",
            total_items=1,
            status=ProcessingStatus.PROCESSING,
        )
        self.jobs[job_id] = job

        try:
            # Track performance
            with self.performance_tracker.measure("text_generation") as timer:
                text = await self.provider.generate_text(  # type: ignore
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    model=model,
                )

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = 1
            job.updated_at = datetime.now()

            # Track stats
            self.stats["text_generation_time_ms"].append(timer.elapsed_ms)
            self.stats["text_generation_length"].append(len(text))

            return {
                "text": text,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    async def generate_text_with_response(
        self, request: TextGenerationRequest, job_id: str | None = None
    ) -> dict[str, Any]:
        """Generate text with detailed response and job tracking"""
        job_id = job_id or f"text_gen_detailed_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(  # type: ignore
            job_id=job_id,
            operation="text_generation_detailed",
            total_items=1,
            status=ProcessingStatus.PROCESSING,
        )
        self.jobs[job_id] = job

        try:
            # Track performance
            with self.performance_tracker.measure("text_generation_detailed") as timer:
                response = await self.provider.generate_text_with_response(request)  # type: ignore

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = 1
            job.updated_at = datetime.now()

            # Track stats
            self.stats["text_generation_detailed_time_ms"].append(timer.elapsed_ms)
            self.stats["text_generation_tokens"].append(response.usage.total_tokens)

            return {
                "response": response,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Error in detailed text generation: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    async def chat_completion(
        self, request: ChatRequest, job_id: str | None = None
    ) -> dict[str, Any]:
        """Complete chat conversation with job tracking"""
        job_id = job_id or f"chat_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(  # type: ignore
            job_id=job_id,
            operation="chat_completion",
            total_items=1,
            status=ProcessingStatus.PROCESSING,
        )
        self.jobs[job_id] = job

        try:
            # Track performance
            with self.performance_tracker.measure("chat_completion") as timer:
                response = await self.provider.chat_completion(request)  # type: ignore

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.processed_items = 1
            job.updated_at = datetime.now()

            # Track stats
            self.stats["chat_completion_time_ms"].append(timer.elapsed_ms)
            self.stats["chat_completion_tokens"].append(response.usage.total_tokens)

            return {
                "response": response,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            error_message = str(e)

            # Update job status
            job.status = ProcessingStatus.FAILED
            job.error_message = error_message
            job.updated_at = datetime.now()

            # Special handling for MAX_TOKENS error - don't retry, fall back immediately
            if "MAX_TOKENS" in error_message:
                logger.warning(
                    f"MAX_TOKENS error encountered for job {job_id}, using fallback instead of retry"
                )
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"MAX_TOKENS: {error_message}",
                    "processing_time_ms": 0,
                    "response": None,
                }

            logger.error(
                f"Error in LLM service chat completion for job {job_id}: {error_message}"
            )

            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_message,
                "processing_time_ms": 0,
                "response": None,
            }

    async def batch_text_generation(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate text for multiple prompts with job tracking"""
        job_id = job_id or f"batch_text_{uuid4().hex[:8]}"

        # Create job
        job = ProcessingJob(  # type: ignore
            job_id=job_id,
            operation="batch_text_generation",
            total_items=len(prompts),
            status=ProcessingStatus.PROCESSING,
        )
        self.jobs[job_id] = job

        try:
            results = []

            # Track performance
            with self.performance_tracker.measure("batch_text_generation") as timer:
                # Process prompts concurrently
                tasks = [
                    self.provider.generate_text(  # type: ignore
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                    )
                    for prompt in prompts
                ]

                # Execute with progress tracking
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    try:
                        result = await task
                        results.append({"text": result, "index": i})
                        job.processed_items = len(results)
                        job.updated_at = datetime.now()
                    except Exception as e:
                        logger.error(f"Error processing prompt {i}: {str(e)}")
                        results.append({"error": str(e), "index": i})

            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.updated_at = datetime.now()

            # Track stats
            self.stats["batch_text_generation_time_ms"].append(timer.elapsed_ms)
            self.stats["batch_text_generation_count"].append(len(results))

            return {
                "results": results,
                "job_id": job_id,
                "status": "completed",
                "processing_time_ms": timer.elapsed_ms,
                "total_processed": len(results),
            }

        except Exception as e:
            logger.error(f"Error in batch text generation: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """Check LLM service health"""
        try:
            with self.performance_tracker.measure("health_check") as timer:
                is_healthy = await self.provider.health_check()

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "provider": type(self.provider).__name__,
                "supported_models": self.provider.get_supported_models(),
                "health_check_time_ms": timer.elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    def get_job_status(self, job_id: str) -> ProcessingJob | None:
        """Get status of a specific job"""
        return self.jobs.get(job_id)

    def get_active_jobs(self) -> list[ProcessingJob]:
        """Get all active jobs"""
        return [
            job
            for job in self.jobs.values()
            if job.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]
        ]

    def get_completed_jobs(self) -> list[ProcessingJob]:
        """Get all completed jobs"""
        return [
            job
            for job in self.jobs.values()
            if job.status == ProcessingStatus.COMPLETED
        ]

    def get_failed_jobs(self) -> list[ProcessingJob]:
        """Get all failed jobs"""
        return [
            job for job in self.jobs.values() if job.status == ProcessingStatus.FAILED
        ]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_jobs": len(self.jobs),
            "active_jobs": len(self.get_active_jobs()),
            "completed_jobs": len(self.get_completed_jobs()),
            "failed_jobs": len(self.get_failed_jobs()),
            "performance_metrics": dict(self.stats),
            "benchmark_stats": self.performance_tracker.get_stats(),
        }

    def clear_old_jobs(self, max_age_hours: int = 24):
        """Clear old jobs to prevent memory buildup"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        jobs_to_remove = [
            job_id
            for job_id, job in self.jobs.items()
            if job.created_at.timestamp() < cutoff_time
        ]

        for job_id in jobs_to_remove:
            del self.jobs[job_id]

        logger.info(f"Cleared {len(jobs_to_remove)} old jobs")

    def create_simple_chat_message(self, role: str, content: str) -> ChatMessage:
        """Helper method to create simple chat messages"""
        return ChatMessage(role=ChatRole(role), content=content)  # type: ignore

    def create_simple_chat_request(
        self,
        user_message: str,
        system_message: str | None = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatRequest:
        """Helper method to create simple chat requests"""
        messages = []

        if system_message:
            messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_message))  # type: ignore

        messages.append(ChatMessage(role=ChatRole.USER, content=user_message))  # type: ignore

        return ChatRequest(  # type: ignore
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
