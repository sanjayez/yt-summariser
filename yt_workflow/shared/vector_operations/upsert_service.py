"""Main upsert service with retry mechanisms and batch processing"""

import asyncio
import time
from typing import Any

from ai_utils.models import VectorDocument
from ai_utils.services.registry import get_vector_service
from telemetry import get_logger
from yt_workflow.transcript.types.models import MacroChunk, MicroChunk

from .batch_operations import create_batch_groups
from .document_converters import macro_to_vector_document, micro_to_vector_document
from .status_utils import calculate_status, process_batch_results

logger = get_logger(__name__)


async def upsert_batch(
    vector_service,
    documents: list[VectorDocument],
    batch_idx: int,
    chunk_type: str,
    video_id: str,
    max_retries: int = 3,
) -> int | BaseException:
    """Upsert a single batch of documents to vector store with retry mechanism"""
    last_error = None

    for attempt in range(max_retries + 1):  # 1 initial attempt + max_retries retries
        try:
            result = await vector_service.upsert_documents(
                documents=documents, job_id=f"{chunk_type}_batch_{video_id}_{batch_idx}"
            )

            upserted = result.get("upserted_count", 0)

            # Log success (with retry info if applicable)
            if attempt > 0:
                logger.info(
                    f"Upserted batch {batch_idx + 1} of {chunk_type} chunks: "
                    f"{upserted} documents (succeeded after {attempt} retries)"
                )
            else:
                logger.debug(
                    f"Upserted batch {batch_idx + 1} of {chunk_type} chunks: "
                    f"{upserted} documents"
                )
            return upserted

        except Exception as e:
            last_error = e

            # If this is the last attempt, don't wait
            if attempt == max_retries:
                break

            # Exponential backoff: 1s, 2s, 4s
            delay = 2**attempt
            logger.warning(
                f"Batch {batch_idx + 1} of {chunk_type} failed (attempt {attempt + 1}), "
                f"retrying in {delay}s: {str(e)}"
            )
            await asyncio.sleep(delay)

    # Log final failure
    logger.error(
        f"Final failure for batch {batch_idx + 1} of {chunk_type} "
        f"after {max_retries} retries: {str(last_error)}"
    )
    return last_error


async def upsert_transcript_chunks(
    micro_chunks: list[MicroChunk],
    macro_chunks: list[MacroChunk],
    video_id: str,
    batch_size: int = 200,
) -> dict[str, Any]:
    """
    Upsert micro and macro chunks to vector store in parallel batches.

    Args:
        micro_chunks: List of MicroChunk objects
        macro_chunks: List of MacroChunk objects
        video_id: Video identifier
        batch_size: Number of documents per batch (default 200)

    Returns:
        Dictionary with upsert metrics
    """

    logger.info(
        f"Starting vector upsert for video {video_id}: "
        f"{len(micro_chunks)} micro chunks, {len(macro_chunks)} macro chunks"
    )

    try:
        # Get vector service instance early (fail-fast if service unavailable)
        vector_service = get_vector_service()

        # Convert chunks to VectorDocument format
        micro_documents = [
            micro_to_vector_document(micro, video_id) for micro in micro_chunks
        ]

        macro_documents = [
            macro_to_vector_document(macro, video_id) for macro in macro_chunks
        ]

        # Create batches
        micro_batches = create_batch_groups(micro_documents, batch_size)
        macro_batches = create_batch_groups(macro_documents, batch_size)

        logger.info(
            f"Created {len(micro_batches)} micro batches and "
            f"{len(macro_batches)} macro batches (batch_size={batch_size})"
        )

    except Exception as e:
        logger.error(f"Vector upsert setup failed for video {video_id}: {str(e)}")
        total_chunks_attempted = len(micro_chunks) + len(macro_chunks)
        return {
            "status": "failed",
            "chunks_upserted": 0,
            "chunks_passed": total_chunks_attempted,
            "failed_chunks": {"setup_error": str(e), "error_type": type(e).__name__},
        }

    # Create tasks for parallel execution
    tasks = []

    # Add micro batch tasks
    for idx, batch in enumerate(micro_batches):
        tasks.append(upsert_batch(vector_service, batch, idx, "micro", video_id))

    # Add macro batch tasks
    for idx, batch in enumerate(macro_batches):
        tasks.append(upsert_batch(vector_service, batch, idx, "macro", video_id))

    start_time = time.time()

    # Execute all batches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time() - start_time

    # Process results using helper function
    batch_metrics = process_batch_results(results, len(micro_batches))

    # Extract metrics
    micro_upserted = batch_metrics["micro_upserted"]
    macro_upserted = batch_metrics["macro_upserted"]
    micro_failures = batch_metrics["micro_failures"]
    macro_failures = batch_metrics["macro_failures"]

    # Calculate totals
    total_chunks = len(micro_chunks) + len(macro_chunks)
    total_upserted = micro_upserted + macro_upserted

    # Determine status
    status = calculate_status(total_upserted, total_chunks)

    logger.info(
        f"Vector upsert completed for video {video_id}: "
        f"{total_upserted}/{total_chunks} chunks upserted, status: {status}"
        f"time to upsert: {end_time:.2f}s"
    )

    if micro_failures or macro_failures:
        logger.warning(
            f"Failed batches: {micro_failures} micro, {macro_failures} macro"
        )
        # TODO: Phase 2 - Store detailed failed chunk info for granular retry

    return {
        "status": status,
        "chunks_upserted": total_upserted,
        "chunks_passed": total_chunks,
        "failed_chunks": {
            "micro_batches": micro_failures,
            "macro_batches": macro_failures,
        },
    }
