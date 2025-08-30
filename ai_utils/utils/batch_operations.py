"""
Batch operation utilities for efficient processing of large datasets.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..config import get_config

T = TypeVar("T")
R = TypeVar("R")


async def batch_process(
    items: list[T],
    processor: Callable[[T], Awaitable[R]],
    batch_size: int | None = None,
    max_concurrent: int | None = None,
) -> list[R]:
    """
    Process items in batches with controlled concurrency.

    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Size of each batch (default from config)
        max_concurrent: Maximum concurrent operations (default from config)

    Returns:
        List of processed results
    """
    if not items:
        return []

    config = get_config()
    batch_size = batch_size or config.batch_size
    max_concurrent = max_concurrent or config.max_concurrent_requests

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_item(item: T) -> R:
        async with semaphore:
            return await processor(item)

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Filter out exceptions
        for result in batch_results:
            if not isinstance(result, Exception):
                results.append(result)

    return results


async def batch_embed_texts(
    texts: list[str],
    embedder: Callable[[str], Awaitable[list[float]]],
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Embed multiple texts in batches.

    Args:
        texts: List of texts to embed
        embedder: Function to embed a single text
        batch_size: Size of each batch

    Returns:
        List of embedding vectors
    """
    return await batch_process(texts, embedder, batch_size)


async def batch_upsert_documents(
    documents: list[Any],
    upsert_func: Callable[[list[Any]], Awaitable[bool]],
    batch_size: int | None = None,
) -> bool:
    """
    Upsert documents in batches.

    Args:
        documents: List of documents to upsert
        upsert_func: Function to upsert a batch of documents
        batch_size: Size of each batch

    Returns:
        True if all batches succeeded
    """
    config = get_config()
    batch_size = batch_size or config.batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        success = await upsert_func(batch)
        if not success:
            return False

    return True
