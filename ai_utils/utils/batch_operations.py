"""
Batch operation utilities for efficient processing of large datasets.
"""

import asyncio
from typing import List, Callable, Any, TypeVar, Awaitable, Optional
from ..config import get_config

T = TypeVar('T')
R = TypeVar('R')

async def batch_process(
    items: List[T],
    processor: Callable[[T], Awaitable[R]],
    batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None
) -> List[R]:
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
        batch = items[i:i + batch_size]
        batch_tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Filter out exceptions
        for result in batch_results:
            if not isinstance(result, Exception):
                results.append(result)
    
    return results

async def batch_embed_texts(
    texts: List[str],
    embedder: Callable[[str], Awaitable[List[float]]],
    batch_size: Optional[int] = None
) -> List[List[float]]:
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
    documents: List[Any],
    upsert_func: Callable[[List[Any]], Awaitable[bool]],
    batch_size: Optional[int] = None
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
        batch = documents[i:i + batch_size]
        success = await upsert_func(batch)
        if not success:
            return False
    
    return True 