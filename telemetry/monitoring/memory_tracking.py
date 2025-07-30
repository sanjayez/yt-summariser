"""
Memory tracking and profiling utilities.

This module provides functions and context managers for tracking memory usage,
profiling memory allocations, and monitoring memory consumption patterns.
"""

import gc
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from ..logging import get_logger

# Default logger for memory tracking
logger = get_logger(__name__)


def log_memory_usage(
    label: str = "Current",
    include_gc_stats: bool = True,
    logger_instance: Optional[Any] = None
) -> Dict[str, float]:
    """
    Log current memory usage statistics.
    
    Args:
        label: Label for the memory log entry
        include_gc_stats: Whether to include garbage collection statistics
        logger_instance: Logger to use (defaults to module logger)
    
    Returns:
        Dictionary containing memory statistics in MB
    
    Example:
        >>> stats = log_memory_usage("After processing")
        >>> # Logs: "After processing Memory Usage: RSS=150.23MB, VMS=512.45MB, ..."
    """
    log = logger_instance or logger
    
    if not HAS_PSUTIL:
        log.warning("psutil not available, memory tracking disabled")
        return {}
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to MB
    stats = {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }
    
    # Add platform-specific metrics if available
    if hasattr(memory_info, 'shared'):
        stats['shared_mb'] = memory_info.shared / 1024 / 1024
    if hasattr(memory_info, 'pss'):
        stats['pss_mb'] = memory_info.pss / 1024 / 1024
    
    log.info(
        f"{label} Memory Usage: "
        f"RSS={stats['rss_mb']:.2f}MB, "
        f"VMS={stats['vms_mb']:.2f}MB, "
        f"Percent={stats['percent']:.1f}%, "
        f"Available={stats['available_mb']:.2f}MB"
    )
    
    if include_gc_stats and gc.isenabled():
        gc_stats = gc.get_stats()
        if gc_stats:
            # Log generation statistics
            for i, gen_stats in enumerate(gc_stats[:3]):  # Only first 3 generations
                log.debug(
                    f"  GC Generation {i}: "
                    f"collections={gen_stats.get('collections', 0)}, "
                    f"collected={gen_stats.get('collected', 0)}, "
                    f"uncollectable={gen_stats.get('uncollectable', 0)}"
                )
    
    return stats


@contextmanager
def memory_tracker(name: str, logger_instance: Optional[Any] = None):
    """
    Context manager for tracking memory usage within a code block.
    
    Args:
        name: Name for the tracked operation
        logger_instance: Logger to use (defaults to module logger)
    
    Yields:
        Dictionary that will contain memory statistics after the block
    
    Example:
        >>> with memory_tracker("data_processing") as mem_stats:
        ...     # Process large dataset
        ...     processed_data = transform_data(raw_data)
        ... print(f"Memory used: {mem_stats['memory_change_mb']}MB")
    """
    log = logger_instance or logger
    
    if not HAS_PSUTIL:
        log.warning("psutil not available, memory tracking disabled")
        empty_stats = {}
        yield empty_stats
        return
    
    process = psutil.Process(os.getpid())
    
    # Initial measurements
    gc.collect()  # Clean up before measuring
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    # Stats dictionary to be populated
    stats = {}
    
    try:
        yield stats
    finally:
        # Final measurements
        gc.collect()  # Clean up before final measurement
        end_memory = process.memory_info().rss / 1024 / 1024
        end_time = time.time()
        
        # Populate statistics
        stats.update({
            'name': name,
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_change_mb': end_memory - start_memory,
            'duration_seconds': end_time - start_time,
            'memory_percent': process.memory_percent()
        })
        
        log.info(
            f"Memory tracker '{name}': "
            f"Used {stats['memory_change_mb']:+.2f}MB in {stats['duration_seconds']:.3f}s "
            f"(Current: {stats['end_memory_mb']:.2f}MB, {stats['memory_percent']:.1f}%)"
        )