"""Batch processing utilities for vector operations"""

from typing import Any


def create_batch_groups(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split items into batches of specified size"""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
