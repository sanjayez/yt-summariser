"""
Text processing utilities for AI operations.
Functions for cleaning, chunking, and normalizing text.
"""

import re

from ..config import get_config


def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove special characters that might interfere with embedding
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", "", text)

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    return text


def chunk_text(
    text: str, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[str]:
    """
    Split text into chunks for processing.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    config = get_config()
    chunk_size = chunk_size or config.llamaindex.chunk_size
    chunk_overlap = chunk_overlap or config.llamaindex.chunk_overlap

    # Simple word-based chunking
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.

    Args:
        text: Text to extract sentences from

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Simple sentence splitting
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def remove_duplicates(texts: list[str]) -> list[str]:
    """
    Remove duplicate texts while preserving order.

    Args:
        texts: List of texts

    Returns:
        List with duplicates removed
    """
    if not texts:
        return []

    seen = set()
    result = []

    for text in texts:
        normalized = normalize_text(text)
        if normalized not in seen:
            seen.add(normalized)
            result.append(text)

    return result
