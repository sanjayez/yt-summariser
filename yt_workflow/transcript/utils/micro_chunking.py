"""Micro chunk generation with overlap for transcript processing"""

from yt_workflow.transcript.types import MicroChunk, NormalizedLine

from .constants import MICRO_MAX, MICRO_OVERLAP, MICRO_TARGET, SENTENCE_ENDINGS


def build_micro_chunks(lines: list[NormalizedLine], video_id: str) -> list[MicroChunk]:
    """Build micro chunks with overlap. O(n) sliding window algorithm."""
    if not lines:
        return []

    chunks = []
    chunk_index = 0
    i = 0

    while i < len(lines):
        chunk_start_idx = i
        chunk_start_time = lines[i].start

        # Expand window to target/max duration or sentence boundary
        j = i
        while j < len(lines):
            current_duration = lines[j].end - chunk_start_time

            if current_duration > MICRO_MAX:
                if j > i:
                    j -= 1
                break

            if current_duration >= MICRO_TARGET and lines[j].text.strip().endswith(
                SENTENCE_ENDINGS
            ):
                break

            j += 1

        if j >= len(lines):
            j = len(lines) - 1

        # Create chunk
        chunk = MicroChunk(
            micro_id=f"{video_id}_micro_{chunk_index}",
            start=chunk_start_time,
            end=lines[j].end,
            text=" ".join(line.text for line in lines[chunk_start_idx : j + 1]),
        )
        chunks.append(chunk)
        chunk_index += 1

        # Calculate next start with overlap
        next_start_time = lines[j].end - MICRO_OVERLAP
        new_i = chunk_start_idx
        while new_i < len(lines) and lines[new_i].start < next_start_time:
            new_i += 1

        i = new_i if new_i > j else j + 1

    return chunks
