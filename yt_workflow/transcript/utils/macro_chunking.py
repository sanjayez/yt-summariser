"""Macro chunk generation without overlap for transcript processing"""

from yt_workflow.transcript.types import MacroChunk, NormalizedLine

from .constants import MACRO_MAX, MACRO_TARGET, SENTENCE_ENDINGS


def build_macro_chunks(lines: list[NormalizedLine], video_id: str) -> list[MacroChunk]:
    """Build macro chunks without overlap. O(n) sliding window algorithm."""
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

            if current_duration > MACRO_MAX:
                if j > i:
                    j -= 1
                break

            if current_duration >= MACRO_TARGET and lines[j].text.strip().endswith(
                SENTENCE_ENDINGS
            ):
                break

            j += 1

        if j >= len(lines):
            j = len(lines) - 1

        # Create chunk
        chunk = MacroChunk(
            macro_id=f"{video_id}_macro_{chunk_index}",
            start=chunk_start_time,
            end=lines[j].end,
            text=" ".join(line.text for line in lines[chunk_start_idx : j + 1]),
        )
        chunks.append(chunk)
        chunk_index += 1

        # Move to next chunk (no overlap)
        i = j + 1

    return chunks
