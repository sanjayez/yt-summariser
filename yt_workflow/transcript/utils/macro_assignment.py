"""Assign micro chunks to their primary macro chunks based on overlap"""

from yt_workflow.transcript.types import MacroChunk, MicroChunk

# Thresholds for assignment
PRIMARY_THRESHOLD = 0.6  # 60% overlap required for primary assignment
SECONDARY_THRESHOLD = 1.0  # 1 second minimum for secondary overlap hints


def calculate_overlap(micro: MicroChunk, macro: MacroChunk) -> float:
    """Calculate time overlap between micro and macro chunks"""
    overlap_start = max(micro.start, macro.start)
    overlap_end = min(micro.end, macro.end)
    overlap_duration = max(0, overlap_end - overlap_start)
    return overlap_duration


def assign_primary_macros(
    micro_chunks: list[MicroChunk], macro_chunks: list[MacroChunk]
) -> None:
    """
    Assign each micro chunk to its primary macro based on overlap percentage.
    Also tracks secondary overlaps for boundary query hints.

    Modifies micro_chunks in place by setting:
    - primary_macro_id: The macro with ≥60% overlap (or max overlap if multiple)
    - also_overlaps: Other macros with ≥1s overlap
    """
    for micro in micro_chunks:
        best_macro = None
        best_overlap_pct = 0.0
        secondary_overlaps = []

        micro_duration = micro.end - micro.start

        for macro in macro_chunks:
            # Quick boundary check to skip non-overlapping chunks
            if micro.end <= macro.start or micro.start >= macro.end:
                continue

            # Calculate overlap
            overlap_duration = calculate_overlap(micro, macro)
            overlap_pct = overlap_duration / micro_duration if micro_duration > 0 else 0

            # Always track the best overlap (regardless of threshold)
            if overlap_pct > best_overlap_pct:
                # Previous best becomes secondary if exists
                if best_macro and best_macro.macro_id not in secondary_overlaps:
                    secondary_overlaps.append(best_macro.macro_id)
                best_macro = macro
                best_overlap_pct = overlap_pct
            elif overlap_duration >= SECONDARY_THRESHOLD:
                # Secondary overlap (≥1s)
                if macro.macro_id not in secondary_overlaps:
                    secondary_overlaps.append(macro.macro_id)

        # Hybrid assignment: prefer ≥60% overlap, fallback to best available
        if best_overlap_pct >= PRIMARY_THRESHOLD:
            # Clean case: ≥60% overlap
            micro.primary_macro_id = best_macro.macro_id
        elif best_macro:
            # Fallback: assign to best available (prevents orphans)
            micro.primary_macro_id = best_macro.macro_id
            # Remove primary from secondary overlaps
            if best_macro.macro_id in secondary_overlaps:
                secondary_overlaps.remove(best_macro.macro_id)
        else:
            # No overlaps found
            micro.primary_macro_id = None

        micro.also_overlaps = secondary_overlaps
