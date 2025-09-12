from yt_workflow.transcript.types import NormalizedLine


def _normalize_lines(raw_segments: list[dict], video_id: str) -> list[NormalizedLine]:
    """Normalize raw segments into Pydantic models with unique line IDs. 1-based idx."""
    lines = []

    for i, seg in enumerate(raw_segments):
        try:
            start = round(float(seg.get("start_time", 0.0)), 2)
            duration = round(float(seg.get("duration", 0.0)), 2)
        except Exception:
            start, duration = 0.0, 0.0

        end = round(start + duration, 2)
        if end < start:
            end = start

        # If this segment would overlap with the next one, truncate it
        if i < len(raw_segments) - 1:
            next_start = round(float(raw_segments[i + 1].get("start_time", 0.0)), 2)
            if end > next_start:
                end = next_start

        text = str(seg.get("text", "")).strip()

        line = NormalizedLine(
            line_id=f"{video_id}_line_{i + 1}",
            idx=i + 1,
            start=start,
            end=end,
            text=text,
        )

        lines.append(line)

    return lines
