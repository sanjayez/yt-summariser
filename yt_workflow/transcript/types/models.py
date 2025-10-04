"""Pydantic models for transcript processing"""

from pydantic import BaseModel


class NormalizedLine(BaseModel):
    """Normalized transcript line after processing"""

    line_id: str  # Format: video_id_line_{idx}
    idx: int
    start: float
    end: float
    text: str


class MicroChunk(BaseModel):
    """Micro-level transcript chunk with overlap"""

    micro_id: str  # Format: video_id_micro_{index}
    start: float
    end: float
    text: str
    primary_macro_id: str | None = None  # Primary macro assignment
    also_overlaps: list[str] = []  # Secondary macro overlaps for boundary queries


class MacroChunk(BaseModel):
    """Macro-level transcript chunk without overlap"""

    macro_id: str  # Format: video_id_macro_{index}
    start: float
    end: float
    text: str


class Chapter(BaseModel):
    """Detected chapter with verbatim boundaries"""

    chapter: str
    content_type: str
    start_string: str
    end_string: str


class ChapterDetectionOutput(BaseModel):
    """Output from chapter detection containing all detected chapters"""

    chapters: list[Chapter]
    method: str = "verbatim_boundaries"


class ChapterSummary(BaseModel):
    """Summary of a single chapter with bullet points"""

    title: str
    bullet_points: list[str]  # 3-4 bullet points without "•" prefixes


class ExecutiveSummary(BaseModel):
    """Executive summary with key highlights"""

    executive_summary: str  # 2-3 sentences
    key_highlights: list[str]  # 2-3 key points
