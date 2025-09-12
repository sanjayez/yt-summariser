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
