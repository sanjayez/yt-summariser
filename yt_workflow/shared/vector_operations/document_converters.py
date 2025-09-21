"""Document conversion utilities for transforming chunks to vector documents"""

from ai_utils.models import VectorDocument
from yt_workflow.transcript.types.models import MacroChunk, MicroChunk


def micro_to_vector_document(micro: MicroChunk, video_id: str) -> VectorDocument:
    """Convert MicroChunk to VectorDocument format"""
    return VectorDocument(
        id=micro.micro_id,
        text=micro.text,
        embedding=None,  # Weaviate handles natively
        metadata={
            "video_id": video_id,
            "type": "micro",
            "start_time": micro.start,
            "end_time": micro.end,
            "primary_macro_id": micro.primary_macro_id,
            "also_overlaps": micro.also_overlaps,
        },
    )


def macro_to_vector_document(macro: MacroChunk, video_id: str) -> VectorDocument:
    """Convert MacroChunk to VectorDocument format"""
    return VectorDocument(
        id=macro.macro_id,
        text=macro.text,
        embedding=None,  # Weaviate handles natively
        metadata={
            "video_id": video_id,
            "type": "macro",
            "start_time": macro.start,
            "end_time": macro.end,
        },
    )
