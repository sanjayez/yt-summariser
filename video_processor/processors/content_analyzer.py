"""
Video Content Analysis Utility Functions
Utility functions for chunking, LLM processing, vector search, and ratio calculations.
Used by the two-phase content analysis system (preliminary + finalization).
"""

import asyncio
import json
import re
from collections import defaultdict

from ai_utils.config import get_config
from ai_utils.providers.weaviate_store import WeaviateVectorStoreProvider
from ai_utils.services.registry import get_gemini_llm_service
from ai_utils.services.vector_service import VectorService
from telemetry import get_logger

logger = get_logger(__name__)

# Supported tone values that match database schema
SUPPORTED_TONES = [
    "formal",
    "informal",
    "positive",
    "negative",
    "neutral",
    "humorous",
    "serious",
]


def chunk_transcript(
    transcript_text: str, max_chunk_size: int = 2000, overlap_sentences: int = 2
) -> list[dict]:
    """
    Chunk transcript by sentences with overlap for context preservation.

    Args:
        transcript_text: Full transcript text to chunk
        max_chunk_size: Maximum words per chunk
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of chunk dictionaries with text and metadata
    """
    try:
        # Split into sentences
        sentences = re.split(r"[.!?]+", transcript_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [{"text": transcript_text, "index": 0}]

        chunks = []
        current_chunk = []
        current_length = 0

        for _i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())

            # If adding this sentence would exceed chunk size and we have content
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ". ".join(current_chunk) + "."
                chunks.append(
                    {
                        "text": chunk_text,
                        "index": len(chunks),
                        "word_count": current_length,
                    }
                )

                # Start new chunk with overlap
                if len(current_chunk) > overlap_sentences:
                    overlap_chunk = current_chunk[-overlap_sentences:]
                    current_chunk = overlap_chunk
                    current_length = sum(len(s.split()) for s in overlap_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ". ".join(current_chunk) + "."
            chunks.append(
                {"text": chunk_text, "index": len(chunks), "word_count": current_length}
            )

        logger.info(
            f"Created {len(chunks)} chunks with {overlap_sentences}-sentence overlap"
        )
        return chunks

    except Exception as e:
        logger.error(f"Error in chunking: {e}")
        return [{"text": transcript_text, "index": 0}]


async def process_with_gemini_llm_async(prompt: str, max_retries: int = 2) -> dict:
    """
    Process prompt with Gemini LLM asynchronously.

    Args:
        prompt: The prompt to send to LLM
        max_retries: Number of retry attempts

    Returns:
        Parsed JSON response from LLM
    """
    try:
        config = get_config()
        llm_service = get_gemini_llm_service()

        for attempt in range(max_retries + 1):
            try:
                # Direct async call with timeout
                response_text = await asyncio.wait_for(
                    llm_service.generate_text(
                        prompt=prompt,
                        temperature=config.gemini.temperature,
                        max_tokens=config.gemini.max_tokens,
                    ),
                    timeout=120,  # 2 minute timeout
                )

                actual_text = response_text.get("text", "")

                if not actual_text or not actual_text.strip():
                    raise ValueError("Empty response from LLM")

                # Try to parse as JSON
                try:
                    return json.loads(actual_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    json_match = re.search(r"\{.*\}", actual_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        logger.warning("Could not parse JSON from LLM response")
                        return {}

            except TimeoutError:
                logger.warning(f"LLM attempt {attempt + 1} timed out")
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"LLM attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                else:
                    raise

        return {}

    except Exception as e:
        logger.error(f"LLM processing failed: {e}")
        return {}


async def analyze_chunk_combined_async(chunk_text: str) -> dict:
    """
    Analyze a chunk for both content classification and tone in a single LLM call.

    Args:
        chunk_text: Text chunk to analyze

    Returns:
        Combined analysis results
    """
    prompt = f"""
Analyze this video transcript chunk for promotional content and speaker tone.

TASK 1 - CONTENT CLASSIFICATION:
Identify sections that are:
- EXPLICIT_AD: Sponsorships, product placements, discount codes, commercial breaks
- FILLER: Subscribe requests, like/comment prompts, channel promotion, social media plugs

TASK 2 - TONE ANALYSIS:
Classify the speaker's tone using ONLY these categories:
{", ".join(SUPPORTED_TONES)}

For each tone detected, provide a confidence score (0.0-1.0).

Transcript chunk:
{chunk_text}

Return JSON in this exact format:
{{
    "ad_segments": [
        {{"text": "exact text from transcript", "description": "one-line description"}}
    ],
    "filler_segments": [
        {{"text": "exact text from transcript", "description": "one-line description"}}
    ],
    "tone": {{
        "detected_tones": [
            {{"tone": "informal", "confidence": 0.9}},
            {{"tone": "positive", "confidence": 0.7}}
        ],
        "primary_tone": "informal",
        "evidence": ["Uses casual language", "Upbeat delivery"]
    }}
}}

If no ads/filler found, return empty arrays. Always include tone analysis.
Use ONLY the allowed tone categories: {", ".join(SUPPORTED_TONES)}
"""

    try:
        result = await process_with_gemini_llm_async(prompt)

        # Validate and clean tone values to ensure they match supported tones
        if isinstance(result, dict) and "tone" in result:
            tone_data = result["tone"]
            if "detected_tones" in tone_data:
                # Filter out any unsupported tones
                valid_tones = []
                for tone_entry in tone_data["detected_tones"]:
                    if (
                        isinstance(tone_entry, dict)
                        and tone_entry.get("tone", "").lower() in SUPPORTED_TONES
                    ):
                        tone_entry["tone"] = tone_entry["tone"].lower()
                        valid_tones.append(tone_entry)
                tone_data["detected_tones"] = valid_tones

            # Ensure primary tone is valid
            if "primary_tone" in tone_data:
                primary = tone_data["primary_tone"].lower()
                if primary not in SUPPORTED_TONES:
                    # Fallback to first valid tone or neutral
                    if tone_data.get("detected_tones"):
                        tone_data["primary_tone"] = tone_data["detected_tones"][0][
                            "tone"
                        ]
                    else:
                        tone_data["primary_tone"] = "neutral"
                else:
                    tone_data["primary_tone"] = primary

        return result

    except Exception as e:
        logger.error(f"Combined analysis failed: {e}")
        return {
            "ad_segments": [],
            "filler_segments": [],
            "tone": {
                "detected_tones": [{"tone": "neutral", "confidence": 1.0}],
                "primary_tone": "neutral",
                "evidence": [],
            },
        }


async def get_timestamps_for_text(
    text_excerpt: str, video_id: str
) -> tuple[float, float]:
    """
    Use vector search to find timestamps for a text excerpt.

    Args:
        text_excerpt: Text to find timestamps for
        video_id: Video ID to search within

    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    try:
        # Initialize vector service
        config = get_config()
        vector_store = WeaviateVectorStoreProvider(config=config)
        vector_service = VectorService(provider=vector_store)

        # Search for the text excerpt
        results = await vector_service.search_by_text(
            text=text_excerpt,
            top_k=5,
            filters={"video_id": video_id, "type": "segment"},
        )

        if results and results.results:
            # Use the best match
            best_match = results.results[0]
            metadata = best_match.metadata

            # Extract timestamps from metadata
            start_time = metadata.get("start_time", 0.0)
            end_time = metadata.get("end_time", 0.0)

            # If end_time is not available, try to calculate from duration
            if end_time == 0.0 and "duration" in metadata:
                duration = metadata.get("duration", 0.0)
                end_time = start_time + duration

            # Ensure end_time is at least start_time
            if end_time < start_time:
                end_time = start_time

            return start_time, end_time

        return 0.0, 0.0

    except Exception as e:
        logger.error(f"Vector search for timestamps failed: {e}")
        return 0.0, 0.0


async def process_chunks_parallel_async(chunks: list[dict]) -> list[dict]:
    """
    Process all chunks in parallel using asyncio.gather.

    Args:
        chunks: List of text chunks to process

    Returns:
        List of analysis results for each chunk
    """
    # Create async tasks for all chunks
    tasks = []
    for chunk in chunks:
        task = analyze_chunk_combined_async(chunk["text"])
        tasks.append(task)

    # Process all chunks in parallel
    try:
        # Use gather with return_exceptions=True to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {i} processing failed: {result}")
                # Add empty result for failed chunk
                processed_results.append(
                    {
                        "chunk_index": i,
                        "ad_segments": [],
                        "filler_segments": [],
                        "tone": {
                            "detected_tones": [{"tone": "neutral", "confidence": 1.0}],
                            "primary_tone": "neutral",
                            "evidence": [],
                        },
                    }
                )
            else:
                # Add chunk index to successful result
                result["chunk_index"] = i
                processed_results.append(result)

        return processed_results

    except Exception as e:
        logger.error(f"Parallel chunk processing failed: {e}")
        # Return empty results for all chunks on complete failure
        return [
            {
                "chunk_index": i,
                "ad_segments": [],
                "filler_segments": [],
                "tone": {
                    "detected_tones": [{"tone": "neutral", "confidence": 1.0}],
                    "primary_tone": "neutral",
                    "evidence": [],
                },
            }
            for i in range(len(chunks))
        ]


async def add_timestamps_to_segments(segments: list[dict], video_id: str) -> list[dict]:
    """
    Add timestamps to content segments using vector search.

    Args:
        segments: List of segments with text excerpts
        video_id: Video ID for filtering

    Returns:
        Segments with added timestamp information
    """
    timestamped_segments = []

    for _i, segment in enumerate(segments):
        text_excerpt = segment.get("text", "")
        description = segment.get("description", "")

        if text_excerpt:
            start_time, end_time = await get_timestamps_for_text(text_excerpt, video_id)

            timestamped_segments.append(
                {"start": start_time, "end": end_time, "desc": description}
            )

    return timestamped_segments


def aggregate_results(chunk_results: list[dict]) -> dict:
    """
    Aggregate and deduplicate results from all chunks.

    Args:
        chunk_results: List of results from each chunk

    Returns:
        Aggregated results with deduplicated segments and tones
    """
    all_ad_segments = []
    all_filler_segments = []
    tone_scores = defaultdict(lambda: {"total_confidence": 0, "count": 0})
    all_evidence = []

    for result in chunk_results:
        # Collect content segments
        all_ad_segments.extend(result.get("ad_segments", []))
        all_filler_segments.extend(result.get("filler_segments", []))

        # Aggregate tone data
        tone_data = result.get("tone", {})
        for tone_entry in tone_data.get("detected_tones", []):
            tone = tone_entry.get("tone", "")
            confidence = tone_entry.get("confidence", 0)
            if tone and confidence > 0:
                tone_scores[tone]["total_confidence"] += confidence
                tone_scores[tone]["count"] += 1

        all_evidence.extend(tone_data.get("evidence", []))

    # Deduplicate content segments (simple text-based deduplication)
    seen_ads = set()
    unique_ad_segments = []
    for seg in all_ad_segments:
        text_key = seg.get("text", "")[:100]  # Use first 100 chars as key
        if text_key and text_key not in seen_ads:
            seen_ads.add(text_key)
            unique_ad_segments.append(seg)

    seen_filler = set()
    unique_filler_segments = []
    for seg in all_filler_segments:
        text_key = seg.get("text", "")[:100]
        if text_key and text_key not in seen_filler:
            seen_filler.add(text_key)
            unique_filler_segments.append(seg)

    # Calculate final tone scores
    final_tones = []
    for tone, data in tone_scores.items():
        if data["count"] > 0:
            avg_confidence = data["total_confidence"] / data["count"]
            if avg_confidence > 0.5 or data["count"] > 1:
                final_tones.append(
                    {
                        "tone": tone,
                        "confidence": avg_confidence,
                        "support": data["count"],
                    }
                )

    # Sort by confidence and take top 3 tones
    final_tones.sort(key=lambda x: x["confidence"], reverse=True)
    speaker_tones = [t["tone"] for t in final_tones[:3]]

    if not speaker_tones:
        speaker_tones = ["neutral"]

    # Deduplicate evidence
    unique_evidence = list(set(all_evidence))[:5]

    return {
        "ad_segments": unique_ad_segments,
        "filler_segments": unique_filler_segments,
        "speaker_tones": speaker_tones,
        "primary_tone": speaker_tones[0] if speaker_tones else "neutral",
        "tone_evidence": unique_evidence,
    }


def calculate_content_ratios(timestamped_results: dict, video_duration: float) -> dict:
    """
    Calculate content quality ratios.

    Args:
        timestamped_results: Dictionary with ad and filler segments
        video_duration: Total video duration in seconds

    Returns:
        Dictionary with calculated ratios
    """
    try:
        if video_duration <= 0:
            return {
                "ad_duration_ratio": 0.0,
                "filler_duration_ratio": 0.0,
                "content_rating": 0.0,
            }

        # Calculate total duration for each type
        ad_duration = sum(
            max(0, seg["end"] - seg["start"])
            for seg in timestamped_results.get("ad_segments", [])
        )

        filler_duration = sum(
            max(0, seg["end"] - seg["start"])
            for seg in timestamped_results.get("filler_segments", [])
        )

        # Calculate ratios
        ad_ratio = min(1.0, ad_duration / video_duration)
        filler_ratio = min(1.0, filler_duration / video_duration)

        # Ensure total doesn't exceed 1.0
        total_classified = ad_ratio + filler_ratio
        if total_classified > 1.0:
            ad_ratio = ad_ratio / total_classified
            filler_ratio = filler_ratio / total_classified

        content_ratio = max(0.0, 1.0 - ad_ratio - filler_ratio)

        return {
            "ad_duration_ratio": round(ad_ratio, 4),
            "filler_duration_ratio": round(filler_ratio, 4),
            "content_rating": round(content_ratio, 4),
        }

    except Exception as e:
        logger.error(f"Error calculating ratios: {e}")
        return {
            "ad_duration_ratio": 0.0,
            "filler_duration_ratio": 0.0,
            "content_rating": 0.0,
        }
