"""
Decodo API service for YouTube transcript extraction
Replaces scrape.do and yt-dlp for reliable transcript extraction
"""

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class DecodoTranscriptService:
    """Service for extracting YouTube transcripts using Decodo API"""

    def __init__(self):
        self.api_url = "https://scraper-api.decodo.com/v2/scrape"

        # Get auth token from settings - REQUIRED for security
        self.auth_token = getattr(settings, "DECODO_AUTH_TOKEN", None)
        if not self.auth_token:
            raise ValueError(
                "DECODO_AUTH_TOKEN environment variable is required. "
                "Please set your Decodo API token in the environment variables."
            )

        self.timeout = 30

    def extract_transcript(self, video_id: str, language_code: str = "en") -> dict:
        """
        Extract transcript for a YouTube video using Decodo API

        Args:
            video_id: YouTube video ID (e.g., 'dQw4w9WgXcQ')
            language_code: Language code for transcript (default: 'en')

        Returns:
            Dict with success status, transcript segments, and metadata
        """
        logger.info(f"Extracting transcript for video {video_id} using Decodo API")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {self.auth_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "target": "youtube_transcript",
            "query": video_id,
            "language_code": language_code,
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                transcript_segments = self._parse_decodo_response(data)

                if transcript_segments:
                    # Create combined transcript text
                    transcript_text = " ".join(
                        [segment["text"] for segment in transcript_segments]
                    )

                    # Calculate duration
                    duration = (
                        transcript_segments[-1]["end"] if transcript_segments else 0
                    )

                    logger.info(
                        f"Successfully extracted {len(transcript_segments)} segments for video {video_id}"
                    )

                    return {
                        "success": True,
                        "transcript_text": transcript_text,
                        "segments": transcript_segments,
                        "language": language_code,
                        "duration": duration,
                        "segment_count": len(transcript_segments),
                    }
                else:
                    logger.warning(f"No transcript segments found for video {video_id}")
                    return {
                        "success": False,
                        "error": "No transcript segments found",
                        "transcript_text": "",
                        "segments": [],
                    }
            else:
                logger.error(
                    f"Decodo API error for video {video_id}: HTTP {response.status_code}"
                )
                return {
                    "success": False,
                    "error": f"API error: HTTP {response.status_code}",
                    "transcript_text": "",
                    "segments": [],
                }

        except requests.exceptions.Timeout:
            logger.error(f"Timeout extracting transcript for video {video_id}")
            return {
                "success": False,
                "error": "Request timeout",
                "transcript_text": "",
                "segments": [],
            }
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Request error extracting transcript for video {video_id}: {e}"
            )
            return {
                "success": False,
                "error": f"Request error: {str(e)}",
                "transcript_text": "",
                "segments": [],
            }
        except Exception as e:
            logger.error(
                f"Unexpected error extracting transcript for video {video_id}: {e}"
            )
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "transcript_text": "",
                "segments": [],
            }

    def _parse_decodo_response(self, data: dict) -> list[dict]:
        """
        Parse Decodo API response to extract transcript segments

        Args:
            data: Raw Decodo API response

        Returns:
            List of transcript segments with text, start, end, and duration
        """
        transcript_segments = []

        try:
            # Navigate to the content array
            results = data.get("results", [])
            if not results:
                logger.warning("No results found in Decodo response")
                return transcript_segments

            content = results[0].get("content", [])
            if not content:
                logger.warning("No content found in Decodo results")
                return transcript_segments

            # Extract transcript segments
            for item in content:
                if "transcriptSegmentRenderer" in item:
                    segment = item["transcriptSegmentRenderer"]

                    # Extract timing (convert from milliseconds to seconds)
                    start_ms = int(segment.get("startMs", 0))
                    end_ms = int(segment.get("endMs", start_ms + 1000))
                    start_seconds = start_ms / 1000.0
                    end_seconds = end_ms / 1000.0

                    # Extract text
                    snippet = segment.get("snippet", {})
                    runs = snippet.get("runs", [])

                    if runs and len(runs) > 0:
                        text = runs[0].get("text", "").strip()

                        # Skip empty text and music markers
                        if text and text != "[Music]":
                            transcript_segments.append(
                                {
                                    "text": text,
                                    "start": start_seconds,
                                    "end": end_seconds,
                                    "duration": end_seconds - start_seconds,
                                }
                            )

            logger.info(
                f"Parsed {len(transcript_segments)} transcript segments from Decodo response"
            )
            return transcript_segments

        except Exception as e:
            logger.error(f"Error parsing Decodo response: {e}")
            return transcript_segments

    def extract_metadata(self, video_id: str) -> dict:
        """
        Extract metadata for a YouTube video using Decodo API

        Args:
            video_id: YouTube video ID (e.g., 'dQw4w9WgXcQ')

        Returns:
            Dict with success status and metadata
        """
        logger.info(f"Extracting metadata for video {video_id} using Decodo API")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {self.auth_token}",
            "Content-Type": "application/json",
        }

        payload = {"target": "youtube_metadata", "query": video_id}

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()

                if "results" in data and data["results"]:
                    results = data["results"]
                    # Handle case where results might be a list
                    if isinstance(results, list):
                        if len(results) > 0:
                            results = results[0]  # Take the first result
                        else:
                            logger.warning(f"Empty results list for video {video_id}")
                            return {
                                "success": False,
                                "metadata": None,
                                "error": "Empty results list",
                            }

                    # Extract the actual metadata from nested structure
                    if (
                        isinstance(results, dict)
                        and "content" in results
                        and "results" in results["content"]
                    ):
                        metadata = results["content"]["results"]
                        logger.info(
                            f"Successfully extracted metadata for video {video_id}"
                        )
                        return {"success": True, "metadata": metadata, "error": None}
                    else:
                        logger.warning(
                            f"Unexpected Decodo response structure for video {video_id}"
                        )
                        return {
                            "success": False,
                            "metadata": None,
                            "error": "Unexpected response structure",
                        }
                else:
                    logger.warning(f"No metadata found for video {video_id}")
                    return {
                        "success": False,
                        "metadata": None,
                        "error": "No metadata found in response",
                    }
            else:
                logger.error(
                    f"Decodo API error for video {video_id}: HTTP {response.status_code}"
                )
                return {
                    "success": False,
                    "metadata": None,
                    "error": f"API error: HTTP {response.status_code}",
                }

        except requests.exceptions.Timeout:
            logger.error(f"Timeout extracting metadata for video {video_id}")
            return {"success": False, "metadata": None, "error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error extracting metadata for video {video_id}: {e}")
            return {
                "success": False,
                "metadata": None,
                "error": f"Request error: {str(e)}",
            }
        except Exception as e:
            logger.error(
                f"Unexpected error extracting metadata for video {video_id}: {e}"
            )
            return {
                "success": False,
                "metadata": None,
                "error": f"Unexpected error: {str(e)}",
            }


# Global service instance
try:
    decodo_service = DecodoTranscriptService()
    logger.info("Decodo service initialized successfully")
except ValueError as e:
    logger.warning(f"Decodo service not available: {e}")
    decodo_service = None


def extract_youtube_transcript(
    video_id: str, language_code: str = "en"
) -> tuple[bool, dict]:
    """
    Extract transcript for a YouTube video using Decodo API

    Args:
        video_id: YouTube video ID
        language_code: Language code for transcript

    Returns:
        Tuple of (success: bool, result: Dict)
    """
    result = decodo_service.extract_transcript(video_id, language_code)
    return result["success"], result
