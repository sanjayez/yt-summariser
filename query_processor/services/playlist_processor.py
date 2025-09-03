"""
Playlist Processing - Extract video URLs from YouTube playlists
"""

import re

import scrapetube

from telemetry.logging.logger import get_logger

logger = get_logger(__name__)


def extract_playlist_videos(playlist_url: str, max_videos: int = 50) -> list[str]:
    """Extract video URLs from YouTube playlist in order."""
    playlist_id = _extract_playlist_id(playlist_url)
    if not playlist_id:
        raise ValueError("Could not extract playlist ID from URL")

    logger.info(f"🔍 Fetching {max_videos} videos from playlist {playlist_id}")

    videos = scrapetube.get_playlist(playlist_id=playlist_id, limit=max_videos, sleep=1)
    video_urls = []

    for video_data in videos:
        video_id = video_data.get("videoId")
        if video_id:
            video_urls.append(f"https://youtube.com/watch?v={video_id}")

    logger.info(f"✅ Extracted {len(video_urls)} videos from playlist")
    return video_urls


def _extract_playlist_id(url: str) -> str | None:
    """Extract playlist ID from YouTube URL."""
    patterns = [r"[?&]list=([^&\n?#]+)", r"youtube\.com/playlist\?list=([^&\n?#]+)"]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
