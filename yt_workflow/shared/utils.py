import re
from urllib.parse import parse_qs, urlparse


def extract_video_id_from_url(video_url: str) -> str | None:
    """Extract YouTube video ID from URL"""
    try:
        # Handle youtu.be short URLs
        if "youtu.be/" in video_url:
            return video_url.split("youtu.be/")[-1].split("?")[0].split("&")[0]

        # Handle standard YouTube URLs
        parsed_url = urlparse(video_url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
            if "watch" in parsed_url.path:
                query_params = parse_qs(parsed_url.query)
                return query_params.get("v", [None])[0]
            elif "embed" in parsed_url.path:
                return parsed_url.path.split("embed/")[-1].split("?")[0]

        # Fallback regex
        video_id_pattern = r"[a-zA-Z0-9_-]{11}"
        matches = re.findall(video_id_pattern, video_url)
        return matches[0] if matches else None

    except Exception:
        return None
