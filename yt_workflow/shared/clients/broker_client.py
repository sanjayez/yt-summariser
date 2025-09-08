import json
from typing import Any

import requests
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(total=3, backoff_factor=0.2, status_forcelist=(502, 503, 504))
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def _url(path: str) -> str:
    base = getattr(settings, "BROKER_BASE_URL", "http://localhost:3000").rstrip("/")
    return f"{base}{path}"


def _get(
    path: str, *, params: dict[str, Any] | None = None, timeout: int = 10
) -> dict[str, Any]:
    response = _session.get(_url(path), params=params, timeout=timeout)
    response.raise_for_status()
    # Avoids crashes if broker returns non-JSON on errors
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"status": response.status_code, "text": response.text}


def get_metadata(video_id: str) -> dict[str, Any]:
    return _get(f"/api/video/{video_id}/metadata")


def get_comments(video_id: str, *, limit: int = 20) -> dict[str, Any]:
    return _get(f"/api/video/{video_id}/comments", params={"limit": limit})


def get_transcript(video_id: str, *, lang: str = "en") -> dict[str, Any]:
    return _get(f"/api/video/{video_id}/transcript", params={"lang": lang})


def get_all(video_id: str) -> dict[str, Any]:
    return _get(f"/api/video/{video_id}/all", timeout=15)
