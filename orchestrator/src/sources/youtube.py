from __future__ import annotations

import asyncio
import logging
import re
from functools import partial

from youtube_transcript_api import YouTubeTranscriptApi

import httpx

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


class YoutubeSource:
    """YouTube search via SearXNG + transcript extraction."""

    source_type = SourceType.YOUTUBE

    async def search(self, queries: list[str], searxng_url: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        f"{searxng_url}/search",
                        params={"q": query, "format": "json", "engines": "youtube"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data.get("results", []):
                        results.append(
                            SearchResult(
                                url=r.get("url", ""),
                                title=r.get("title", ""),
                                snippet=r.get("content", ""),
                                source_type=SourceType.YOUTUBE,
                                source_date=r.get("publishedDate"),
                            )
                        )
                except Exception as e:
                    logger.warning("YouTube search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        loop = asyncio.get_running_loop()
        try:
            match = VIDEO_ID_RE.search(url)
            if not match:
                return FetchedPage(
                    url=url,
                    title="",
                    text="",
                    source_type=SourceType.YOUTUBE,
                    fetch_error="Could not extract video ID from URL",
                )
            video_id = match.group(1)

            def _get_transcript() -> list[dict]:
                return YouTubeTranscriptApi.get_transcript(video_id)

            segments = await loop.run_in_executor(None, _get_transcript)
            transcript_text = " ".join(seg["text"] for seg in segments)

            return FetchedPage(
                url=url,
                title=f"YouTube transcript: {video_id}",
                text=transcript_text,
                source_type=SourceType.YOUTUBE,
            )
        except Exception as e:
            logger.warning("YouTube fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.YOUTUBE,
                fetch_error=str(e),
            )
