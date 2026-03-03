from __future__ import annotations

import logging

import httpx

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "DeepResearch/1.0 (research bot)"}


class RedditSource:
    """Reddit search via SearXNG + .json endpoint for content."""

    source_type = SourceType.REDDIT

    async def search(self, queries: list[str], searxng_url: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        f"{searxng_url}/search",
                        params={"q": query, "format": "json", "engines": "reddit"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data.get("results", []):
                        results.append(
                            SearchResult(
                                url=r.get("url", ""),
                                title=r.get("title", ""),
                                snippet=r.get("content", ""),
                                source_type=SourceType.REDDIT,
                                source_date=r.get("publishedDate"),
                            )
                        )
                except Exception as e:
                    logger.warning("Reddit search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        try:
            # Normalise URL: strip trailing slash, append .json
            clean_url = url.rstrip("/")
            if not clean_url.endswith(".json"):
                clean_url += ".json"

            async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=HEADERS) as client:
                resp = await client.get(clean_url)
                resp.raise_for_status()
                data = resp.json()

            # Parse Reddit JSON structure
            post_data = data[0]["data"]["children"][0]["data"]
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")

            # Extract top comments
            comments: list[str] = []
            if len(data) > 1:
                for child in data[1]["data"]["children"][:20]:
                    if child.get("kind") == "t1":
                        body = child["data"].get("body", "")
                        if body:
                            comments.append(body)

            text_parts = [f"Title: {title}"]
            if selftext:
                text_parts.append(f"\n{selftext}")
            if comments:
                text_parts.append("\n--- Top Comments ---")
                for i, c in enumerate(comments, 1):
                    text_parts.append(f"\n[{i}] {c}")

            return FetchedPage(
                url=url,
                title=title,
                text="\n".join(text_parts),
                source_type=SourceType.REDDIT,
            )
        except Exception as e:
            logger.warning("Reddit fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.REDDIT,
                fetch_error=str(e),
            )
