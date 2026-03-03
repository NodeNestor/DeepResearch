from __future__ import annotations

import logging

import httpx
import trafilatura

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)


class WebSource:
    """SearXNG web search + trafilatura text extraction."""

    source_type = SourceType.WEB

    async def search(self, queries: list[str], searxng_url: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        f"{searxng_url}/search",
                        params={"q": query, "format": "json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data.get("results", []):
                        results.append(
                            SearchResult(
                                url=r.get("url", ""),
                                title=r.get("title", ""),
                                snippet=r.get("content", ""),
                                source_type=SourceType.WEB,
                                source_date=r.get("publishedDate"),
                            )
                        )
                except Exception as e:
                    logger.warning("SearXNG search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
            text = trafilatura.extract(html) or ""
            title = ""
            if "<title>" in html.lower():
                start = html.lower().index("<title>") + 7
                end = html.lower().index("</title>", start)
                title = html[start:end].strip()
            return FetchedPage(
                url=url,
                title=title,
                text=text,
                source_type=SourceType.WEB,
            )
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.WEB,
                fetch_error=str(e),
            )
