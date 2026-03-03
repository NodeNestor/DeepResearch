from __future__ import annotations

import logging
import re

import httpx

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

API_BASE = "https://en.wikipedia.org/w/api.php"
REST_BASE = "https://en.wikipedia.org/api/rest_v1"

# Simple HTML tag stripper for article text
TAG_RE = re.compile(r"<[^>]+>")


class WikipediaSource:
    """Wikipedia API search and article extraction."""

    source_type = SourceType.WIKIPEDIA

    async def search(self, queries: list[str]) -> list[SearchResult]:
        results: list[SearchResult] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        API_BASE,
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "srlimit": 10,
                            "format": "json",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get("query", {}).get("search", []):
                        title = item.get("title", "")
                        snippet = TAG_RE.sub("", item.get("snippet", ""))
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        results.append(
                            SearchResult(
                                url=url,
                                title=title,
                                snippet=snippet,
                                source_type=SourceType.WIKIPEDIA,
                                source_date=item.get("timestamp"),
                            )
                        )
                except Exception as e:
                    logger.warning("Wikipedia search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        try:
            # Extract title from URL
            title = url.split("/wiki/")[-1] if "/wiki/" in url else url
            title = title.replace("_", " ")

            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                # Use the TextExtracts API for clean plaintext
                resp = await client.get(
                    API_BASE,
                    params={
                        "action": "query",
                        "titles": title,
                        "prop": "extracts",
                        "explaintext": "1",
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                page = next(iter(pages.values()))
                text = page.get("extract", "")
                page_title = page.get("title", title)

            return FetchedPage(
                url=url,
                title=page_title,
                text=text,
                source_type=SourceType.WIKIPEDIA,
            )
        except Exception as e:
            logger.warning("Wikipedia fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.WIKIPEDIA,
                fetch_error=str(e),
            )
