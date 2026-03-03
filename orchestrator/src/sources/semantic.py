from __future__ import annotations

import logging
import re

import httpx

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

API_BASE = "https://api.semanticscholar.org/graph/v1"

# Match Semantic Scholar paper IDs (40-char hex or arxiv:XXXX.XXXXX or DOI)
PAPER_ID_RE = re.compile(r"/paper/([a-f0-9]{40})")
S2_URL_RE = re.compile(r"semanticscholar\.org/paper/[^/]*/([a-f0-9]{40})")


class SemanticScholarSource:
    """Semantic Scholar API search and paper details."""

    source_type = SourceType.SEMANTIC_SCHOLAR

    async def search(self, queries: list[str]) -> list[SearchResult]:
        results: list[SearchResult] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        f"{API_BASE}/paper/search",
                        params={
                            "query": query,
                            "limit": 10,
                            "fields": "title,abstract,url,year,authors",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for paper in data.get("data", []):
                        authors = ", ".join(
                            a.get("name", "") for a in (paper.get("authors") or [])[:3]
                        )
                        snippet = paper.get("abstract", "") or ""
                        if authors:
                            snippet = f"[{authors}] {snippet[:250]}"
                        results.append(
                            SearchResult(
                                url=paper.get("url", ""),
                                title=paper.get("title", ""),
                                snippet=snippet[:300],
                                source_type=SourceType.SEMANTIC_SCHOLAR,
                                source_date=str(paper.get("year", "")) or None,
                            )
                        )
                except Exception as e:
                    logger.warning("Semantic Scholar search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        try:
            # Try to extract paper ID from URL
            paper_id = None
            match = S2_URL_RE.search(url)
            if match:
                paper_id = match.group(1)
            elif PAPER_ID_RE.search(url):
                paper_id = PAPER_ID_RE.search(url).group(1)

            if not paper_id:
                # Try using the URL as a lookup key
                paper_id = f"URL:{url}"

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{API_BASE}/paper/{paper_id}",
                    params={"fields": "title,abstract,tldr,citationCount,year,authors"},
                )
                resp.raise_for_status()
                data = resp.json()

            authors = ", ".join(a.get("name", "") for a in (data.get("authors") or [])[:20])
            tldr = data.get("tldr", {})
            tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""

            text_parts = [
                f"Title: {data.get('title', '')}",
                f"Authors: {authors}",
                f"Year: {data.get('year', 'N/A')}",
                f"Citations: {data.get('citationCount', 'N/A')}",
            ]
            if tldr_text:
                text_parts.append(f"\nTL;DR: {tldr_text}")
            abstract = data.get("abstract", "")
            if abstract:
                text_parts.append(f"\nAbstract:\n{abstract}")

            return FetchedPage(
                url=url,
                title=data.get("title", ""),
                text="\n".join(text_parts),
                source_type=SourceType.SEMANTIC_SCHOLAR,
                source_date=str(data.get("year", "")) or None,
            )
        except Exception as e:
            logger.warning("Semantic Scholar fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.SEMANTIC_SCHOLAR,
                fetch_error=str(e),
            )
