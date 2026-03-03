from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import Settings
from ..models import FetchedPage, SearchResult, SourceType

from .arxiv import ArxivSource
from .github import GithubSource
from .huggingface import HuggingFaceSource
from .reddit import RedditSource
from .semantic import SemanticScholarSource
from .web import WebSource
from .wikipedia import WikipediaSource
from .youtube import YoutubeSource

logger = logging.getLogger(__name__)

# Source registry — one instance per source type
SOURCE_REGISTRY: dict[SourceType, Any] = {
    SourceType.WEB: WebSource(),
    SourceType.ARXIV: ArxivSource(),
    SourceType.REDDIT: RedditSource(),
    SourceType.YOUTUBE: YoutubeSource(),
    SourceType.GITHUB: GithubSource(),
    SourceType.SEMANTIC_SCHOLAR: SemanticScholarSource(),
    SourceType.HUGGINGFACE: HuggingFaceSource(),
    SourceType.WIKIPEDIA: WikipediaSource(),
}

# Sources that use SearXNG for search
_SEARXNG_SOURCES = {SourceType.WEB, SourceType.REDDIT, SourceType.YOUTUBE}
# Sources that use API tokens
_TOKEN_SOURCES = {SourceType.GITHUB: "github_token", SourceType.HUGGINGFACE: "hf_token"}


async def search_all(
    queries_by_source: dict[SourceType, list[str]],
    config: Settings,
) -> list[SearchResult]:
    """Run all source searches in parallel, deduplicate by URL."""

    async def _search_one(source_type: SourceType, queries: list[str]) -> list[SearchResult]:
        source = SOURCE_REGISTRY[source_type]
        try:
            if source_type in _SEARXNG_SOURCES:
                return await source.search(queries, config.searxng_url)
            elif source_type in _TOKEN_SOURCES:
                token = getattr(config, _TOKEN_SOURCES[source_type], "")
                return await source.search(queries, token=token)
            else:
                return await source.search(queries)
        except Exception as e:
            logger.error("Search failed for source %s: %s", source_type, e)
            return []

    tasks = [
        _search_one(st, qs) for st, qs in queries_by_source.items() if st in SOURCE_REGISTRY
    ]
    all_results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[SearchResult] = []
    for results in all_results:
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)

    return deduped


async def fetch_all(
    results: list[SearchResult],
    config: Settings,
) -> list[FetchedPage]:
    """Fetch all URLs in parallel with concurrency limit."""
    sem = asyncio.Semaphore(config.max_concurrent_fetches)

    async def _fetch_one(result: SearchResult) -> FetchedPage:
        async with sem:
            source = SOURCE_REGISTRY.get(result.source_type, SOURCE_REGISTRY[SourceType.WEB])
            try:
                if result.source_type in _TOKEN_SOURCES:
                    token = getattr(config, _TOKEN_SOURCES[result.source_type], "")
                    return await source.fetch(result.url, token=token)
                else:
                    return await source.fetch(result.url)
            except Exception as e:
                logger.error("Fetch failed for %s: %s", result.url, e)
                return FetchedPage(
                    url=result.url,
                    title=result.title,
                    text="",
                    source_type=result.source_type,
                    fetch_error=str(e),
                )

    pages = await asyncio.gather(*[_fetch_one(r) for r in results])
    return list(pages)
