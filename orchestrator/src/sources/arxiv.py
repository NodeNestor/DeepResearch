from __future__ import annotations

import asyncio
import logging
import re
from functools import partial

import arxiv

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


class ArxivSource:
    """ArXiv API search and abstract extraction."""

    source_type = SourceType.ARXIV

    async def search(self, queries: list[str]) -> list[SearchResult]:
        results: list[SearchResult] = []
        loop = asyncio.get_running_loop()
        for query in queries:
            try:
                search_obj = arxiv.Search(
                    query=query,
                    max_results=10,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                client = arxiv.Client()
                papers = await loop.run_in_executor(
                    None, partial(list, client.results(search_obj))
                )
                for paper in papers:
                    results.append(
                        SearchResult(
                            url=paper.entry_id,
                            title=paper.title,
                            snippet=paper.summary[:300] if paper.summary else "",
                            source_type=SourceType.ARXIV,
                            source_date=paper.published.isoformat() if paper.published else None,
                        )
                    )
            except Exception as e:
                logger.warning("ArXiv search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str) -> FetchedPage:
        loop = asyncio.get_running_loop()
        try:
            # Extract arxiv ID from URL
            match = ARXIV_ID_RE.search(url)
            if match:
                arxiv_id = match.group(1)
                search_obj = arxiv.Search(id_list=[arxiv_id])
                client = arxiv.Client()
                papers = await loop.run_in_executor(
                    None, partial(list, client.results(search_obj))
                )
                if papers:
                    paper = papers[0]
                    authors = ", ".join(a.name for a in paper.authors[:10])
                    text = (
                        f"Title: {paper.title}\n"
                        f"Authors: {authors}\n"
                        f"Published: {paper.published}\n"
                        f"Categories: {', '.join(paper.categories)}\n\n"
                        f"Abstract:\n{paper.summary}"
                    )
                    return FetchedPage(
                        url=url,
                        title=paper.title,
                        text=text,
                        source_type=SourceType.ARXIV,
                        source_date=paper.published.isoformat() if paper.published else None,
                    )
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.ARXIV,
                fetch_error="Could not parse arxiv ID from URL",
            )
        except Exception as e:
            logger.warning("ArXiv fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.ARXIV,
                fetch_error=str(e),
            )
