from __future__ import annotations

import asyncio
import logging
from functools import partial

import httpx
from huggingface_hub import HfApi

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)


class HuggingFaceSource:
    """HuggingFace Hub API search and model card fetching."""

    source_type = SourceType.HUGGINGFACE

    async def search(self, queries: list[str], token: str = "") -> list[SearchResult]:
        results: list[SearchResult] = []
        loop = asyncio.get_running_loop()
        api = HfApi(token=token or None)

        for query in queries:
            try:
                models = await loop.run_in_executor(
                    None,
                    partial(api.list_models, search=query, limit=10),
                )
                for model in models:
                    model_id = model.modelId if hasattr(model, "modelId") else str(model.id)
                    results.append(
                        SearchResult(
                            url=f"https://huggingface.co/{model_id}",
                            title=model_id,
                            snippet=getattr(model, "pipeline_tag", "") or "",
                            source_type=SourceType.HUGGINGFACE,
                            source_date=getattr(model, "lastModified", None),
                        )
                    )
            except Exception as e:
                logger.warning("HuggingFace search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str, token: str = "") -> FetchedPage:
        try:
            # Extract model/dataset ID from URL: huggingface.co/{owner}/{name}
            path = url.replace("https://huggingface.co/", "").strip("/")
            parts = path.split("/")
            if len(parts) < 2:
                return FetchedPage(
                    url=url,
                    title="",
                    text="",
                    source_type=SourceType.HUGGINGFACE,
                    fetch_error="Could not parse HuggingFace model ID from URL",
                )

            model_id = f"{parts[0]}/{parts[1]}"
            # Fetch README.md via the HF API
            readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
                resp = await client.get(readme_url)
                resp.raise_for_status()
                text = resp.text

            return FetchedPage(
                url=url,
                title=model_id,
                text=text,
                source_type=SourceType.HUGGINGFACE,
            )
        except Exception as e:
            logger.warning("HuggingFace fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.HUGGINGFACE,
                fetch_error=str(e),
            )
