from __future__ import annotations

import base64
import logging

import httpx

from ..models import FetchedPage, SearchResult, SourceType

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"


class GithubSource:
    """GitHub REST API search and content fetching."""

    source_type = SourceType.GITHUB

    def _headers(self, token: str = "") -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if token:
            h["Authorization"] = f"token {token}"
        return h

    async def search(self, queries: list[str], token: str = "") -> list[SearchResult]:
        results: list[SearchResult] = []
        headers = self._headers(token)
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            for query in queries:
                # Search repositories
                try:
                    resp = await client.get(
                        f"{API_BASE}/search/repositories",
                        params={"q": query, "per_page": 10},
                    )
                    resp.raise_for_status()
                    for item in resp.json().get("items", []):
                        results.append(
                            SearchResult(
                                url=item.get("html_url", ""),
                                title=item.get("full_name", ""),
                                snippet=item.get("description", "") or "",
                                source_type=SourceType.GITHUB,
                                source_date=item.get("updated_at"),
                            )
                        )
                except Exception as e:
                    logger.warning("GitHub repo search failed for %r: %s", query, e)

                # Search code
                try:
                    resp = await client.get(
                        f"{API_BASE}/search/code",
                        params={"q": query, "per_page": 5},
                    )
                    resp.raise_for_status()
                    for item in resp.json().get("items", []):
                        results.append(
                            SearchResult(
                                url=item.get("html_url", ""),
                                title=item.get("path", ""),
                                snippet=item.get("repository", {}).get("description", "") or "",
                                source_type=SourceType.GITHUB,
                            )
                        )
                except Exception as e:
                    logger.warning("GitHub code search failed for %r: %s", query, e)
        return results

    async def fetch(self, url: str, token: str = "") -> FetchedPage:
        headers = self._headers(token)
        try:
            async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
                # Detect if this is a repo root or a file URL
                # github.com/{owner}/{repo}/blob/{branch}/{path}
                parts = url.replace("https://github.com/", "").split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]

                    if len(parts) >= 4 and parts[2] == "blob":
                        # File URL — fetch file content
                        branch = parts[3]
                        file_path = "/".join(parts[4:])
                        resp = await client.get(
                            f"{API_BASE}/repos/{owner}/{repo}/contents/{file_path}",
                            params={"ref": branch},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        content = ""
                        if data.get("encoding") == "base64" and data.get("content"):
                            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                        return FetchedPage(
                            url=url,
                            title=data.get("path", file_path),
                            text=content,
                            source_type=SourceType.GITHUB,
                        )
                    else:
                        # Repo root — fetch README
                        resp = await client.get(
                            f"{API_BASE}/repos/{owner}/{repo}/readme",
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        content = ""
                        if data.get("encoding") == "base64" and data.get("content"):
                            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                        return FetchedPage(
                            url=url,
                            title=f"{owner}/{repo} README",
                            text=content,
                            source_type=SourceType.GITHUB,
                        )

            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.GITHUB,
                fetch_error="Could not parse GitHub URL",
            )
        except Exception as e:
            logger.warning("GitHub fetch failed for %s: %s", url, e)
            return FetchedPage(
                url=url,
                title="",
                text="",
                source_type=SourceType.GITHUB,
                fetch_error=str(e),
            )
