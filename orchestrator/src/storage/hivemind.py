"""Async REST client for HiveMindDB."""

from __future__ import annotations

import logging

import httpx

from .models import (
    EntityCreate,
    EntityResponse,
    GraphTraverseNode,
    GraphTraverseResponse,
    MemoryCreate,
    MemoryResponse,
    MemoryUpdate,
    RelationCreate,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)


class HiveMindClient:
    """Async client for the HiveMindDB REST API.

    All methods return empty/default results on connection errors so the
    orchestrator can function even when HiveMindDB is not running yet.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """GET /health — returns True if HiveMindDB is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def create_memory(self, memory: MemoryCreate) -> MemoryResponse | None:
        """POST /api/v1/memories"""
        try:
            resp = await self._client.post(
                "/api/v1/memories",
                json=memory.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return MemoryResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            logger.warning("create_memory failed: %s", exc)
            return None

    async def get_memory(self, memory_id: int) -> MemoryResponse | None:
        """GET /api/v1/memories/{id}"""
        try:
            resp = await self._client.get(f"/api/v1/memories/{memory_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return MemoryResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            logger.warning("get_memory(%d) failed: %s", memory_id, exc)
            return None

    async def update_memory_metadata(
        self, memory_id: int, metadata: dict
    ) -> MemoryResponse | None:
        """PUT /api/v1/memories/{id} — partial update (metadata only)."""
        try:
            update = MemoryUpdate(metadata=metadata)
            resp = await self._client.put(
                f"/api/v1/memories/{memory_id}",
                json=update.model_dump(mode="json", exclude_none=True),
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return MemoryResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            logger.warning("update_memory_metadata(%d) failed: %s", memory_id, exc)
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 50,
        tags: list[str] | None = None,
        include_graph: bool = False,
    ) -> list[SearchResult]:
        """POST /api/v1/search — hybrid keyword + vector search."""
        try:
            req = SearchRequest(
                query=query,
                limit=limit,
                tags=tags or [],
                include_graph=include_graph,
            )
            resp = await self._client.post(
                "/api/v1/search",
                json=req.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return [SearchResult.model_validate(r) for r in resp.json()]
        except httpx.HTTPError as exc:
            logger.warning("search failed: %s", exc)
            return []

    async def get_memories_by_tag(
        self, tag: str, limit: int = 100
    ) -> list[SearchResult]:
        """Search memories filtered to a specific tag."""
        return await self.search(query="", limit=limit, tags=[tag])

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    async def create_entity(self, entity: EntityCreate) -> EntityResponse | None:
        """POST /api/v1/entities"""
        try:
            resp = await self._client.post(
                "/api/v1/entities",
                json=entity.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return EntityResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            logger.warning("create_entity failed: %s", exc)
            return None

    async def find_entity(self, name: str) -> EntityResponse | None:
        """POST /api/v1/entities/find — find entity by name."""
        try:
            resp = await self._client.post(
                "/api/v1/entities/find",
                json={"name": name},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return EntityResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            logger.warning("find_entity(%s) failed: %s", name, exc)
            return None

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    async def create_relation(self, relation: RelationCreate) -> dict | None:
        """POST /api/v1/relationships"""
        try:
            resp = await self._client.post(
                "/api/v1/relationships",
                json=relation.model_dump(mode="json"),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.warning("create_relation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    async def graph_traverse(
        self, entity_id: int, depth: int = 3
    ) -> GraphTraverseResponse:
        """POST /api/v1/graph/traverse — traverse the knowledge graph."""
        try:
            resp = await self._client.post(
                "/api/v1/graph/traverse",
                json={"entity_id": entity_id, "depth": depth},
            )
            resp.raise_for_status()
            raw = resp.json()
            # API returns list of [entity, [relationships]] pairs
            nodes = [
                GraphTraverseNode(entity=item[0], relationships=item[1])
                for item in raw
            ]
            return GraphTraverseResponse(nodes=nodes)
        except httpx.HTTPError as exc:
            logger.warning("graph_traverse(%d) failed: %s", entity_id, exc)
            return GraphTraverseResponse()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
