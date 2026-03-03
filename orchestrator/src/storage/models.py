"""Pydantic models for HiveMindDB API requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (mirror HiveMindDB types.rs)
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    FACT = "fact"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------

class MemoryCreate(BaseModel):
    """POST /api/v1/memories request body."""
    content: str
    memory_type: MemoryType = MemoryType.FACT
    agent_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Memory object returned by HiveMindDB."""
    id: int
    content: str
    memory_type: MemoryType
    agent_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    confidence: float = 0.9
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    valid_from: datetime
    valid_until: datetime | None = None
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    """PUT /api/v1/memories/{id} request body."""
    content: str | None = None
    tags: list[str] | None = None
    confidence: float | None = None
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Search models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """POST /api/v1/search request body."""
    query: str
    agent_id: str | None = None
    user_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    limit: int = 10
    include_graph: bool = False


class SearchResult(BaseModel):
    """A single search hit (memory + score)."""
    memory: MemoryResponse
    score: float
    related_entities: list[EntityResponse] = Field(default_factory=list)
    related_relationships: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity models
# ---------------------------------------------------------------------------

class EntityCreate(BaseModel):
    """POST /api/v1/entities request body."""
    name: str
    entity_type: str
    description: str | None = None
    agent_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class EntityResponse(BaseModel):
    """Entity object returned by HiveMindDB."""
    id: int
    name: str
    entity_type: str
    description: str | None = None
    agent_id: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Relationship models
# ---------------------------------------------------------------------------

class RelationCreate(BaseModel):
    """POST /api/v1/relationships request body."""
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    description: str | None = None
    weight: float = 1.0
    created_by: str = "deep-research"
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph traversal models
# ---------------------------------------------------------------------------

class GraphTraverseRequest(BaseModel):
    """POST /api/v1/graph/traverse request body."""
    entity_id: int
    depth: int = 2


class GraphTraverseNode(BaseModel):
    """A node in the traversal result (entity + its relationships)."""
    entity: EntityResponse
    relationships: list[dict] = Field(default_factory=list)


class GraphTraverseResponse(BaseModel):
    """Response from graph traversal — list of (entity, relationships) pairs."""
    nodes: list[GraphTraverseNode] = Field(default_factory=list)


# Forward-ref update (SearchResult references EntityResponse)
SearchResult.model_rebuild()
