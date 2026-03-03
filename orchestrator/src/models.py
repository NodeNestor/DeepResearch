from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class SourceType(str, Enum):
    WEB = "web"
    ARXIV = "arxiv"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    GITHUB = "github"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    HUGGINGFACE = "huggingface"
    WIKIPEDIA = "wikipedia"


class ResearchPhase(str, Enum):
    PRIOR_KNOWLEDGE = "prior_knowledge"
    QUERY_EXPLOSION = "query_explosion"
    SOURCE_SEARCH = "source_search"
    FETCH_EXTRACT = "fetch_extract"
    STORE = "store"
    RECURSIVE_DEEPENING = "recursive_deepening"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"


@dataclass
class SearchResult:
    """A single search result from any source."""

    url: str
    title: str
    snippet: str
    source_type: SourceType
    source_date: str | None = None


@dataclass
class ExtractedFact:
    """A fact extracted from a fetched page by the LLM."""

    content: str
    confidence: float = 0.9


@dataclass
class ExtractedEntity:
    """An entity extracted from a fetched page."""

    name: str
    entity_type: str
    description: str = ""


@dataclass
class ExtractedRelation:
    """A relationship between two entities."""

    source: str
    target: str
    relation_type: str


@dataclass
class PageExtraction:
    """Full LLM extraction result for a single page."""

    url: str
    facts: list[ExtractedFact] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelation] = field(default_factory=list)
    quality_score: int = 5
    source_date: str | None = None
    promising_links: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)


@dataclass
class SourcedFact:
    """A fact with full provenance tracking."""

    content: str
    entities: list[str]
    source_url: str
    source_type: SourceType
    source_title: str
    source_author: str | None = None
    source_published: datetime | None = None
    source_credibility: float = 0.5
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    discovered_by: str = ""
    discovery_query: str = ""
    quality_score: int = 5
    corroborated_by: list[int] = field(default_factory=list)
    contradicted_by: list[int] = field(default_factory=list)


@dataclass
class ResearchProgress:
    """Live progress update sent over WebSocket."""

    phase: ResearchPhase
    message: str
    facts_so_far: int = 0
    entities_so_far: int = 0
    urls_processed: int = 0
    urls_total: int = 0
    depth: int = 0


@dataclass
class TokenStats:
    """Token usage tracking for a research run."""

    web_pages_fetched: int = 0
    web_chars_ingested: int = 0
    web_tokens_estimated: int = 0  # ~4 chars per token
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_requests: int = 0
    llm_failed_requests: int = 0
    # Per-phase breakdown
    phase_tokens: dict = field(default_factory=dict)
    # Per-source-type breakdown
    source_counts: dict = field(default_factory=dict)  # {source_type: count}

    def add_source_result(self, source_type: str) -> None:
        self.source_counts[source_type] = self.source_counts.get(source_type, 0) + 1

    def add_llm_usage(self, prompt: int, completion: int, phase: str = "", requests: int = 1) -> None:
        self.llm_prompt_tokens += prompt
        self.llm_completion_tokens += completion
        self.llm_total_tokens += prompt + completion
        self.llm_requests += requests
        if phase:
            if phase not in self.phase_tokens:
                self.phase_tokens[phase] = {"prompt": 0, "completion": 0, "requests": 0}
            self.phase_tokens[phase]["prompt"] += prompt
            self.phase_tokens[phase]["completion"] += completion
            self.phase_tokens[phase]["requests"] += requests

    def add_web_content(self, text: str) -> None:
        self.web_pages_fetched += 1
        chars = len(text)
        self.web_chars_ingested += chars
        self.web_tokens_estimated += chars // 4  # rough estimate


@dataclass
class ResearchSession:
    """Tracks a single research run."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    query: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    phases_completed: list[str] = field(default_factory=list)
    facts_discovered: int = 0
    entities_discovered: int = 0
    urls_processed: int = 0
    depth_reached: int = 0
    model_bulk: str = ""
    model_synthesis: str = ""
    report: str = ""
    tokens: TokenStats = field(default_factory=TokenStats)


@dataclass
class ResearchRequest:
    """API request to start a research run."""

    query: str
    depth: int | None = None
    sources: list[SourceType] | None = None


@dataclass
class FetchedPage:
    """A page that has been downloaded and text-extracted."""

    url: str
    title: str
    text: str
    source_type: SourceType
    source_date: str | None = None
    fetch_error: str | None = None
