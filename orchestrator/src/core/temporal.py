"""Temporal decay scoring and contradiction resolution."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..models import SourcedFact

log = logging.getLogger(__name__)

DEFAULT_HALF_LIFE_DAYS = 180  # 6 months


def temporal_score(
    fact: SourcedFact,
    reference_date: datetime | None = None,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Score a fact by recency using exponential decay.

    Returns a value between 0.0 (very old) and 1.0 (very recent).
    Facts without a published date get a neutral score of 0.5.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # Use source_published if available, otherwise discovered_at
    fact_date = fact.source_published or fact.discovered_at
    if fact_date is None:
        return 0.5

    # Ensure timezone-aware comparison
    if fact_date.tzinfo is None:
        fact_date = fact_date.replace(tzinfo=timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    age_days = (reference_date - fact_date).total_seconds() / 86400
    if age_days < 0:
        return 1.0  # Future date — treat as maximally recent

    # Exponential decay: score = 0.5^(age / half_life)
    return math.pow(0.5, age_days / half_life_days)


@dataclass
class Contradiction:
    """A pair of facts that contradict each other."""

    fact_a: SourcedFact
    fact_b: SourcedFact
    resolution: str  # "newer_wins", "higher_credibility_wins", "unresolved"
    winner: SourcedFact | None = None


def resolve_contradictions(
    facts: list[SourcedFact],
    contradiction_pairs: list[tuple[int, int]] | None = None,
) -> list[Contradiction]:
    """Resolve contradictions between facts.

    If contradiction_pairs is provided, uses those specific pairs.
    Otherwise, checks all facts with contradicted_by references.

    Resolution strategy:
    1. Newer source wins by default
    2. If same age, higher credibility wins
    3. If tied, mark as unresolved
    """
    contradictions: list[Contradiction] = []

    if contradiction_pairs:
        for idx_a, idx_b in contradiction_pairs:
            if idx_a < len(facts) and idx_b < len(facts):
                result = _resolve_pair(facts[idx_a], facts[idx_b])
                contradictions.append(result)
    else:
        # Build index of facts by their position
        for i, fact in enumerate(facts):
            for contra_idx in fact.contradicted_by:
                if contra_idx < len(facts) and contra_idx > i:
                    result = _resolve_pair(fact, facts[contra_idx])
                    contradictions.append(result)

    return contradictions


def _resolve_pair(fact_a: SourcedFact, fact_b: SourcedFact) -> Contradiction:
    """Resolve a single contradiction between two facts."""
    date_a = fact_a.source_published or fact_a.discovered_at
    date_b = fact_b.source_published or fact_b.discovered_at

    # Compare by date
    if date_a and date_b:
        if date_a > date_b:
            return Contradiction(
                fact_a=fact_a,
                fact_b=fact_b,
                resolution="newer_wins",
                winner=fact_a,
            )
        elif date_b > date_a:
            return Contradiction(
                fact_a=fact_a,
                fact_b=fact_b,
                resolution="newer_wins",
                winner=fact_b,
            )

    # Compare by credibility
    if fact_a.source_credibility != fact_b.source_credibility:
        winner = fact_a if fact_a.source_credibility > fact_b.source_credibility else fact_b
        return Contradiction(
            fact_a=fact_a,
            fact_b=fact_b,
            resolution="higher_credibility_wins",
            winner=winner,
        )

    # Unresolved
    return Contradiction(
        fact_a=fact_a,
        fact_b=fact_b,
        resolution="unresolved",
        winner=None,
    )


def score_facts_by_recency(
    facts: list[SourcedFact],
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
) -> list[tuple[SourcedFact, float]]:
    """Score all facts by recency and return sorted (most recent first)."""
    now = datetime.now(timezone.utc)
    scored = [(f, temporal_score(f, now, half_life_days)) for f in facts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
