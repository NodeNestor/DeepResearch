"""Intelligent stopping logic — coverage scoring and diminishing returns detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import COMPLETENESS_SYSTEM, completeness_prompt
from ..models import ExtractedEntity, SourcedFact

log = logging.getLogger(__name__)

MIN_DEPTH = 2
DEFAULT_MAX_DEPTH = 5
DIMINISHING_RETURNS_THRESHOLD = 0.10  # <10% new facts = saturated


@dataclass
class CompletenessResult:
    """Result of a completeness assessment."""

    should_continue: bool
    coverage_score: float  # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


async def assess_completeness(
    query: str,
    facts: list[SourcedFact],
    entities: list[ExtractedEntity],
    depth: int,
    new_facts_this_depth: int,
    new_entities_this_depth: int,
    max_depth: int = DEFAULT_MAX_DEPTH,
    client: LLMClient | None = None,
) -> CompletenessResult:
    """Decide whether to continue researching or stop.

    Uses a combination of heuristics and (optionally) LLM evaluation.
    """
    reasons: list[str] = []
    gaps: list[str] = []

    # Hard cap — never exceed max_depth
    if depth >= max_depth:
        return CompletenessResult(
            should_continue=False,
            coverage_score=0.8,
            reasons=[f"Reached maximum depth ({max_depth})"],
        )

    # Min depth — always do at least MIN_DEPTH passes
    if depth < MIN_DEPTH:
        return CompletenessResult(
            should_continue=True,
            coverage_score=0.0,
            reasons=[f"Below minimum depth ({MIN_DEPTH})"],
        )

    # No facts at all — keep going
    if not facts:
        return CompletenessResult(
            should_continue=True,
            coverage_score=0.0,
            reasons=["No facts discovered yet"],
        )

    # Diminishing returns — fewer than 10% new facts
    if len(facts) > 0 and new_facts_this_depth > 0:
        ratio = new_facts_this_depth / len(facts)
        if ratio < DIMINISHING_RETURNS_THRESHOLD:
            reasons.append(
                f"Diminishing returns: only {new_facts_this_depth} new facts "
                f"({ratio:.0%} of {len(facts)} total)"
            )

    # Entity saturation — no new entities found
    if new_entities_this_depth == 0 and len(entities) > 0:
        reasons.append("Entity saturation: no new entities discovered")

    # Zero new facts this depth
    if new_facts_this_depth == 0:
        return CompletenessResult(
            should_continue=False,
            coverage_score=0.9,
            reasons=["No new facts found this depth — likely saturated"],
        )

    # If we have a client, use LLM to evaluate coverage
    if client is not None and len(facts) >= 5:
        try:
            coverage_score, llm_gaps = await _llm_coverage_check(
                query, facts, client
            )
            if coverage_score >= 0.85:
                reasons.append(f"LLM coverage assessment: {coverage_score:.0%}")
                return CompletenessResult(
                    should_continue=False,
                    coverage_score=coverage_score,
                    reasons=reasons,
                    gaps=llm_gaps,
                )
            gaps.extend(llm_gaps)
        except Exception as e:
            log.warning("LLM coverage check failed: %s", e)

    # If heuristics flagged diminishing returns, stop
    if reasons:
        return CompletenessResult(
            should_continue=False,
            coverage_score=0.7,
            reasons=reasons,
            gaps=gaps,
        )

    # Default: keep going
    return CompletenessResult(
        should_continue=True,
        coverage_score=len(facts) / max(len(facts) + 10, 1),
        reasons=["Research still productive"],
        gaps=gaps,
    )


async def _llm_coverage_check(
    query: str,
    facts: list[SourcedFact],
    client: LLMClient,
) -> tuple[float, list[str]]:
    """Use LLM to assess how well the facts cover the query."""
    facts_text = "\n".join(f"- {f.content}" for f in facts[:100])
    prompt = completeness_prompt(query, facts_text)

    result = await client.complete(
        messages=[
            {"role": "system", "content": COMPLETENESS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        thinking=False,
        temperature=0.1,
    )

    # Parse JSON response
    text = result.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return 0.5, []
        else:
            return 0.5, []

    score = float(parsed.get("coverage_score", 0.5))
    gaps = parsed.get("gaps", [])
    if isinstance(gaps, list):
        gaps = [str(g) for g in gaps]
    else:
        gaps = []

    return score, gaps
