"""Main research orchestration loop — phases 0 through 6.

Architecture:
  Phase 0: Quick Prior Knowledge Check (HiveMindDB query)
  Phase 1: PARALLEL DUAL SEARCH
    Branch A — Memory Swarm: N agents explore HiveMindDB graph
    Branch B — Web Research: Query explosion → source search → fetch + extract
  Phase 2: Store results in HiveMindDB
  Phase 3: Completeness Check — assess coverage, identify gaps
  Phase 4: Loop — if incomplete, gap-targeted queries → back to Phase 1 Branch B
  Phase 5: Synthesis — big model generates final report
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from .agent.swarm import run_memory_swarm
from .config import Settings
from .core.completeness import assess_completeness
from .core.temporal import score_facts_by_recency
from .llm.client import LLMClient
from .llm.batch import batch_complete
from .llm.prompts import (
    EXTRACTION_SYSTEM,
    GAP_ANALYSIS_SYSTEM,
    LINK_RANKING_SYSTEM,
    QUERY_GENERATION_SYSTEM,
    SYNTHESIS_SYSTEM,
    extraction_prompt,
    gap_analysis_prompt,
    link_ranking_prompt,
    query_generation_prompt,
    synthesis_prompt,
)
from .model_manager import swap_model
from .models import (
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelation,
    FetchedPage,
    PageExtraction,
    ResearchPhase,
    ResearchProgress,
    ResearchSession,
    SearchResult,
    SourceType,
    SourcedFact,
    TokenStats,
)
from .sources import fetch_all, search_all
from .storage import HiveMindClient
from .storage.models import MemoryCreate, MemoryType, EntityCreate, RelationCreate

log = logging.getLogger(__name__)

# Type alias for the progress callback
ProgressCallback = asyncio.Queue[ResearchProgress] | None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_fact(text: str) -> str:
    """Normalize fact text for deduplication — lowercase, strip, remove trailing punct."""
    t = text.strip().lower().rstrip(".")
    # Collapse whitespace
    return " ".join(t.split())


def _is_duplicate(key: str, seen: set[str]) -> bool:
    """Check if a normalized fact is a duplicate — exact match OR substring of existing."""
    if key in seen:
        return True
    # Check if new fact is contained in an existing one or vice versa
    for existing in seen:
        # If >80% of the shorter string appears in the longer one, it's a dup
        shorter, longer = (key, existing) if len(key) <= len(existing) else (existing, key)
        if shorter in longer:
            return True
    return False


def _parse_json(text: str) -> dict | list | None:
    """Best-effort parse JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ```
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None


async def _emit(queue: ProgressCallback, progress: ResearchProgress) -> None:
    if queue is not None:
        await queue.put(progress)


# ══════════════════════════════════════════════════════════════════════
# Web Research Pass — extracted from the original recursive loop
# ══════════════════════════════════════════════════════════════════════

async def _web_research_pass(
    query: str,
    queries_by_source: dict[SourceType, list[str]],
    settings: Settings,
    session: ResearchSession,
    bulk_client: LLMClient,
    visited_urls: set[str],
    seen_fact_keys: set[str],
    pending_links: list[str],
    progress: ProgressCallback,
    current_depth: int,
) -> tuple[list[SourcedFact], list[ExtractedEntity], list[PageExtraction], list[str], list[str]]:
    """Run a single web research pass: search → fetch → extract.

    Returns (new_facts, new_entities, extractions, new_pending_links, new_pending_questions).
    """
    new_facts: list[SourcedFact] = []
    new_entities: list[ExtractedEntity] = []
    new_extractions: list[PageExtraction] = []
    new_pending_links: list[str] = []
    new_pending_questions: list[str] = []

    # ── Source Search ──────────────────────────────────────────────
    await _emit(progress, ResearchProgress(
        phase=ResearchPhase.SOURCE_SEARCH,
        message=f"Searching {len(queries_by_source)} source types (depth {current_depth})...",
        depth=current_depth,
    ))

    search_results = await search_all(queries_by_source, settings)

    # Track source type counts
    for r in search_results:
        session.tokens.add_source_result(r.source_type.value)

    # Deduplicate against visited URLs
    new_results = [r for r in search_results if r.url not in visited_urls]
    log.info(
        "Depth %d: %d search results (%d new)",
        current_depth, len(search_results), len(new_results),
    )

    # Combine new search results with pending links
    to_fetch = new_results.copy()
    for link in pending_links:
        if link not in visited_urls:
            to_fetch.append(SearchResult(
                url=link, title="", snippet="",
                source_type=SourceType.WEB,
            ))

    if not to_fetch:
        return new_facts, new_entities, new_extractions, new_pending_links, new_pending_questions

    # ── Fetch + Extract ───────────────────────────────────────────
    await _emit(progress, ResearchProgress(
        phase=ResearchPhase.FETCH_EXTRACT,
        message=f"Fetching {len(to_fetch)} pages...",
        depth=current_depth,
        urls_total=len(to_fetch),
    ))

    fetched_pages = await fetch_all(to_fetch, settings)
    visited_urls.update(p.url for p in fetched_pages)
    session.urls_processed += len(fetched_pages)

    # Track web content tokens
    for p in fetched_pages:
        if p.text:
            session.tokens.add_web_content(p.text)

    # Filter out failed fetches and empty pages
    good_pages = [p for p in fetched_pages if p.text and not p.fetch_error]

    if not good_pages:
        return new_facts, new_entities, new_extractions, new_pending_links, new_pending_questions

    # LLM extraction — one call per page, all batched
    extraction_prompts = [
        extraction_prompt(
            page_text=p.text,
            page_url=p.url,
            research_query=query,
        )
        for p in good_pages
    ]

    extraction_batch = await batch_complete(
        bulk_client,
        extraction_prompts,
        system=EXTRACTION_SYSTEM,
        max_tokens=2048,
        thinking=False,
        temperature=0.1,
    )
    session.tokens.add_llm_usage(
        extraction_batch.total_prompt_tokens,
        extraction_batch.total_completion_tokens,
        phase="extraction",
        requests=extraction_batch.successful + extraction_batch.failed,
    )
    session.tokens.llm_failed_requests += extraction_batch.failed

    # Parse extractions
    for page, response in zip(good_pages, extraction_batch.texts):
        if not response:
            continue
        parsed = _parse_json(response)
        if not isinstance(parsed, dict):
            continue

        # Support both "quality" (new) and "quality_score" (old) keys
        quality = parsed.get("quality", parsed.get("quality_score", 5))
        if isinstance(quality, str):
            try:
                quality = int(float(quality))
            except (ValueError, TypeError):
                quality = 5

        extraction = PageExtraction(
            url=page.url,
            facts=[
                ExtractedFact(
                    content=f.get("fact", f.get("content", "")),
                    confidence=f.get("confidence", 0.5),
                )
                for f in parsed.get("facts", [])
                if isinstance(f, dict)
            ],
            entities=[
                ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "unknown"),
                    description=e.get("description", ""),
                )
                for e in parsed.get("entities", [])
                if isinstance(e, dict)
            ],
            relationships=[
                ExtractedRelation(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relation_type=r.get("type", "related_to"),
                )
                for r in parsed.get("relationships", [])
                if isinstance(r, dict)
            ],
            quality_score=quality,
            source_date=parsed.get("source_date"),
            promising_links=parsed.get("promising_links", []),
            follow_up_questions=parsed.get("follow_up_questions", []),
        )

        # Quality filter
        if extraction.quality_score >= settings.quality_threshold:
            new_extractions.append(extraction)
            new_pending_links.extend(extraction.promising_links)
            new_pending_questions.extend(extraction.follow_up_questions)

            # Build sourced facts (with dedup)
            for fact in extraction.facts:
                key = _normalize_fact(fact.content)
                if not key or _is_duplicate(key, seen_fact_keys):
                    continue
                seen_fact_keys.add(key)
                new_facts.append(SourcedFact(
                    content=fact.content,
                    entities=[e.name for e in extraction.entities],
                    source_url=page.url,
                    source_type=page.source_type,
                    source_title=page.title,
                    source_published=None,
                    discovered_by=session.id,
                    discovery_query=query,
                    quality_score=extraction.quality_score,
                ))

            new_entities.extend(extraction.entities)

    return new_facts, new_entities, new_extractions, new_pending_links, new_pending_questions


# ══════════════════════════════════════════════════════════════════════
# Store Phase — extracted for reuse
# ══════════════════════════════════════════════════════════════════════

async def _store_results(
    facts: list[SourcedFact],
    entities: list[ExtractedEntity],
    extractions: list[PageExtraction],
    stored_fact_count: int,
    stored_entity_names: set[str],
    entity_id_map: dict[str, int],
    hivemind: HiveMindClient,
    session: ResearchSession,
    query: str,
    progress: ProgressCallback,
) -> int:
    """Store new facts, entities, and relationships in HiveMindDB. Returns new stored_fact_count."""
    await _emit(progress, ResearchProgress(
        phase=ResearchPhase.STORE,
        message=f"Storing {len(facts)} facts and {len(entities)} entities...",
        facts_so_far=len(facts),
        entities_so_far=len(entities),
    ))

    # Store only NEW facts
    new_facts_to_store = facts[stored_fact_count:]
    for fact in new_facts_to_store:
        try:
            await hivemind.create_memory(MemoryCreate(
                content=fact.content,
                memory_type=MemoryType.FACT,
                tags=[query, fact.source_type.value],
                metadata={
                    "source_url": fact.source_url,
                    "source_type": fact.source_type.value,
                    "source_title": fact.source_title,
                    "source_credibility": fact.source_credibility,
                    "discovered_at": _now().strftime("%Y-%m-%d %H:%M:%S"),
                    "discovery_query": fact.discovery_query,
                    "discovery_session": session.id,
                    "quality_score": fact.quality_score,
                },
                agent_id="deepresearch",
            ))
        except Exception as e:
            log.warning("Failed to store fact: %s", e)

    # Store only NEW entities
    for entity in entities:
        if entity.name in stored_entity_names:
            continue
        stored_entity_names.add(entity.name)
        try:
            resp = await hivemind.create_entity(EntityCreate(
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                agent_id="deepresearch",
            ))
            if resp:
                entity_id_map[entity.name] = resp.id
        except Exception as e:
            log.warning("Failed to store entity %s: %s", entity.name, e)

    # Store relationships
    for extraction in extractions:
        for rel in extraction.relationships:
            src_id = entity_id_map.get(rel.source)
            tgt_id = entity_id_map.get(rel.target)
            if src_id and tgt_id:
                try:
                    await hivemind.create_relation(RelationCreate(
                        source_entity_id=src_id,
                        target_entity_id=tgt_id,
                        relation_type=rel.relation_type,
                        created_by="deepresearch",
                    ))
                except Exception as e:
                    log.warning("Failed to store relation: %s", e)

    return len(facts)


# ══════════════════════════════════════════════════════════════════════
# Main Research Pipeline
# ══════════════════════════════════════════════════════════════════════

async def run_research(
    query: str,
    settings: Settings,
    progress: ProgressCallback = None,
    depth: int | None = None,
    sources: list[SourceType] | None = None,
    session: ResearchSession | None = None,
) -> ResearchSession:
    """Execute the full research pipeline and return a completed session."""

    max_depth = depth if depth is not None else settings.max_depth
    active_sources = sources or list(SourceType)

    if session is None:
        session = ResearchSession(
            query=query,
            model_bulk=settings.bulk_model,
            model_synthesis=settings.synthesis_model,
        )
    else:
        session.model_bulk = settings.bulk_model
        session.model_synthesis = settings.synthesis_model

    bulk_client = LLMClient(settings.bulk_llm)
    synthesis_client = LLMClient(settings.synthesis_llm)
    hivemind = HiveMindClient(settings.hivemind_url)

    visited_urls: set[str] = set()
    all_facts: list[SourcedFact] = []
    all_entities: list[ExtractedEntity] = []
    all_extractions: list[PageExtraction] = []
    pending_links: list[str] = []

    # Dedup tracking
    seen_fact_keys: set[str] = set()
    stored_fact_count: int = 0
    stored_entity_names: set[str] = set()
    entity_id_map: dict[str, int] = {}

    try:
        # ── Phase 0: Prior Knowledge ──────────────────────────────────
        await _emit(progress, ResearchProgress(
            phase=ResearchPhase.PRIOR_KNOWLEDGE,
            message="Checking existing knowledge...",
        ))

        existing_knowledge = ""
        existing_entities: list[str] = []
        try:
            search_hits = await hivemind.search(query, limit=50)
            if search_hits:
                existing_knowledge = "\n".join(
                    f"- {hit.memory.content}" for hit in search_hits
                )
            # Try to find main topic entity and traverse
            entity_resp = await hivemind.find_entity(query)
            if entity_resp:
                existing_entities = [entity_resp.name]
                graph = await hivemind.graph_traverse(entity_resp.id, depth=3)
                if graph and graph.nodes:
                    existing_knowledge += "\n\nKnowledge graph:\n"
                    for node in graph.nodes:
                        existing_knowledge += f"- Entity: {node.entity.name} ({node.entity.entity_type})\n"
                        for rel in node.relationships:
                            existing_knowledge += f"  - {rel.get('relation_type', 'related_to')} -> {rel.get('target', '')}\n"
        except Exception as e:
            log.warning("HiveMindDB not available for prior knowledge: %s", e)

        session.phases_completed.append(ResearchPhase.PRIOR_KNOWLEDGE)

        # ── Phase 1: Query Explosion ──────────────────────────────────
        await _emit(progress, ResearchProgress(
            phase=ResearchPhase.QUERY_EXPLOSION,
            message=f"Generating search queries for {len(active_sources)} source types...",
        ))

        source_names = [s.value for s in active_sources]
        query_gen_prompt = query_generation_prompt(
            topic=query,
            source_types=source_names,
            existing_knowledge=existing_knowledge or "No prior knowledge.",
            queries_per_source=settings.queries_per_source,
        )

        query_batch = await batch_complete(
            bulk_client,
            [query_gen_prompt],
            system=QUERY_GENERATION_SYSTEM,
            max_tokens=2048,
            thinking=False,
        )
        session.tokens.add_llm_usage(
            query_batch.total_prompt_tokens,
            query_batch.total_completion_tokens,
            phase="query_explosion",
            requests=query_batch.successful + query_batch.failed,
        )

        # Parse generated queries
        queries_by_source: dict[SourceType, list[str]] = {}
        if query_batch.texts and query_batch.texts[0]:
            parsed = _parse_json(query_batch.texts[0])
            if isinstance(parsed, dict):
                for src_name, qs in parsed.items():
                    try:
                        src_type = SourceType(src_name)
                        if src_type in active_sources and isinstance(qs, list):
                            queries_by_source[src_type] = [str(q) for q in qs]
                    except ValueError:
                        continue

        # Fallback: if LLM didn't generate queries, use the original query for all sources
        for src in active_sources:
            if src not in queries_by_source:
                queries_by_source[src] = [query]

        session.phases_completed.append(ResearchPhase.QUERY_EXPLOSION)

        # ── PARALLEL DUAL SEARCH + COMPLETENESS LOOP ──────────────────
        for current_depth in range(max_depth):
            facts_before = len(all_facts)
            entities_before = len(all_entities)

            if current_depth == 0:
                # First pass: run Memory Swarm AND Web Research in PARALLEL
                await _emit(progress, ResearchProgress(
                    phase=ResearchPhase.SOURCE_SEARCH,
                    message="Running parallel dual search (memory swarm + web)...",
                    depth=current_depth,
                    facts_so_far=len(all_facts),
                    entities_so_far=len(all_entities),
                ))

                # Branch A: Memory Swarm (explores existing HiveMindDB knowledge)
                swarm_task = run_memory_swarm(
                    query=query,
                    client=bulk_client,
                    hivemind=hivemind,
                    num_agents=settings.swarm_agents,
                    max_iterations=8,
                )

                # Branch B: Web Research (searches the internet)
                web_task = _web_research_pass(
                    query=query,
                    queries_by_source=queries_by_source,
                    settings=settings,
                    session=session,
                    bulk_client=bulk_client,
                    visited_urls=visited_urls,
                    seen_fact_keys=seen_fact_keys,
                    pending_links=pending_links,
                    progress=progress,
                    current_depth=current_depth,
                )

                # Run both simultaneously
                swarm_result, web_result = await asyncio.gather(
                    swarm_task, web_task, return_exceptions=True
                )

                # Merge swarm findings into facts
                if not isinstance(swarm_result, Exception):
                    for fact_text in swarm_result.facts:
                        key = _normalize_fact(fact_text)
                        if key and not _is_duplicate(key, seen_fact_keys):
                            seen_fact_keys.add(key)
                            all_facts.append(SourcedFact(
                                content=fact_text,
                                entities=[],
                                source_url="hiveminddb://memory-swarm",
                                source_type=SourceType.WEB,  # from existing knowledge
                                source_title="Memory Swarm",
                                source_credibility=0.8,
                                discovered_by=session.id,
                                discovery_query=query,
                                quality_score=7,
                            ))
                    for ent_name in swarm_result.entities:
                        all_entities.append(ExtractedEntity(
                            name=ent_name, entity_type="unknown",
                        ))
                    log.info(
                        "Memory swarm contributed %d facts, %d entities",
                        len(swarm_result.facts), len(swarm_result.entities),
                    )
                else:
                    log.warning("Memory swarm failed: %s", swarm_result)

                # Merge web results
                if not isinstance(web_result, Exception):
                    web_facts, web_entities, web_extractions, web_links, web_questions = web_result
                    all_facts.extend(web_facts)
                    all_entities.extend(web_entities)
                    all_extractions.extend(web_extractions)
                    pending_links = web_links
                else:
                    log.warning("Web research failed: %s", web_result)
                    pending_links = []

            else:
                # Subsequent passes: web-only (swarm already explored existing knowledge)
                web_facts, web_entities, web_extractions, web_links, web_questions = (
                    await _web_research_pass(
                        query=query,
                        queries_by_source=queries_by_source,
                        settings=settings,
                        session=session,
                        bulk_client=bulk_client,
                        visited_urls=visited_urls,
                        seen_fact_keys=seen_fact_keys,
                        pending_links=pending_links,
                        progress=progress,
                        current_depth=current_depth,
                    )
                )
                all_facts.extend(web_facts)
                all_entities.extend(web_entities)
                all_extractions.extend(web_extractions)
                pending_links = web_links

            session.phases_completed.append(ResearchPhase.FETCH_EXTRACT)

            new_facts_this_depth = len(all_facts) - facts_before
            new_entities_this_depth = len(all_entities) - entities_before

            await _emit(progress, ResearchProgress(
                phase=ResearchPhase.FETCH_EXTRACT,
                message=f"Depth {current_depth}: {new_facts_this_depth} new facts, {new_entities_this_depth} new entities",
                depth=current_depth,
                urls_processed=session.urls_processed,
                facts_so_far=len(all_facts),
                entities_so_far=len(all_entities),
            ))

            # ── Store in HiveMindDB ───────────────────────────────────
            stored_fact_count = await _store_results(
                facts=all_facts,
                entities=all_entities,
                extractions=all_extractions,
                stored_fact_count=stored_fact_count,
                stored_entity_names=stored_entity_names,
                entity_id_map=entity_id_map,
                hivemind=hivemind,
                session=session,
                query=query,
                progress=progress,
            )

            session.phases_completed.append(ResearchPhase.STORE)
            session.facts_discovered = len(all_facts)
            session.entities_discovered = len(all_entities)
            session.depth_reached = current_depth + 1

            # ── Completeness Check ────────────────────────────────────
            completeness = await assess_completeness(
                query=query,
                facts=all_facts,
                entities=all_entities,
                depth=current_depth + 1,
                new_facts_this_depth=new_facts_this_depth,
                new_entities_this_depth=new_entities_this_depth,
                max_depth=max_depth,
                client=bulk_client if current_depth >= 1 else None,  # LLM check after first depth
            )

            log.info(
                "Completeness check at depth %d: continue=%s, coverage=%.0f%%, reasons=%s",
                current_depth + 1,
                completeness.should_continue,
                completeness.coverage_score * 100,
                completeness.reasons,
            )

            if not completeness.should_continue:
                break

            # ── Recursive Deepening — generate new queries ────────────
            await _emit(progress, ResearchProgress(
                phase=ResearchPhase.RECURSIVE_DEEPENING,
                message=f"Analyzing gaps for depth {current_depth + 1}...",
                depth=current_depth,
                facts_so_far=len(all_facts),
                entities_so_far=len(all_entities),
            ))

            # Filter pending links to unvisited only
            pending_links = [l for l in pending_links if l not in visited_urls]

            if pending_links:
                # Rank links by expected value
                ranking_batch = await batch_complete(
                    bulk_client,
                    [link_ranking_prompt(pending_links[:100], query, list(entity_id_map.keys()))],
                    system=LINK_RANKING_SYSTEM,
                    max_tokens=4096,
                    thinking=False,
                )
                session.tokens.add_llm_usage(
                    ranking_batch.total_prompt_tokens,
                    ranking_batch.total_completion_tokens,
                    phase="link_ranking",
                    requests=ranking_batch.successful + ranking_batch.failed,
                )
                if ranking_batch.texts and ranking_batch.texts[0]:
                    ranked = _parse_json(ranking_batch.texts[0])
                    if isinstance(ranked, list):
                        ranked.sort(key=lambda x: x.get("score", 0), reverse=True)
                        pending_links = [
                            r["url"] for r in ranked[:20]
                            if isinstance(r, dict) and "url" in r
                        ]
                    else:
                        pending_links = pending_links[:20]
                else:
                    pending_links = pending_links[:20]

            # Use completeness gaps + gap analysis to generate new queries
            gap_sources = completeness.gaps.copy()

            gap_batch = await batch_complete(
                bulk_client,
                [gap_analysis_prompt(
                    current_knowledge="\n".join(f.content for f in all_facts[-50:]),
                    original_query=query,
                )],
                system=GAP_ANALYSIS_SYSTEM,
                max_tokens=4096,
                thinking=False,
            )
            session.tokens.add_llm_usage(
                gap_batch.total_prompt_tokens,
                gap_batch.total_completion_tokens,
                phase="gap_analysis",
                requests=gap_batch.successful + gap_batch.failed,
            )
            if gap_batch.texts and gap_batch.texts[0]:
                gaps = _parse_json(gap_batch.texts[0])
                if isinstance(gaps, list):
                    new_queries = []
                    for gap in gaps:
                        if isinstance(gap, dict):
                            new_queries.extend(gap.get("suggested_queries", []))
                    if new_queries:
                        queries_by_source = {}
                        for src in active_sources:
                            queries_by_source[src] = new_queries[:settings.queries_per_source]

            session.phases_completed.append(ResearchPhase.RECURSIVE_DEEPENING)

        # ── Phase 5: Synthesis ─────────────────────────────────────────
        await _emit(progress, ResearchProgress(
            phase=ResearchPhase.SYNTHESIS,
            message="Synthesizing research report...",
            facts_so_far=len(all_facts),
            entities_so_far=len(all_entities),
        ))

        if all_facts:
            try:
                # Score facts by recency for temporal context
                scored_facts = score_facts_by_recency(all_facts)

                facts_text = "\n".join(
                    f"[{i+1}] (recency={score:.2f}) {f.content} (source: {f.source_url})"
                    for i, (f, score) in enumerate(scored_facts[:200])
                )
                graph_text = "\n".join(
                    f"- {e.name} ({e.entity_type}): {e.description}"
                    for e in all_entities[:100]
                )

                synth_prompt = synthesis_prompt(
                    facts_text=facts_text,
                    graph_text=graph_text or "No graph data.",
                    contradictions_text="",
                    query=query,
                )

                # Use synthesis model (can be a different provider, e.g. ModelGate)
                synth_batch = await batch_complete(
                    synthesis_client,
                    [synth_prompt],
                    system=SYNTHESIS_SYSTEM,
                    max_tokens=settings.synthesis_max_tokens,
                    thinking=False,
                    temperature=0.3,
                )
                session.tokens.add_llm_usage(
                    synth_batch.total_prompt_tokens,
                    synth_batch.total_completion_tokens,
                    phase="synthesis",
                    requests=synth_batch.successful + synth_batch.failed,
                )

                if synth_batch.texts and synth_batch.texts[0]:
                    session.report = synth_batch.texts[0]
                else:
                    session.report = ""
            except Exception as e:
                log.warning("Synthesis failed: %s", e)
                session.report = ""
        else:
            session.report = ""

        session.phases_completed.append(ResearchPhase.SYNTHESIS)

        session.finished_at = _now()
        session.phases_completed.append(ResearchPhase.COMPLETE)

        t = session.tokens
        log.info(
            "Token stats — LLM: %d prompt + %d completion = %d total (%d requests, %d failed) | "
            "Web: %d pages, %d chars (~%d tokens)",
            t.llm_prompt_tokens, t.llm_completion_tokens, t.llm_total_tokens,
            t.llm_requests, t.llm_failed_requests,
            t.web_pages_fetched, t.web_chars_ingested, t.web_tokens_estimated,
        )
        for phase, stats in t.phase_tokens.items():
            log.info(
                "  Phase %-20s: %d prompt + %d completion (%d requests)",
                phase, stats["prompt"], stats["completion"], stats["requests"],
            )

        await _emit(progress, ResearchProgress(
            phase=ResearchPhase.COMPLETE,
            message=f"Research complete! {len(all_facts)} facts, {len(all_entities)} entities from {session.urls_processed} URLs",
            facts_so_far=len(all_facts),
            entities_so_far=len(all_entities),
            urls_processed=session.urls_processed,
            depth=session.depth_reached,
        ))

        return session

    except Exception as e:
        log.error("Research pipeline failed: %s", e, exc_info=True)
        session.report = f"Research failed: {e}"
        session.finished_at = _now()
        await _emit(progress, ResearchProgress(
            phase=ResearchPhase.COMPLETE,
            message=f"Research failed: {e}",
            facts_so_far=len(all_facts),
            entities_so_far=len(all_entities),
            urls_processed=session.urls_processed,
        ))
        return session
    finally:
        await hivemind.close()
        await synthesis_client.close()


async def quick_search(
    query: str,
    settings: Settings,
) -> ResearchSession:
    """Single-pass search without recursive deepening."""
    return await run_research(query, settings, depth=1)
