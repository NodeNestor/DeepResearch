from __future__ import annotations

# ================================================================== #
# Prompt templates for every LLM call in the research pipeline.
# All prompts instruct the model to output raw JSON (no markdown).
# ================================================================== #

# ------------------------------------------------------------------ #
# 1. Query Generation
# ------------------------------------------------------------------ #

QUERY_GENERATION_SYSTEM = (
    "You are a research query strategist. Your job is to generate diverse, "
    "targeted search queries that will uncover comprehensive information about "
    "a topic across different source types. Output raw JSON only."
)


def query_generation_prompt(
    topic: str,
    source_types: list[str],
    existing_knowledge: str,
    queries_per_source: int,
) -> str:
    return (
        f"Research topic: {topic}\n\n"
        f"Source types to search: {', '.join(source_types)}\n\n"
        f"What we already know:\n{existing_knowledge or 'Nothing yet.'}\n\n"
        f"Generate exactly {queries_per_source} search queries PER source type. "
        "Queries should target knowledge gaps and avoid redundancy with what we "
        "already know. Vary query specificity — mix broad overview queries with "
        "narrow technical ones.\n\n"
        "Return a JSON object mapping each source type to its list of query strings:\n"
        '{"source_type_1": ["query1", "query2", ...], ...}'
    )


# ------------------------------------------------------------------ #
# 2. Page Extraction
# ------------------------------------------------------------------ #

EXTRACTION_SYSTEM = (
    "You extract useful knowledge from web pages. Output JSON only."
)


def extraction_prompt(page_text: str, page_url: str, research_query: str) -> str:
    return (
        f"TOPIC: {research_query}\n"
        f"URL: {page_url}\n\n"
        f"TEXT:\n{page_text}\n\n"
        "Is this text about the topic? If NO, return:\n"
        '{"quality": 0, "facts": [], "entities": [], "relationships": []}\n\n'
        "If YES, extract ALL knowledge worth remembering from this page. Each fact "
        "should be specific enough that someone reading it later learns something useful.\n\n"
        "Good facts (specific, useful to remember):\n"
        '- "Qwen3.5 uses Gated DeltaNet linear attention in a 3:1 ratio with softmax attention"\n'
        '- "The 0.8B model supports 262k context length"\n'
        '- "ArtificialAnalysis.ai ranks Qwen3.5-27B above the larger MoE variants"\n'
        '- "Training used 36 trillion tokens"\n\n'
        "Bad facts (DO NOT extract these):\n"
        '- "The paper discusses the architecture" (meta-description, not knowledge)\n'
        '- "The paper presents/introduces/develops X" (describes the paper, not what X IS)\n'
        '- "The model shows improvements" (no specifics — which model? how much?)\n'
        '- "The model surpasses its counterparts" (no specifics — on what?)\n'
        '- "There are no dates listed" (absence is not knowledge)\n'
        '- "FAQ includes a section on X" (existence of a FAQ is not knowledge)\n'
        '- "John Smith is a Developer who mentioned X" (testimonials are not knowledge)\n\n'
        "Rate quality based on how much useful knowledge the page has:\n"
        "0 = not about the topic at all\n"
        "3 = barely relevant, few useful details\n"
        "5 = somewhat relevant with some facts\n"
        "8 = very relevant with good specific info\n"
        "10 = primary source packed with data\n\n"
        "{\n"
        '  "quality": 0-10,\n'
        '  "facts": [{"fact": "...", "confidence": 0.0-1.0}],\n'
        '  "entities": [{"name": "...", "type": "model|org|person|benchmark|tool|concept|dataset|paper|event"}],\n'
        '  "relationships": [{"source": "entity", "target": "entity", "type": "..."}]\n'
        "}"
    )


# ------------------------------------------------------------------ #
# 3. Link Ranking
# ------------------------------------------------------------------ #

LINK_RANKING_SYSTEM = (
    "You are a research link evaluator. Given a list of URLs and a research "
    "query, score each link by its expected value for answering the query. "
    "Output raw JSON only."
)


def link_ranking_prompt(
    links: list[str], research_query: str, existing_entities: list[str]
) -> str:
    links_text = "\n".join(f"- {u}" for u in links)
    entities_text = ", ".join(existing_entities[:50]) if existing_entities else "none yet"
    return (
        f"Research query: {research_query}\n\n"
        f"Known entities so far: {entities_text}\n\n"
        f"Candidate links:\n{links_text}\n\n"
        "Score each link from 0.0 (useless) to 1.0 (essential). Prefer links "
        "that are likely to contain novel information not already covered by "
        "known entities. Deprioritize login walls, generic index pages, and "
        "duplicate content.\n\n"
        "Return a JSON array:\n"
        '[{"url": "...", "score": 0.0-1.0, "reason": "..."}]'
    )


# ------------------------------------------------------------------ #
# 4. Synthesis
# ------------------------------------------------------------------ #

SYNTHESIS_SYSTEM = (
    "You are a research synthesis expert. Produce a comprehensive, well-structured "
    "report from the provided evidence. Cite sources, flag contradictions, and "
    "rate confidence. Write in clear, precise prose. Output raw JSON only."
)


def synthesis_prompt(
    facts_text: str,
    graph_text: str,
    contradictions_text: str,
    query: str,
) -> str:
    return (
        f"Original research query: {query}\n\n"
        f"=== COLLECTED FACTS ===\n{facts_text}\n\n"
        f"=== ENTITY/RELATIONSHIP GRAPH ===\n{graph_text}\n\n"
        f"=== CONTRADICTIONS & CONFLICTS ===\n{contradictions_text or 'None detected.'}\n\n"
        "Produce a research report as a JSON object:\n"
        "{\n"
        '  "title": "...",\n'
        '  "summary": "2-3 sentence executive summary",\n'
        '  "sections": [\n'
        '    {"heading": "...", "body": "...", "confidence": 0.0-1.0, "citations": ["url1"]}\n'
        "  ],\n"
        '  "temporal_evolution": "How the topic has changed over time",\n'
        '  "controversies": ["..."],\n'
        '  "open_questions": ["..."],\n'
        '  "confidence_overall": 0.0-1.0\n'
        "}\n\n"
        "Each section body should cite sources inline as [n] where n is the 1-based "
        "index in that section's citations list. Include a temporal_evolution narrative "
        "if the facts span different time periods."
    )


# ------------------------------------------------------------------ #
# 5. Gap Analysis
# ------------------------------------------------------------------ #

GAP_ANALYSIS_SYSTEM = (
    "You are a research gap analyst. Given what is currently known about a topic, "
    "identify what is still missing or uncertain and suggest targeted queries to "
    "fill those gaps. Output raw JSON only."
)


def gap_analysis_prompt(current_knowledge: str, original_query: str) -> str:
    return (
        f"Original research query: {original_query}\n\n"
        f"What we currently know:\n{current_knowledge}\n\n"
        "Identify knowledge gaps — important aspects of the query that are not yet "
        "covered, underrepresented, or only supported by a single low-confidence "
        "source. For each gap, suggest 2-3 search queries that would help fill it.\n\n"
        "Return a JSON array:\n"
        '[{"gap": "description of what is missing", "suggested_queries": ["q1", "q2"]}]'
    )


# ------------------------------------------------------------------ #
# 6. Memory Swarm Agent
# ------------------------------------------------------------------ #

_SWARM_AGENT_BASE = (
    "You are a knowledge graph research agent exploring a HiveMindDB instance. "
    "Your specific research angle is: {angle}\n\n"
    "You have these tools:\n"
    "- memory_search: Search for memories by query and optional tags\n"
    "- entity_lookup: Find an entity by name\n"
    "- graph_traverse: Explore the graph neighborhood around an entity\n"
    "- get_memory: Read the full content of a specific memory\n\n"
    "Strategy:\n"
    "1. Search for memories related to the topic from your angle\n"
    "2. Look up entities mentioned in results\n"
    "3. Traverse the graph to discover connections\n"
    "4. When you have enough findings, call done() with your results\n\n"
    "Focus on your specific angle. Be thorough but efficient — "
    "don't repeat searches that return empty results. "
    "Call done(findings=[...], entities_found=[...]) when you have findings or "
    "have exhausted the available information."
)


def swarm_agent_system_prompt(angle: str) -> str:
    return _SWARM_AGENT_BASE.format(angle=angle)


# ------------------------------------------------------------------ #
# 7. Completeness Assessment
# ------------------------------------------------------------------ #

COMPLETENESS_SYSTEM = (
    "You assess research completeness. Given a research query and collected facts, "
    "evaluate how well the facts cover the topic. Output raw JSON only."
)


def completeness_prompt(query: str, facts_text: str) -> str:
    return (
        f"Research query: {query}\n\n"
        f"Collected facts:\n{facts_text}\n\n"
        "Assess how completely these facts answer the research query.\n\n"
        "Consider:\n"
        "- Are the main aspects of the topic covered?\n"
        "- Are there obvious gaps (missing technical details, timeline, comparisons)?\n"
        "- Is there enough depth on each aspect?\n\n"
        "Return JSON:\n"
        "{\n"
        '  "coverage_score": 0.0-1.0,\n'
        '  "gaps": ["description of gap 1", "description of gap 2"],\n'
        '  "well_covered": ["aspect 1", "aspect 2"],\n'
        '  "assessment": "brief overall assessment"\n'
        "}"
    )
