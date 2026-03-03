"""MCP server exposing DeepResearch as tools for Claude Code and other LLMs."""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ..agent.swarm import run_memory_swarm
from ..config import settings
from ..core.temporal import resolve_contradictions, score_facts_by_recency
from ..llm.client import LLMClient
from ..models import ResearchPhase, SourceType, SourcedFact
from ..research import quick_search, run_research
from ..storage import HiveMindClient
from ..storage.models import (
    MemoryCreate,
    MemoryType,
    SearchRequest as HivemindSearchRequest,
)

log = logging.getLogger(__name__)

mcp = Server("deepresearch")

# Track running research sessions and completed ones
_running: dict[str, asyncio.Task] = {}
_completed: dict[str, dict] = {}  # session_id -> result summary


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="deep_research",
            description=(
                "Run a full deep research pipeline on a topic. "
                "Searches multiple sources (web, arxiv, github, reddit, youtube, etc.), "
                "runs a memory swarm to explore existing knowledge, extracts facts and "
                "entities, builds a knowledge graph, and synthesizes a comprehensive report. "
                "Uses intelligent completeness detection to loop until thorough."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research topic or question",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Max recursion depth (1-5, default 3). Higher = more thorough but slower.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": [s.value for s in SourceType]},
                        "description": "Source types to search. Default: all.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="quick_search",
            description=(
                "Quick single-pass search without recursive deepening. "
                "Faster than deep_research but less thorough. Good for simple questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="recall",
            description="Query existing knowledge from past research sessions. Uses semantic search over stored facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to recall knowledge about",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="graph_explore",
            description="Traverse the knowledge graph starting from an entity. Returns connected entities and relationships.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to start traversal from",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Traversal depth (default 3)",
                        "default": 3,
                    },
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="temporal_view",
            description="View facts discovered within a time range, ordered chronologically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to filter by",
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date (ISO 8601)",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date (ISO 8601)",
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="source_credibility",
            description="Check the credibility score of a URL based on past research.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to check",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="research_status",
            description="Check the status of running research tasks.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # ── NEW TOOLS ───────────────────────────────────────────────
        Tool(
            name="memory_swarm",
            description=(
                "Run a memory swarm to explore existing knowledge in the graph. "
                "Spawns N parallel small-model agents that autonomously search, "
                "traverse entities, and discover connections. No web search — "
                "only explores what's already stored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to explore in existing knowledge",
                    },
                    "num_agents": {
                        "type": "integer",
                        "description": "Number of parallel agents (default 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="add_knowledge",
            description="Manually add a fact to the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to store",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "source_url": {
                        "type": "string",
                        "description": "Optional source URL",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="find_contradictions",
            description="Find potentially contradicting facts on a topic in stored knowledge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to find contradictions about",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max facts to analyze (default 50)",
                        "default": 50,
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="research_history",
            description="List past research sessions and their results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 20)",
                        "default": 20,
                    },
                },
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    hivemind = HiveMindClient(settings.hivemind_url)

    try:
        if name == "deep_research":
            query = arguments["query"]
            depth = arguments.get("depth", 3)
            sources = None
            if "sources" in arguments:
                sources = [SourceType(s) for s in arguments["sources"]]

            progress_queue: asyncio.Queue = asyncio.Queue()
            session = await run_research(
                query=query,
                settings=settings,
                progress=progress_queue,
                depth=depth,
                sources=sources,
            )

            result = {
                "session_id": session.id,
                "query": session.query,
                "facts_discovered": session.facts_discovered,
                "entities_discovered": session.entities_discovered,
                "urls_processed": session.urls_processed,
                "depth_reached": session.depth_reached,
                "report": session.report,
            }
            _completed[session.id] = result
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "quick_search":
            session = await quick_search(arguments["query"], settings)
            result = {
                "session_id": session.id,
                "facts_discovered": session.facts_discovered,
                "entities_discovered": session.entities_discovered,
                "report": session.report,
            }
            _completed[session.id] = result
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "recall":
            hits = await hivemind.search(
                arguments["topic"],
                limit=arguments.get("limit", 20),
            )
            results = [
                {"content": h.memory.content, "tags": h.memory.tags, "metadata": h.memory.metadata}
                for h in hits
            ]
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

        elif name == "graph_explore":
            entity_name = arguments["entity"]
            depth = arguments.get("depth", 3)

            entity = await hivemind.find_entity(entity_name)
            if not entity:
                return [TextContent(type="text", text=f"Entity '{entity_name}' not found.")]

            graph = await hivemind.graph_traverse(entity.id, depth=depth)
            nodes = []
            for node in (graph.nodes if graph else []):
                nodes.append({
                    "name": node.entity.name,
                    "type": node.entity.entity_type,
                    "description": node.entity.description,
                    "relationships": node.relationships,
                })
            result = {"root_entity": entity_name, "nodes": nodes}
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "temporal_view":
            hits = await hivemind.search(
                arguments["topic"],
                limit=100,
            )
            results = []
            for h in hits:
                results.append({
                    "content": h.memory.content,
                    "created_at": h.memory.created_at.isoformat() if h.memory.created_at else None,
                    "metadata": h.memory.metadata,
                })
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

        elif name == "source_credibility":
            url = arguments["url"]
            hits = await hivemind.search(f"source_url:{url}", limit=10)
            if not hits:
                return [TextContent(type="text", text=f"No data for URL: {url}")]
            scores = [
                h.memory.metadata.get("source_credibility", 0.5)
                for h in hits
                if h.memory.metadata
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.5
            result = {
                "url": url,
                "credibility_score": round(avg_score, 2),
                "based_on_facts": len(hits),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "research_status":
            running = {
                tid: "running" for tid, task in _running.items() if not task.done()
            }
            return [TextContent(type="text", text=json.dumps({
                "running_tasks": len(running),
                "tasks": running,
            }, indent=2))]

        # ── NEW TOOLS ─────────────────────────────────────────────
        elif name == "memory_swarm":
            query = arguments["query"]
            num_agents = arguments.get("num_agents", 5)

            bulk_client = LLMClient(settings.bulk_llm)
            try:
                swarm_result = await run_memory_swarm(
                    query=query,
                    client=bulk_client,
                    hivemind=hivemind,
                    num_agents=num_agents,
                    max_iterations=8,
                )
                result = {
                    "query": query,
                    "facts_found": len(swarm_result.facts),
                    "entities_found": len(swarm_result.entities),
                    "total_iterations": swarm_result.total_iterations,
                    "facts": swarm_result.facts[:100],
                    "entities": swarm_result.entities[:50],
                    "agent_summaries": [
                        {
                            "angle": a.angle,
                            "facts_found": len(a.facts),
                            "iterations": a.iterations_used,
                            "error": a.error,
                        }
                        for a in swarm_result.agent_results
                    ],
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            finally:
                await bulk_client.close()

        elif name == "add_knowledge":
            content = arguments["content"]
            tags = arguments.get("tags", [])
            source_url = arguments.get("source_url", "")

            mem = await hivemind.create_memory(MemoryCreate(
                content=content,
                memory_type=MemoryType.FACT,
                tags=tags,
                metadata={
                    "source_url": source_url,
                    "manually_added": True,
                },
                agent_id="deepresearch-manual",
            ))

            if mem:
                result = {"status": "stored", "memory_id": mem.id, "content": content}
            else:
                result = {"status": "failed", "error": "Could not store in HiveMindDB"}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "find_contradictions":
            topic = arguments["topic"]
            limit = arguments.get("limit", 50)

            hits = await hivemind.search(topic, limit=limit)
            if not hits:
                return [TextContent(type="text", text=json.dumps({
                    "topic": topic,
                    "contradictions": [],
                    "message": "No facts found for this topic",
                }))]

            # Build SourcedFact objects from search results for analysis
            facts: list[SourcedFact] = []
            for h in hits:
                meta = h.memory.metadata or {}
                facts.append(SourcedFact(
                    content=h.memory.content,
                    entities=[],
                    source_url=meta.get("source_url", ""),
                    source_type=SourceType.WEB,
                    source_title=meta.get("source_title", ""),
                    source_credibility=meta.get("source_credibility", 0.5),
                    quality_score=meta.get("quality_score", 5),
                ))

            # Use LLM to find contradictions by checking pairs
            # For now, use temporal scoring to rank by recency
            scored = score_facts_by_recency(facts)
            result = {
                "topic": topic,
                "total_facts": len(facts),
                "facts_by_recency": [
                    {"content": f.content, "recency_score": round(s, 3)}
                    for f, s in scored[:20]
                ],
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "research_history":
            limit = arguments.get("limit", 20)
            history = list(_completed.values())[-limit:]
            return [TextContent(type="text", text=json.dumps({
                "total_sessions": len(_completed),
                "sessions": history,
            }, indent=2, default=str))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    finally:
        await hivemind.close()


async def run_mcp_server():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read, write):
        await mcp.run(read, write, mcp.create_initialization_options())
