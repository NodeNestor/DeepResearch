"""Agent tool definitions and executor for HiveMindDB exploration.

Lifted from deep-memory-explorer/backend/src/routers/agent.py and adapted
for the orchestrator's HiveMindClient.
"""

from __future__ import annotations

import json
import logging

from ..storage.hivemind import HiveMindClient

log = logging.getLogger(__name__)

# OpenAI-compatible tool definitions for the agent
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search for memories by semantic query and optional tag filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tag filters",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "entity_lookup",
            "description": "Find an entity by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name to look up"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_traverse",
            "description": "Explore the graph neighborhood around an entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "integer", "description": "Entity ID to start from"},
                    "depth": {
                        "type": "integer",
                        "description": "Traversal depth",
                        "default": 2,
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory",
            "description": "Read the full content of a specific memory by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Memory ID"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Return your findings as a JSON list of facts discovered from the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of facts/findings discovered",
                    },
                    "entities_found": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names encountered",
                    },
                },
                "required": ["findings"],
            },
        },
    },
]


async def execute_tool(
    tool_name: str,
    args: dict,
    hivemind: HiveMindClient,
) -> str:
    """Execute a single agent tool and return JSON result string."""
    try:
        if tool_name == "memory_search":
            results = await hivemind.search(
                query=args["query"],
                tags=args.get("tags"),
                limit=args.get("limit", 10),
            )
            out = []
            for r in results:
                mem = r.memory
                out.append({
                    "id": mem.id,
                    "content": mem.content[:500],
                    "score": r.score,
                    "tags": mem.tags,
                    "metadata": {
                        k: v for k, v in (mem.metadata or {}).items()
                        if k in ("source_url", "source_type", "discovered_at")
                    },
                })
            return json.dumps(out, default=str)

        elif tool_name == "entity_lookup":
            entity = await hivemind.find_entity(args["name"])
            if not entity:
                return json.dumps({"error": f"Entity '{args['name']}' not found"})
            return json.dumps({
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
            }, default=str)

        elif tool_name == "graph_traverse":
            graph = await hivemind.graph_traverse(
                entity_id=args["entity_id"],
                depth=args.get("depth", 2),
            )
            nodes = []
            for n in graph.nodes:
                nodes.append({
                    "entity": {
                        "id": n.entity.id,
                        "name": n.entity.name,
                        "type": n.entity.entity_type,
                        "description": n.entity.description,
                    },
                    "relationship_count": len(n.relationships),
                    "relationships": n.relationships[:10],
                })
            return json.dumps(nodes, default=str)

        elif tool_name == "get_memory":
            mem = await hivemind.get_memory(args["id"])
            if not mem:
                return json.dumps({"error": f"Memory {args['id']} not found"})
            return json.dumps({
                "id": mem.id,
                "content": mem.content,
                "tags": mem.tags,
                "metadata": mem.metadata,
            }, default=str)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        log.error("Tool %s execution failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})
