"""Memory Swarm — N parallel small-model agents exploring HiveMindDB."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import swarm_agent_system_prompt
from ..storage.hivemind import HiveMindClient
from .tools import AGENT_TOOLS, execute_tool

log = logging.getLogger(__name__)

# Different research angles for swarm agents
SWARM_ANGLES = [
    "Find technical details, specifications, and architecture information",
    "Find the timeline and history — when things happened, versions, milestones",
    "Find competing or alternative approaches and comparisons",
    "Find limitations, criticisms, known issues, and failure modes",
    "Find real-world applications, use cases, and adoption",
    "Find key people, organizations, and their contributions",
    "Find performance benchmarks, metrics, and quantitative data",
    "Find future plans, roadmaps, and upcoming developments",
]


@dataclass
class AgentFindings:
    """Findings from a single swarm agent."""

    angle: str
    facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    iterations_used: int = 0
    error: str | None = None


@dataclass
class SwarmResult:
    """Merged results from all swarm agents."""

    facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    agent_results: list[AgentFindings] = field(default_factory=list)
    total_iterations: int = 0


async def run_memory_swarm(
    query: str,
    client: LLMClient,
    hivemind: HiveMindClient,
    num_agents: int = 5,
    max_iterations: int = 8,
) -> SwarmResult:
    """Spawn N parallel small-model agents to explore HiveMindDB.

    Each agent gets a different research angle and autonomously explores
    the knowledge graph using tool calls. Results are merged and deduplicated.
    """
    # Select angles (cycle if num_agents > len(SWARM_ANGLES))
    angles = [SWARM_ANGLES[i % len(SWARM_ANGLES)] for i in range(num_agents)]

    # Run all agents in parallel
    tasks = [
        _run_single_agent(query, angle, client, hivemind, max_iterations)
        for angle in angles
    ]
    agent_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge results
    all_facts: list[str] = []
    all_entities: list[str] = []
    findings_list: list[AgentFindings] = []
    total_iters = 0

    seen_facts: set[str] = set()
    seen_entities: set[str] = set()

    for result in agent_results:
        if isinstance(result, Exception):
            log.warning("Swarm agent failed: %s", result)
            findings_list.append(AgentFindings(
                angle="unknown",
                error=str(result),
            ))
            continue

        findings_list.append(result)
        total_iters += result.iterations_used

        for fact in result.facts:
            key = fact.strip().lower()
            if key and key not in seen_facts:
                seen_facts.add(key)
                all_facts.append(fact)

        for entity in result.entities:
            key = entity.strip().lower()
            if key and key not in seen_entities:
                seen_entities.add(key)
                all_entities.append(entity)

    log.info(
        "Memory swarm complete: %d agents, %d total iterations, %d unique facts, %d unique entities",
        num_agents, total_iters, len(all_facts), len(all_entities),
    )

    return SwarmResult(
        facts=all_facts,
        entities=all_entities,
        agent_results=findings_list,
        total_iterations=total_iters,
    )


async def _run_single_agent(
    query: str,
    angle: str,
    client: LLMClient,
    hivemind: HiveMindClient,
    max_iterations: int,
) -> AgentFindings:
    """Run a single swarm agent with a specific research angle."""
    system = swarm_agent_system_prompt(angle)
    conversation: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Research this topic in the knowledge graph: {query}"},
    ]

    findings = AgentFindings(angle=angle)

    for iteration in range(max_iterations):
        findings.iterations_used = iteration + 1

        try:
            response = await client.complete_with_tools(
                messages=conversation,
                tools=AGENT_TOOLS,
            )
        except Exception as exc:
            log.warning("Swarm agent LLM call failed at iteration %d: %s", iteration, exc)
            findings.error = str(exc)
            break

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            # Model responded with text directly — try to extract facts
            content = response.get("content", "")
            if content:
                findings.facts.append(content)
            break

        # Append assistant message to conversation
        conversation.append(response)

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}
            tc_id = tc.get("id", f"call_{iteration}")

            # Handle done() tool
            if fn_name == "done":
                findings.facts.extend(fn_args.get("findings", []))
                findings.entities.extend(fn_args.get("entities_found", []))
                return findings

            # Execute tool
            result = await execute_tool(fn_name, fn_args, hivemind)
            result_summary = result if len(result) < 4000 else result[:4000] + "...(truncated)"

            conversation.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_summary,
            })

    # Max iterations reached — ask for summary
    conversation.append({
        "role": "user",
        "content": "You've used all your exploration steps. Call done() now with whatever findings you have.",
    })
    try:
        response = await client.complete_with_tools(
            messages=conversation,
            tools=AGENT_TOOLS,
        )
        tool_calls = response.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                if fn_name == "done":
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}
                    findings.facts.extend(fn_args.get("findings", []))
                    findings.entities.extend(fn_args.get("entities_found", []))
        else:
            content = response.get("content", "")
            if content:
                findings.facts.append(content)
    except Exception as exc:
        log.warning("Swarm agent final summary failed: %s", exc)

    return findings
