# DeepResearch — Development Guide

## What This Is

Autonomous AI research agent. Searches 8 source types simultaneously, uses Memory Swarm (parallel small-model agents) to explore existing knowledge, extracts facts/entities/relationships, stores everything in HiveMindDB, and synthesizes reports using a configurable synthesis model.

## Architecture

4 Docker containers on `deepresearch-net` bridge network:
- **searxng** (:8888→8080) — metasearch engine proxy
- **hiveminddb** (:8100) — knowledge graph + vector DB (git submodule from NodeNestor/HiveMindDB)
- **orchestrator** (:8082→8080) — Python FastAPI backend + React frontend + MCP server
- **vllm** (:8000) — CUDA local LLM inference (Qwen3.5)

The orchestrator container serves everything: REST API, WebSocket, static frontend, and MCP (stdio).

## Two-Model Architecture

- **Extraction model** (bulk): Small, fast. Used for query generation, fact extraction, memory swarm agents. Default: Qwen3.5-0.8B on local vLLM.
- **Synthesis model**: Larger, smarter. Used for final report. Can be any OpenAI-compatible endpoint — vLLM, ModelGate, OpenAI, Ollama, Anthropic. Configured independently via `SYNTHESIS_API_URL` / `SYNTHESIS_MODEL` or the web UI.

Both models are configured at runtime via `PUT /api/config` — no restart needed.

## Key Directories

```
orchestrator/
├── Dockerfile           — Multi-stage: Node builds frontend, Python serves everything
├── pyproject.toml       — Python dependencies
├── frontend/            — React + Vite + Tailwind web UI
│   └── src/
│       ├── pages/       — ResearchPage, HistoryPage, SettingsPage
│       ├── components/  — layout/ (Shell, Sidebar, StatusBar) + ui/ (Button, Card, etc.)
│       └── lib/         — api.ts (REST/WS client), utils.ts
└── src/
    ├── main.py          — FastAPI app, static file serving, WebSocket, config API, MCP entry
    ├── config.py        — Pydantic settings + mutable runtime config (persisted to /app/data/)
    ├── models.py        — All data classes (SourcedFact, SearchResult, ResearchSession, etc.)
    ├── research.py      — Main pipeline (parallel dual search: memory swarm + web)
    ├── model_manager.py — vLLM model swap logic
    ├── core/
    │   ├── completeness.py — Coverage scoring, diminishing returns, intelligent stopping
    │   └── temporal.py     — Temporal decay scoring, contradiction resolution
    ├── agent/
    │   ├── swarm.py     — Memory Swarm: N parallel small-model agents exploring HiveMindDB
    │   └── tools.py     — Agent tool definitions (memory_search, entity_lookup, etc.)
    ├── llm/
    │   ├── client.py    — Provider-agnostic LLM client (OpenAI-compat + Anthropic)
    │   ├── batch.py     — Batched concurrent LLM dispatcher
    │   └── prompts.py   — All prompt templates
    ├── sources/         — 8 source modules + registry (web, arxiv, github, reddit, etc.)
    ├── storage/
    │   ├── hivemind.py  — HiveMindDB REST client
    │   └── models.py    — Pydantic models for HiveMindDB API
    └── mcp/
        └── server.py    — 11 MCP tool definitions
```

## Commands

```bash
# Start everything
docker compose up -d

# Rebuild orchestrator after code changes
docker compose build orchestrator && docker compose up -d orchestrator

# Start without vLLM (if running vLLM natively)
docker compose up -d searxng hiveminddb orchestrator
bash vllm/start.sh

# Open web UI
open http://localhost:8082

# Test health
curl http://localhost:8082/api/health

# Run research via API
curl -X POST http://localhost:8082/api/research \
  -H "Content-Type: application/json" \
  -d '{"query":"quantum error correction", "depth": 2}'

# Get/update runtime config
curl http://localhost:8082/api/config
curl -X PUT http://localhost:8082/api/config \
  -H "Content-Type: application/json" \
  -d '{"synthesis_api_url":"http://modelgate:8989/v1","synthesis_model":"gpt-4o"}'

# MCP mode (stdio)
python -m orchestrator.src.main mcp
```

## Research Pipeline

0. **Prior Knowledge** — query HiveMindDB for existing facts/graph
1. **Parallel Dual Search** — Memory Swarm + Web Research run simultaneously via `asyncio.gather()`
   - Branch A: N small-model agents explore HiveMindDB with different angles
   - Branch B: Query explosion → 8 sources → parallel fetch + LLM extraction
   - Results merged + deduplicated
2. **Store** — facts→memories, entities→graph nodes, relations→edges in HiveMindDB
3. **Completeness Check** — coverage scoring, diminishing returns detection
4. **Loop** — if incomplete, generate gap-targeted queries → re-run web branch
5. **Synthesis** — synthesis model generates final report with temporal context

## API Routes

All REST under `/api/`:
- `GET /api/health` — service status
- `GET /api/config` / `PUT /api/config` — runtime settings
- `POST /api/research` — blocking research
- `POST /api/research/async` — async research (returns session_id)
- `GET /api/research/{id}` — get session results
- `GET /api/sessions` — list all sessions

Legacy routes without `/api/` prefix still work for backwards compatibility.

WebSocket: `ws://localhost:8082/ws` — send `{"session_id": "..."}` to subscribe.

Static frontend served at `/` with SPA fallback.

## Frontend Development

```bash
cd orchestrator/frontend
npm install
npm run dev    # Dev server at :5174 with API proxy to :8082
npm run build  # Production build to dist/
```

The Dockerfile handles the build automatically — just rebuild the container.

## Config

Runtime config persisted to `/app/data/config.json` (Docker volume `orchestrator-data`).

Key env vars (`.env`):
- `BULK_MODEL`, `BULK_API_URL` — extraction model
- `SYNTHESIS_MODEL`, `SYNTHESIS_API_URL` — synthesis model (can be different provider)
- `GPU_DEVICE` — GPU UUID for vLLM
- `MAX_CONCURRENT_LLM` — parallel LLM calls (default 100)
- `MAX_CONCURRENT_FETCHES` — parallel page downloads (default 50)
- `MAX_DEPTH` — max research iterations (default 3)

## Dependencies

- Python 3.12, Node 22 (for frontend build)
- Docker + Docker Compose
- NVIDIA GPU with recent CUDA driver
- No external API keys required (all sources are free)
