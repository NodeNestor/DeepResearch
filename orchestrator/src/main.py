"""FastAPI application — REST API, WebSocket progress, static frontend, and MCP server entry."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings, get_runtime_config, update_runtime_config
from .models import ResearchPhase, ResearchProgress, ResearchSession, SourceType
from .research import quick_search, run_research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ── Pydantic request/response models ─────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    depth: int | None = None
    sources: list[str] | None = None


class TokenStatsResponse(BaseModel):
    web_pages_fetched: int = 0
    web_chars_ingested: int = 0
    web_tokens_estimated: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_requests: int = 0
    llm_failed_requests: int = 0
    phase_tokens: dict = {}
    source_counts: dict = {}  # {source_type: count}


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    facts_discovered: int
    entities_discovered: int
    urls_processed: int
    depth_reached: int
    report: str
    tokens: TokenStatsResponse = TokenStatsResponse()


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


# ══════════════════════════════════════════════════════════════════════
# ConcurrentResearchManager — tracks all active and completed sessions
# ══════════════════════════════════════════════════════════════════════

class ConcurrentResearchManager:
    """Manages concurrent research sessions with progress tracking."""

    def __init__(self) -> None:
        self.sessions: dict[str, ResearchSession] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.progress_queues: dict[str, asyncio.Queue[ResearchProgress]] = {}
        self.ws_clients: dict[str, list[WebSocket]] = {}

    def get_session(self, session_id: str) -> ResearchSession | None:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[ResearchSession]:
        return list(self.sessions.values())

    def is_running(self, session_id: str) -> bool:
        task = self.tasks.get(session_id)
        return task is not None and not task.done()

    async def start_research(
        self,
        query: str,
        depth: int | None = None,
        sources: list[SourceType] | None = None,
        blocking: bool = False,
    ) -> ResearchSession:
        """Start a research session. If blocking=True, waits for completion."""
        progress_queue: asyncio.Queue[ResearchProgress] = asyncio.Queue()

        session = ResearchSession(query=query)
        self.sessions[session.id] = session
        self.progress_queues[session.id] = progress_queue

        if blocking:
            session = await run_research(
                query=query,
                settings=settings,
                progress=progress_queue,
                depth=depth,
                sources=sources,
                session=session,
            )
            # Drain progress queue to WebSocket clients
            await self._drain_progress(session.id)
            return session

        # Non-blocking — run in background
        async def _run():
            try:
                await run_research(
                    query=query,
                    settings=settings,
                    progress=progress_queue,
                    depth=depth,
                    sources=sources,
                    session=session,
                )
            except Exception as e:
                log.error("Background research failed: %s", e)
            finally:
                await self._drain_progress(session.id)

        task = asyncio.create_task(_run())
        self.tasks[session.id] = task

        # Start progress forwarding task
        asyncio.create_task(self._forward_progress(session.id))

        return session

    async def _forward_progress(self, session_id: str) -> None:
        """Forward progress updates from queue to WebSocket clients."""
        queue = self.progress_queues.get(session_id)
        if not queue:
            return

        while True:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=1.0)
                for ws in self.ws_clients.get(session_id, []):
                    try:
                        await ws.send_json(asdict(update))
                    except Exception:
                        pass

                # Stop forwarding when research is complete
                if update.phase == ResearchPhase.COMPLETE:
                    break
            except asyncio.TimeoutError:
                # Check if the task is done
                task = self.tasks.get(session_id)
                if task and task.done():
                    break
            except Exception:
                break

    async def _drain_progress(self, session_id: str) -> None:
        """Drain remaining progress updates to WebSocket clients."""
        queue = self.progress_queues.get(session_id)
        if not queue:
            return

        while True:
            try:
                update = queue.get_nowait()
                for ws in self.ws_clients.get(session_id, []):
                    try:
                        await ws.send_json(asdict(update))
                    except Exception:
                        pass
            except asyncio.QueueEmpty:
                break

    def subscribe_ws(self, session_id: str, ws: WebSocket) -> None:
        self.ws_clients.setdefault(session_id, []).append(ws)

    def unsubscribe_ws(self, ws: WebSocket) -> None:
        for clients in self.ws_clients.values():
            if ws in clients:
                clients.remove(ws)


# Global manager instance
manager = ConcurrentResearchManager()


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("DeepResearch orchestrator starting")
    log.info("  SearXNG: %s", settings.searxng_url)
    log.info("  HiveMindDB: %s", settings.hivemind_url)
    log.info("  Bulk model: %s @ %s", settings.bulk_model, settings.bulk_api_url)
    log.info("  Synthesis model: %s @ %s", settings.synthesis_model, settings.synthesis_api_url)
    if STATIC_DIR.exists():
        log.info("  Frontend: serving from %s", STATIC_DIR)
    else:
        log.info("  Frontend: not found at %s (API-only mode)", STATIC_DIR)
    yield
    log.info("DeepResearch orchestrator shutting down")


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepResearch",
    description="Hyper-parallel AI research agent with persistent knowledge graph",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_to_response(session: ResearchSession) -> ResearchResponse:
    t = session.tokens
    return ResearchResponse(
        session_id=session.id,
        query=session.query,
        facts_discovered=session.facts_discovered,
        entities_discovered=session.entities_discovered,
        urls_processed=session.urls_processed,
        depth_reached=session.depth_reached,
        report=session.report,
        tokens=TokenStatsResponse(
            web_pages_fetched=t.web_pages_fetched,
            web_chars_ingested=t.web_chars_ingested,
            web_tokens_estimated=t.web_tokens_estimated,
            llm_prompt_tokens=t.llm_prompt_tokens,
            llm_completion_tokens=t.llm_completion_tokens,
            llm_total_tokens=t.llm_total_tokens,
            llm_requests=t.llm_requests,
            llm_failed_requests=t.llm_failed_requests,
            phase_tokens=t.phase_tokens,
            source_counts=t.source_counts,
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# API routes (all under /api/)
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check with service status."""
    import httpx

    services = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in [
            ("searxng", settings.searxng_url),
            ("hiveminddb", settings.hivemind_url + "/api/v1/status"),
            ("vllm", settings.bulk_api_url.replace("/v1", "") + "/health"),
        ]:
            try:
                resp = await client.get(url)
                services[name] = "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
            except Exception:
                services[name] = "unreachable"

    return HealthResponse(status="ok", services=services)


@app.get("/api/config")
async def get_config():
    """Get current runtime configuration."""
    return get_runtime_config()


@app.put("/api/config")
async def put_config(request: Request):
    """Update runtime configuration (partial updates supported)."""
    body = await request.json()
    updated = update_runtime_config(body)
    return updated


@app.post("/api/research", response_model=ResearchResponse)
async def api_start_research(req: ResearchRequest):
    """Start a full research pipeline. Returns when complete."""
    source_types = None
    if req.sources:
        source_types = [SourceType(s) for s in req.sources]

    session = await manager.start_research(
        query=req.query,
        depth=req.depth,
        sources=source_types,
        blocking=True,
    )

    return _session_to_response(session)


@app.post("/api/research/async")
async def api_start_research_async(req: ResearchRequest):
    """Start research in background, returns session_id immediately."""
    source_types = None
    if req.sources:
        source_types = [SourceType(s) for s in req.sources]

    session = await manager.start_research(
        query=req.query,
        depth=req.depth,
        sources=source_types,
        blocking=False,
    )

    return {
        "session_id": session.id,
        "status": "started",
        "query": req.query,
    }


@app.get("/api/research/{session_id}")
async def api_get_session(session_id: str):
    """Get the result of a research session."""
    session = manager.get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    resp = _session_to_response(session)
    running = manager.is_running(session_id)
    return {
        **resp.model_dump(),
        "status": "running" if running else "complete",
    }


@app.get("/api/sessions")
async def api_list_sessions():
    """List all research sessions."""
    return [
        {
            "session_id": s.id,
            "query": s.query,
            "facts_discovered": s.facts_discovered,
            "entities_discovered": s.entities_discovered,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "finished_at": s.finished_at.isoformat() if s.finished_at else None,
            "status": "running" if manager.is_running(s.id) else "complete",
        }
        for s in manager.list_sessions()
    ]


# ── Legacy routes (no /api prefix) for backwards compat ──────────────

@app.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health_legacy():
    return await health()


@app.post("/research", response_model=ResearchResponse, include_in_schema=False)
async def start_research_legacy(req: ResearchRequest):
    return await api_start_research(req)


@app.post("/research/async", include_in_schema=False)
async def start_research_async_legacy(req: ResearchRequest):
    return await api_start_research_async(req)


@app.get("/research/{session_id}", include_in_schema=False)
async def get_session_legacy(session_id: str):
    return await api_get_session(session_id)


@app.get("/sessions", include_in_schema=False)
async def list_sessions_legacy():
    return await api_list_sessions()


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_progress(ws: WebSocket):
    """WebSocket for live research progress updates.

    Client sends: {"session_id": "abc123"} to subscribe to a session.
    """
    await ws.accept()
    log.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                session_id = msg.get("session_id")
                if session_id:
                    manager.subscribe_ws(session_id, ws)
                    await ws.send_json({"subscribed": session_id})
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
        manager.unsubscribe_ws(ws)


# ── Static frontend (SPA) ────────────────────────────────────────────
# Mount AFTER all API routes so /api/* takes priority.
# Serve index.html for any non-API, non-file path (SPA client-side routing).

if STATIC_DIR.exists():
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Try to serve the exact file first (JS, CSS, images, etc.)
        file_path = STATIC_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fall back to index.html for SPA routes
        return FileResponse(STATIC_DIR / "index.html")


# ── MCP mode entry ────────────────────────────────────────────────────

def run_mcp():
    """Entry point for MCP server mode (stdio)."""
    from .mcp.server import run_mcp_server
    asyncio.run(run_mcp_server())


# ── CLI entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        run_mcp()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
