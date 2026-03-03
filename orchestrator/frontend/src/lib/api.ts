const BASE = '/api';

export interface TokenStats {
  web_pages_fetched: number;
  web_chars_ingested: number;
  web_tokens_estimated: number;
  llm_prompt_tokens: number;
  llm_completion_tokens: number;
  llm_total_tokens: number;
  llm_requests: number;
  llm_failed_requests: number;
  phase_tokens: Record<string, { prompt: number; completion: number; requests: number }>;
  source_counts: Record<string, number>;
}

export interface ResearchSession {
  session_id: string;
  query: string;
  facts_discovered: number;
  entities_discovered: number;
  urls_processed: number;
  depth_reached: number;
  report: string;
  tokens: TokenStats;
  status?: string;
}

export interface SessionSummary {
  session_id: string;
  query: string;
  facts_discovered: number;
  entities_discovered: number;
  started_at: string | null;
  finished_at: string | null;
  status: string;
}

export interface HealthResponse {
  status: string;
  services: Record<string, string>;
}

export interface ResearchProgress {
  phase: string;
  message: string;
  facts_so_far: number;
  entities_so_far: number;
  urls_processed: number;
  urls_total: number;
  depth: number;
}

export interface AppConfig {
  bulk_provider: string;
  bulk_model: string;
  bulk_api_url: string;
  bulk_api_key: string;
  bulk_max_tokens: number;
  synthesis_provider: string;
  synthesis_model: string;
  synthesis_api_url: string;
  synthesis_api_key: string;
  synthesis_max_tokens: number;
  hivemind_url: string;
  searxng_url: string;
  max_depth: number;
  swarm_agents: number;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json();
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson(`${BASE}/health`);
}

export async function getConfig(): Promise<AppConfig> {
  return fetchJson(`${BASE}/config`);
}

export async function updateConfig(config: Partial<AppConfig>): Promise<AppConfig> {
  return fetchJson(`${BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function startResearch(
  query: string,
  depth?: number,
  sources?: string[],
): Promise<{ session_id: string; status: string }> {
  return fetchJson(`${BASE}/research/async`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, depth, sources }),
  });
}

export async function getSession(sessionId: string): Promise<ResearchSession> {
  return fetchJson(`${BASE}/research/${sessionId}`);
}

export async function listSessions(): Promise<SessionSummary[]> {
  return fetchJson(`${BASE}/sessions`);
}

export function connectWs(
  sessionId: string,
  onProgress: (p: ResearchProgress) => void,
  onClose?: () => void,
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ session_id: sessionId }));
  };
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.phase) onProgress(data);
    } catch {
      /* ignore non-JSON */
    }
  };
  ws.onclose = () => onClose?.();
  return ws;
}
