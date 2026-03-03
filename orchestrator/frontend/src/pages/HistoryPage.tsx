import { useEffect, useState, useCallback } from 'react';
import { Clock, FileText, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { timeAgo } from '../lib/utils';
import { listSessions, getSession, type SessionSummary, type ResearchSession } from '../lib/api';

export function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selected, setSelected] = useState<ResearchSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    listSessions()
      .then((s) => {
        if (mounted) setSessions(s);
      })
      .catch(() => {})
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  const handleSelect = useCallback(async (sessionId: string) => {
    setDetailLoading(true);
    try {
      const sess = await getSession(sessionId);
      setSelected(sess);
    } catch { /* ignore */ }
    setDetailLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="animate-spin text-[hsl(var(--muted-foreground))]" size={24} />
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Session list */}
      <div className="w-[320px] shrink-0 overflow-y-auto border-r p-4">
        <h2 className="mb-4 text-sm font-semibold text-[hsl(var(--muted-foreground))]">
          Research History
        </h2>
        {sessions.length === 0 ? (
          <p className="text-sm text-[hsl(var(--muted-foreground))]">No research sessions yet.</p>
        ) : (
          <div className="space-y-2">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                onClick={() => handleSelect(s.session_id)}
                className={`w-full rounded-lg border p-3 text-left transition-colors hover:bg-[hsl(var(--accent))] ${
                  selected?.session_id === s.session_id ? 'border-[hsl(var(--ring))] bg-[hsl(var(--accent))]' : ''
                }`}
              >
                <div className="mb-1 truncate text-sm font-medium">{s.query}</div>
                <div className="flex items-center gap-2 text-xs text-[hsl(var(--muted-foreground))]">
                  <Badge variant="outline" className="text-[10px]">
                    {s.status === 'running' ? (
                      <span className="flex items-center gap-1">
                        <Loader2 size={10} className="animate-spin" /> Running
                      </span>
                    ) : (
                      s.status
                    )}
                  </Badge>
                  <span>{s.facts_discovered} facts</span>
                  {s.started_at && (
                    <>
                      <span className="text-[hsl(var(--border))]">|</span>
                      <Clock size={10} />
                      <span>{timeAgo(s.started_at)}</span>
                    </>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Detail view */}
      <div className="flex-1 overflow-y-auto p-6">
        {detailLoading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="animate-spin text-[hsl(var(--muted-foreground))]" size={24} />
          </div>
        ) : selected ? (
          <div className="mx-auto max-w-3xl space-y-4">
            <h1 className="text-lg font-semibold">{selected.query}</h1>

            <div className="flex flex-wrap gap-2">
              <Badge>{selected.facts_discovered} facts</Badge>
              <Badge>{selected.entities_discovered} entities</Badge>
              <Badge>{selected.urls_processed} sources</Badge>
              <Badge>depth {selected.depth_reached}</Badge>
            </div>

            {selected.tokens && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Token Usage</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-y-1 text-sm sm:grid-cols-4">
                    <div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">Total Tokens</div>
                      <div className="font-medium">{selected.tokens.llm_total_tokens.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">LLM Requests</div>
                      <div className="font-medium">{selected.tokens.llm_requests}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">Pages Fetched</div>
                      <div className="font-medium">{selected.tokens.web_pages_fetched}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">Failed</div>
                      <div className="font-medium">{selected.tokens.llm_failed_requests}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {selected.report && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <FileText size={14} /> Report
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap text-sm leading-relaxed">
                    {selected.report}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-[hsl(var(--muted-foreground))]">
            <FileText size={32} className="mb-2 opacity-50" />
            <p className="text-sm">Select a session to view details</p>
          </div>
        )}
      </div>
    </div>
  );
}
