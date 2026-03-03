import { useState, useCallback, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Globe, Database, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Textarea } from '../components/ui/Textarea';
import { Badge } from '../components/ui/Badge';
import { cn } from '../lib/utils';
import { startResearch, getSession, connectWs, type ResearchProgress, type ResearchSession } from '../lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'progress';
  content: string;
  timestamp: Date;
  progress?: ResearchProgress;
  session?: ResearchSession;
}

function ProgressMessage({ progress }: { progress: ResearchProgress }) {
  const phaseIcon: Record<string, React.ReactNode> = {
    'memory_swarm': <Brain size={14} />,
    'web_search': <Globe size={14} />,
    'storing': <Database size={14} />,
    'synthesis': <Sparkles size={14} />,
  };

  const icon = Object.entries(phaseIcon).find(([k]) => progress.phase.toLowerCase().includes(k))?.[1]
    ?? <Loader2 size={14} className="animate-spin" />;

  return (
    <div className="flex items-start gap-3 rounded-lg bg-[hsl(var(--muted))] px-4 py-3 text-sm">
      <span className="mt-0.5 text-[hsl(var(--muted-foreground))]">{icon}</span>
      <div className="flex-1 min-w-0">
        <div className="font-medium">{progress.phase}</div>
        <div className="text-[hsl(var(--muted-foreground))] truncate">{progress.message}</div>
        <div className="mt-1.5 flex flex-wrap gap-2">
          {progress.facts_so_far > 0 && (
            <Badge variant="outline">{progress.facts_so_far} facts</Badge>
          )}
          {progress.entities_so_far > 0 && (
            <Badge variant="outline">{progress.entities_so_far} entities</Badge>
          )}
          {progress.urls_processed > 0 && (
            <Badge variant="outline">
              {progress.urls_processed}/{progress.urls_total} URLs
            </Badge>
          )}
          {progress.depth > 0 && (
            <Badge variant="outline">depth {progress.depth}</Badge>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultMessage({ session }: { session: ResearchSession }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium text-green-400">
        <CheckCircle2 size={16} />
        Research Complete
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        <Badge>{session.facts_discovered} facts</Badge>
        <Badge>{session.entities_discovered} entities</Badge>
        <Badge>{session.urls_processed} sources</Badge>
        <Badge>depth {session.depth_reached}</Badge>
      </div>

      {session.tokens && (
        <div className="text-xs text-[hsl(var(--muted-foreground))]">
          {session.tokens.llm_total_tokens.toLocaleString()} tokens used
          {' / '}
          {session.tokens.llm_requests} LLM calls
          {session.tokens.web_pages_fetched > 0 && ` / ${session.tokens.web_pages_fetched} pages fetched`}
        </div>
      )}

      {session.report && (
        <div className="mt-3 rounded-lg border bg-[hsl(var(--card))] p-4">
          <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap text-sm leading-relaxed">
            {session.report}
          </div>
        </div>
      )}
    </div>
  );
}

export function ResearchPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'quick' | 'deep'>('deep');
  const [depth, setDepth] = useState(3);
  const [isResearching, setIsResearching] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<number | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const addMessage = useCallback((msg: Omit<Message, 'id' | 'timestamp'>) => {
    setMessages((prev) => [...prev, { ...msg, id: crypto.randomUUID(), timestamp: new Date() }]);
  }, []);

  const updateLastProgress = useCallback((progress: ResearchProgress) => {
    setMessages((prev) => {
      // Replace the last progress message or add new one
      const last = prev[prev.length - 1];
      if (last?.role === 'progress') {
        return [...prev.slice(0, -1), { ...last, content: progress.message, progress }];
      }
      return [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'progress' as const,
          content: progress.message,
          timestamp: new Date(),
          progress,
        },
      ];
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    const query = input.trim();
    if (!query || isResearching) return;

    setInput('');
    setIsResearching(true);
    addMessage({ role: 'user', content: query });

    try {
      const { session_id } = await startResearch(query, mode === 'quick' ? 1 : depth);

      addMessage({
        role: 'assistant',
        content: `Starting research on: "${query}"`,
      });

      // WebSocket for live updates
      wsRef.current?.close();
      wsRef.current = connectWs(session_id, (p) => updateLastProgress(p));

      // Poll for completion
      pollRef.current = window.setInterval(async () => {
        try {
          const sess = await getSession(session_id);
          if (sess.status !== 'running') {
            setIsResearching(false);
            if (pollRef.current) clearInterval(pollRef.current);
            wsRef.current?.close();

            // Remove trailing progress message, add result
            setMessages((prev) => {
              const filtered = prev.filter((m) => m.role !== 'progress');
              return [
                ...filtered,
                {
                  id: crypto.randomUUID(),
                  role: 'assistant' as const,
                  content: 'Research complete',
                  timestamp: new Date(),
                  session: sess,
                },
              ];
            });
          }
        } catch { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setIsResearching(false);
      addMessage({
        role: 'assistant',
        content: `Error: ${e instanceof Error ? e.message : 'Research failed'}`,
      });
    }
  }, [input, isResearching, addMessage, updateLastProgress]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-[hsl(var(--muted))]">
                <Brain size={32} className="text-[hsl(var(--muted-foreground))]" />
              </div>
              <h2 className="mb-2 text-xl font-semibold">DeepResearch</h2>
              <p className="max-w-md text-sm text-[hsl(var(--muted-foreground))]">
                Ask any research question. The system will search the web and your knowledge graph
                simultaneously, using parallel AI agents to build a comprehensive answer.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {[
                  'Latest advances in quantum computing',
                  'Compare React vs Svelte in 2026',
                  'History of transformer architectures',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="rounded-full border px-3 py-1.5 text-xs text-[hsl(var(--muted-foreground))] transition-colors hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--accent-foreground))]"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-br-md bg-[hsl(var(--primary))] px-4 py-2.5 text-sm text-[hsl(var(--primary-foreground))]">
                    {msg.content}
                  </div>
                </div>
              );
            }

            if (msg.role === 'progress' && msg.progress) {
              return (
                <div key={msg.id} className="max-w-[85%]">
                  <ProgressMessage progress={msg.progress} />
                </div>
              );
            }

            // assistant message
            return (
              <div key={msg.id} className="max-w-[85%]">
                {msg.session ? (
                  <ResultMessage session={msg.session} />
                ) : msg.content.startsWith('Error:') ? (
                  <div className="flex items-center gap-2 rounded-lg bg-[hsl(var(--destructive))/0.1] px-4 py-3 text-sm text-red-400">
                    <AlertCircle size={16} />
                    {msg.content}
                  </div>
                ) : (
                  <div className="rounded-2xl rounded-bl-md bg-[hsl(var(--muted))] px-4 py-2.5 text-sm">
                    {msg.content}
                  </div>
                )}
              </div>
            );
          })}

          {isResearching && messages[messages.length - 1]?.role !== 'progress' && (
            <div className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))]">
              <Loader2 size={14} className="animate-spin" />
              Researching...
            </div>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="border-t p-4">
        <div className="mx-auto max-w-3xl space-y-3">
          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg bg-[hsl(var(--muted))] p-0.5">
              <button
                onClick={() => setMode('quick')}
                className={cn(
                  'px-3 py-1 rounded-md text-xs font-medium transition-colors',
                  mode === 'quick'
                    ? 'bg-[hsl(var(--background))] text-[hsl(var(--foreground))] shadow'
                    : 'text-[hsl(var(--muted-foreground))]',
                )}
              >
                Quick
              </button>
              <button
                onClick={() => setMode('deep')}
                className={cn(
                  'px-3 py-1 rounded-md text-xs font-medium transition-colors',
                  mode === 'deep'
                    ? 'bg-[hsl(var(--background))] text-[hsl(var(--foreground))] shadow'
                    : 'text-[hsl(var(--muted-foreground))]',
                )}
              >
                Deep
              </button>
            </div>

            {mode === 'deep' && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-[hsl(var(--muted-foreground))] whitespace-nowrap">Depth</span>
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  className="h-1.5 w-24 cursor-pointer appearance-none rounded-full bg-[hsl(var(--secondary))] [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[hsl(var(--primary))] [&::-webkit-slider-thumb]:shadow"
                />
                <span className="text-xs text-[hsl(var(--muted-foreground))] tabular-nums w-4 text-right">{depth}</span>
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                mode === 'quick'
                  ? 'Ask a quick research question...'
                  : 'Ask the agent to research something in depth...'
              }
              className="min-h-[44px] max-h-[200px] resize-none"
              rows={1}
              disabled={isResearching}
            />
            <Button onClick={handleSubmit} disabled={!input.trim() || isResearching} size="icon">
              {isResearching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
