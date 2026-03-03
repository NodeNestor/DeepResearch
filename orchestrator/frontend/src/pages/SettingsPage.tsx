import { useState, useEffect, useCallback } from 'react';
import { CheckCircle2, XCircle, Loader2, Sun, Moon, Save } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { cn } from '../lib/utils';
import { getConfig, updateConfig, type AppConfig } from '../lib/api';

type TestStatus = 'idle' | 'testing' | 'ok' | 'error';

function StatusIcon({ status }: { status: TestStatus }) {
  switch (status) {
    case 'testing':
      return <Loader2 size={14} className="animate-spin text-yellow-500" />;
    case 'ok':
      return <CheckCircle2 size={14} className="text-green-500" />;
    case 'error':
      return <XCircle size={14} className="text-red-500" />;
    default:
      return null;
  }
}

function FieldLabel({ children, status }: { children: React.ReactNode; status?: TestStatus }) {
  return (
    <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-[hsl(var(--muted-foreground))]">
      {children}
      {status && <StatusIcon status={status} />}
    </label>
  );
}

export function SettingsPage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [testStatuses, setTestStatuses] = useState<Record<string, TestStatus>>({});
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  });

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('dr-theme', JSON.stringify(theme));
  }, [theme]);

  const update = useCallback((field: keyof AppConfig, value: string | number) => {
    setConfig((prev) => (prev ? { ...prev, [field]: value } : prev));
    setSaved(false);
  }, []);

  const handleSave = useCallback(async () => {
    if (!config) return;
    setSaving(true);
    try {
      const updated = await updateConfig(config);
      setConfig(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error('Save failed:', e);
    }
    setSaving(false);
  }, [config]);

  const testEndpoint = useCallback(async (name: string, url: string) => {
    setTestStatuses((s) => ({ ...s, [name]: 'testing' }));
    try {
      // For most services, just hit the base URL
      const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
      setTestStatuses((s) => ({ ...s, [name]: resp.ok ? 'ok' : 'error' }));
    } catch {
      setTestStatuses((s) => ({ ...s, [name]: 'error' }));
    }
  }, []);

  if (loading || !config) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="animate-spin text-[hsl(var(--muted-foreground))]" size={24} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Settings</h1>
        <Button onClick={handleSave} disabled={saving} size="sm">
          {saving ? (
            <Loader2 size={14} className="mr-1.5 animate-spin" />
          ) : saved ? (
            <CheckCircle2 size={14} className="mr-1.5 text-green-400" />
          ) : (
            <Save size={14} className="mr-1.5" />
          )}
          {saved ? 'Saved' : 'Save Settings'}
        </Button>
      </div>

      {/* Appearance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Appearance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="inline-flex rounded-lg bg-[hsl(var(--muted))] p-0.5">
            <button
              onClick={() => setTheme('light')}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                theme === 'light'
                  ? 'bg-[hsl(var(--background))] text-[hsl(var(--foreground))] shadow-sm'
                  : 'text-[hsl(var(--muted-foreground))]',
              )}
            >
              <Sun size={14} /> Light
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                theme === 'dark'
                  ? 'bg-[hsl(var(--background))] text-[hsl(var(--foreground))] shadow-sm'
                  : 'text-[hsl(var(--muted-foreground))]',
              )}
            >
              <Moon size={14} /> Dark
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Extraction Model (bulk / swarm) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Extraction Model</CardTitle>
          <p className="text-xs text-[hsl(var(--muted-foreground))]">
            Small, fast model used for fact extraction and memory swarm agents.
            Any OpenAI-compatible endpoint works (vLLM, Ollama, ModelGate, etc.)
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <FieldLabel status={testStatuses['bulk']}>API URL</FieldLabel>
            <div className="flex gap-2">
              <Input
                value={config.bulk_api_url}
                onChange={(e) => update('bulk_api_url', e.target.value)}
                placeholder="http://localhost:8000/v1"
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={() => testEndpoint('bulk', config.bulk_api_url.replace('/v1', '') + '/health')}
              >
                Test
              </Button>
            </div>
          </div>
          <div>
            <FieldLabel>Model</FieldLabel>
            <Input
              value={config.bulk_model}
              onChange={(e) => update('bulk_model', e.target.value)}
              placeholder="Qwen/Qwen3.5-0.8B"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <FieldLabel>API Key</FieldLabel>
              <Input
                type="password"
                value={config.bulk_api_key}
                onChange={(e) => update('bulk_api_key', e.target.value)}
                placeholder="Optional"
              />
            </div>
            <div>
              <FieldLabel>Max Tokens</FieldLabel>
              <Input
                type="number"
                value={config.bulk_max_tokens}
                onChange={(e) => update('bulk_max_tokens', parseInt(e.target.value) || 0)}
                min={256}
                max={65536}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Synthesis Model (report writer) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Synthesis Model</CardTitle>
          <p className="text-xs text-[hsl(var(--muted-foreground))]">
            Larger model for writing the final research report. Can be a completely different
            provider — e.g., use vLLM locally for extraction but ModelGate or OpenAI for synthesis.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <FieldLabel status={testStatuses['synthesis']}>API URL</FieldLabel>
            <div className="flex gap-2">
              <Input
                value={config.synthesis_api_url}
                onChange={(e) => update('synthesis_api_url', e.target.value)}
                placeholder="http://localhost:8989/v1"
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={() => testEndpoint('synthesis', config.synthesis_api_url.replace('/v1', '') + '/health')}
              >
                Test
              </Button>
            </div>
          </div>
          <div>
            <FieldLabel>Model</FieldLabel>
            <Input
              value={config.synthesis_model}
              onChange={(e) => update('synthesis_model', e.target.value)}
              placeholder="gpt-4o / claude-sonnet-4-20250514 / Qwen/Qwen3.5-9B"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <FieldLabel>API Key</FieldLabel>
              <Input
                type="password"
                value={config.synthesis_api_key}
                onChange={(e) => update('synthesis_api_key', e.target.value)}
                placeholder="Optional"
              />
            </div>
            <div>
              <FieldLabel>Max Tokens</FieldLabel>
              <Input
                type="number"
                value={config.synthesis_max_tokens}
                onChange={(e) => update('synthesis_max_tokens', parseInt(e.target.value) || 0)}
                min={256}
                max={131072}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Services */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Services</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <FieldLabel status={testStatuses['hivemind']}>HiveMindDB</FieldLabel>
            <div className="flex gap-2">
              <Input
                value={config.hivemind_url}
                onChange={(e) => update('hivemind_url', e.target.value)}
                placeholder="http://hiveminddb:8100"
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={() => testEndpoint('hivemind', config.hivemind_url + '/api/v1/status')}
              >
                Test
              </Button>
            </div>
          </div>
          <div>
            <FieldLabel status={testStatuses['searxng']}>SearXNG</FieldLabel>
            <div className="flex gap-2">
              <Input
                value={config.searxng_url}
                onChange={(e) => update('searxng_url', e.target.value)}
                placeholder="http://searxng:8080"
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={() => testEndpoint('searxng', config.searxng_url)}
              >
                Test
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Research Defaults */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Research Defaults</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <FieldLabel>Max Depth</FieldLabel>
              <Input
                type="number"
                value={config.max_depth}
                onChange={(e) => update('max_depth', parseInt(e.target.value) || 3)}
                min={1}
                max={10}
              />
              <p className="mt-1 text-[10px] text-[hsl(var(--muted-foreground))]">
                Max research iterations (completeness may stop earlier)
              </p>
            </div>
            <div>
              <FieldLabel>Swarm Agents</FieldLabel>
              <Input
                type="number"
                value={config.swarm_agents}
                onChange={(e) => update('swarm_agents', parseInt(e.target.value) || 5)}
                min={1}
                max={20}
              />
              <p className="mt-1 text-[10px] text-[hsl(var(--muted-foreground))]">
                Parallel memory swarm agents per research run
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
