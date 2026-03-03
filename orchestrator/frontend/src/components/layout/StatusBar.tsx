import { useEffect, useState } from 'react';
import { getHealth, type HealthResponse } from '../../lib/api';

function StatusDot({ status }: { status: string }) {
  const color =
    status === 'ok' || status === 'healthy'
      ? 'bg-green-500'
      : status === 'loading'
        ? 'bg-yellow-500'
        : 'bg-red-500';
  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} />;
}

export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const h = await getHealth();
        if (mounted) setHealth(h);
      } catch {
        if (mounted) setHealth(null);
      }
    };
    check();
    const iv = setInterval(check, 15000);
    return () => {
      mounted = false;
      clearInterval(iv);
    };
  }, []);

  const services = health?.services ?? {};
  const entries = Object.entries(services);

  return (
    <footer className="flex items-center gap-4 border-t bg-[hsl(var(--card))] px-4 py-1.5 text-xs text-[hsl(var(--muted-foreground))]">
      {entries.length === 0 ? (
        <span className="flex items-center gap-1.5">
          <StatusDot status="error" /> Connecting...
        </span>
      ) : (
        entries.map(([name, status]) => (
          <span key={name} className="flex items-center gap-1.5">
            <StatusDot status={status} />
            {name}
          </span>
        ))
      )}
    </footer>
  );
}
