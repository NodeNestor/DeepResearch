import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { StatusBar } from './StatusBar';

export function Shell() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen flex-col">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
      <StatusBar />
    </div>
  );
}
