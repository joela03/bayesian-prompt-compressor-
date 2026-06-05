import { useEffect, useState } from 'react';
import Compare from './pages/Compare';
import Benchmark from './pages/Benchmark';
import Calculator from './pages/Calculator';

type TabId = 'compare' | 'benchmark' | 'calculator';

const TABS: { id: TabId; label: string }[] = [
  { id: 'compare', label: 'Live comparison' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'calculator', label: 'Cost & carbon' },
];

function readHashTab(): TabId {
  const h = window.location.hash.replace(/^#\/?/, '');
  if (h === 'benchmark' || h === 'calculator') return h;
  return 'compare';
}

export default function App() {
  const [tab, setTab] = useState<TabId>(readHashTab());

  useEffect(() => {
    const onChange = () => setTab(readHashTab());
    window.addEventListener('hashchange', onChange);
    if (!window.location.hash) window.location.hash = '#/compare';
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  return (
    <div className="min-h-full">
      <header className="border-b border-line bg-canvas/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <span className="font-mono font-semibold text-ink">prompt-compress</span>
            <span className="text-muted text-sm">structural prompt compression with safety gating</span>
          </div>
          <nav className="flex gap-1" role="tablist">
            {TABS.map(t => (
              <a
                key={t.id}
                href={`#/${t.id}`}
                role="tab"
                aria-selected={tab === t.id}
                className={
                  'px-3 py-1.5 rounded-md text-sm transition-colors ' +
                  (tab === t.id
                    ? 'bg-ink text-canvas'
                    : 'text-muted hover:text-ink')
                }
              >
                {t.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 fade-in" key={tab}>
        {tab === 'compare' && <Compare />}
        {tab === 'benchmark' && <Benchmark />}
        {tab === 'calculator' && <Calculator />}
      </main>

      <footer className="max-w-6xl mx-auto px-6 py-8 text-xs text-muted">
        Live compression backed by the local <span className="font-mono">prompt_compress</span> library ·
        backend on <span className="font-mono">localhost:8000</span>
      </footer>
    </div>
  );
}
