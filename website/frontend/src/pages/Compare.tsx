import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  DemoPrompt,
  LiveCompressionResponse,
  LiveCompressionResult,
  Summary,
} from '../types';

type Mode = 'idle' | 'running' | 'slow' | 'done' | 'error' | 'offline';

const tierColor = (tier: number | string) => {
  const t = String(tier);
  if (t === '1' || t === 'T1') return 'bg-blue-100 text-blue-800';
  if (t === '2' || t === 'T2') return 'bg-amber-100 text-amber-800';
  if (t === '3' || t === 'T3') return 'bg-emerald-100 text-emerald-800';
  return 'bg-slate-100 text-slate-800';
};

const tierLabel = (tier: number | string) => {
  const t = String(tier);
  if (t === '1') return 'T1 · Bayesian BO';
  if (t === '2') return 'T2 · TextRank';
  if (t === '3') return 'T3 · Preserved';
  return `${t}`;
};

function useCountUp(target: number, duration = 300) {
  const [value, setValue] = useState(target);
  const prevRef = useRef(target);
  useEffect(() => {
    const start = prevRef.current;
    const end = target;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(Math.round(start + (end - start) * eased));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      prevRef.current = end;
    };
  }, [target, duration]);
  return value;
}

export default function Compare() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');           // textarea content
  const [submitted, setSubmitted] = useState('');   // last prompt actually sent
  const [mode, setMode] = useState<Mode>('idle');
  const [errMsg, setErrMsg] = useState('');
  const [result, setResult] = useState<LiveCompressionResult | null>(null);
  const slowTimer = useRef<number | null>(null);

  // Load summary on mount; queue the default demo
  useEffect(() => {
    fetch('/summary.json')
      .then(r => r.json())
      .then((s: Summary) => {
        setSummary(s);
        const fitness = s.demo_prompts.find(d => d.id === 'awesome_079') ?? s.demo_prompts[0];
        if (fitness) {
          setActiveId(fitness.id);
          setDraft(fitness.original_text);
          runCompress(fitness.original_text);
        }
      })
      .catch(() => setMode('error'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runCompress = async (prompt: string) => {
    setMode('running');
    setErrMsg('');
    setSubmitted(prompt);
    if (slowTimer.current) window.clearTimeout(slowTimer.current);
    slowTimer.current = window.setTimeout(() => {
      setMode(m => (m === 'running' ? 'slow' : m));
    }, 5000);
    try {
      const r = await fetch('/api/compress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, min_similarity: 0.75 }),
      });
      if (slowTimer.current) window.clearTimeout(slowTimer.current);
      // Vite returns 500 (often with a non-JSON body) when the upstream
      // backend is dead — surface that as "offline" rather than a generic
      // error so the operator sees the actionable instruction.
      if (!r.ok) {
        setMode('offline');
        return;
      }
      const data: LiveCompressionResponse = await r.json();
      if (!data.ok) {
        setErrMsg(data.error);
        setMode('error');
        return;
      }
      setResult(data.result);
      setMode('done');
    } catch (e) {
      if (slowTimer.current) window.clearTimeout(slowTimer.current);
      setMode('offline');
    }
  };

  const onSelectDemo = (d: DemoPrompt) => {
    setActiveId(d.id);
    setDraft(d.original_text);
    runCompress(d.original_text);
  };

  const onCustomCompress = () => {
    if (!draft.trim()) return;
    setActiveId(null);
    runCompress(draft);
  };

  const activeDemo = useMemo<DemoPrompt | null>(() => {
    if (!summary || !activeId) return null;
    return summary.demo_prompts.find(d => d.id === activeId) ?? null;
  }, [summary, activeId]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Live compression</h1>
        <p className="text-muted text-sm mt-1">
          Pick a demo or paste your own system prompt. Compression runs against the local{' '}
          <span className="font-mono">prompt_compress</span> library; LLMLingua results
          are pre-computed for the four reference prompts.
        </p>
      </div>

      {/* Demo selectors */}
      <section>
        <div className="text-xs uppercase tracking-wide text-muted mb-2">Reference prompts</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
          {summary?.demo_prompts.map(d => (
            <button
              key={d.id}
              onClick={() => onSelectDemo(d)}
              className={
                'text-left rounded-md border px-3 py-2 transition-all ' +
                (activeId === d.id
                  ? 'border-ink bg-white shadow-card'
                  : 'border-line bg-white/60 hover:bg-white hover:border-ink/40')
              }
            >
              <div className="text-[11px] uppercase tracking-wider text-muted">{d.category.replace('_', ' ')}</div>
              <div className="text-sm font-medium mt-0.5">{d.label}</div>
              <div className="text-xs text-muted mt-1 line-clamp-2">{d.caption}</div>
            </button>
          ))}
        </div>
      </section>

      {/* Input box */}
      <section className="rounded-lg border border-line bg-white shadow-card p-5">
        <label className="block text-sm font-medium mb-2">Your prompt</label>
        <textarea
          value={draft}
          onChange={e => setDraft(e.target.value)}
          rows={6}
          placeholder="Paste your own system prompt to compress live…"
          className="w-full font-mono text-sm rounded-md border border-line p-3 focus:outline-none focus:border-ink/60"
        />
        <div className="flex items-center justify-between mt-3">
          <div className="text-xs text-muted">
            {draft.split(/\s+/).filter(Boolean).length} words
            {activeDemo ? ` · viewing ${activeDemo.label}` : draft ? ' · custom prompt' : ''}
          </div>
          <div className="flex items-center gap-3">
            {(mode === 'running' || mode === 'slow') && (
              <span className="text-xs text-muted flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-ours animate-pulse" />
                {mode === 'slow' ? 'Taking longer than usual…' : 'Running…'}
              </span>
            )}
            <button
              onClick={onCustomCompress}
              disabled={mode === 'running' || mode === 'slow' || !draft.trim()}
              className="px-4 py-1.5 text-sm rounded-md bg-ours text-canvas font-medium disabled:opacity-50"
            >
              Compress
            </button>
          </div>
        </div>
        {mode === 'error' && (
          <div className="mt-3 text-sm text-fail">Error: {errMsg}</div>
        )}
        {mode === 'offline' && (
          <div className="mt-3 text-sm text-fail">
            Backend offline.  with{'Start the API '}
            <span className="font-mono">cd website/backend && uvicorn main:app --port 8000</span>{' '}
            and refresh.
          </div>
        )}
      </section>

      {/* Results columns */}
      {result && mode !== 'offline' && (
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4 fade-in">
          <Column
            title="Original"
            text={submitted}
            tone="neutral"
            footer={`${result.original_tokens} tokens`}
          />
          <OursColumn result={result} demo={activeDemo} />
          <LinguaColumn demo={activeDemo} ourJudge={activeDemo?.ours.judge_score ?? null} />
        </section>
      )}

      {/* Summary line */}
      {result && mode === 'done' && (
        <ResultSummary result={result} demo={activeDemo} />
      )}
    </div>
  );
}

// Returns (oursClass, linguaClass) for the judge-score number colouring.
// |diff| < 1 = essentially tied → both neutral; otherwise the winner gets its
// system colour, the loser stays neutral.
function judgeColours(ours: number | null | undefined, lingua: number | null | undefined): [string, string] {
  if (ours == null || lingua == null) return ['text-ink', 'text-ink'];
  const diff = ours - lingua;
  if (diff >= 1) return ['text-ours font-medium', 'text-ink'];
  if (diff <= -1) return ['text-ink', 'text-lingua font-medium'];
  return ['text-ink', 'text-ink'];
}

function Column({
  title,
  text,
  tone,
  footer,
  headerRight,
}: {
  title: string;
  text: string;
  tone: 'neutral' | 'ours' | 'lingua';
  footer?: React.ReactNode;
  headerRight?: React.ReactNode;
}) {
  const ring =
    tone === 'ours'
      ? 'border-ours/30'
      : tone === 'lingua'
      ? 'border-lingua/30'
      : 'border-line';
  return (
    <div className={'rounded-lg border bg-white shadow-card flex flex-col ' + ring}>
      <div className="flex items-center justify-between px-4 py-2 border-b border-line">
        <span className="text-xs uppercase tracking-wide text-muted">{title}</span>
        {headerRight}
      </div>
      <div className="p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-ink flex-1">
        {text || <span className="text-muted italic">(empty)</span>}
      </div>
      {footer && (
        <div className="px-4 py-2 text-xs text-muted border-t border-line">{footer}</div>
      )}
    </div>
  );
}

function OursColumn({ result, demo }: { result: LiveCompressionResult; demo: DemoPrompt | null }) {
  const savedAnim = useCountUp(result.tokens_saved);
  const badge = result.safe_to_use
    ? <span className="text-[11px] px-2 py-0.5 rounded-full bg-ours/10 text-ours font-medium">Safe to deploy</span>
    : <span className="text-[11px] px-2 py-0.5 rounded-full bg-fail/10 text-fail font-medium">Validator rejected</span>;

  // Judge score is a benchmark metric — only shown for the four reference
  // demos where it's pre-computed. Live custom prompts have no judge score.
  const ourJudge = demo?.ours.judge_score ?? null;
  const linguaJudge = demo?.llmlingua.judge_score ?? null;
  const [oursCls] = judgeColours(ourJudge, linguaJudge);

  return (
    <Column
      title="Ours (live)"
      tone="ours"
      text={result.compressed_text}
      headerRight={
        <div className="flex items-center gap-2">
          <span className={'text-[11px] px-2 py-0.5 rounded-full font-medium ' + tierColor(result.tier)}>
            {tierLabel(result.tier)}
          </span>
          {badge}
        </div>
      }
      footer={
        <div className="space-y-0.5">
          <div>
            {`${result.compressed_tokens} tokens · saved ${savedAnim} (${(result.compression_ratio * 100).toFixed(1)}%) · `}
            {`sim ${result.semantic_similarity.toFixed(3)} · density ${result.density.toFixed(2)} · `}
            {`${result.time_seconds.toFixed(2)}s`}
          </div>
          {ourJudge != null && (
            <div>
              LLM judge score:{' '}
              <span className={oursCls}>{ourJudge.toFixed(1)} / 100</span>
            </div>
          )}
        </div>
      }
    />
  );
}

function LinguaColumn({ demo, ourJudge }: { demo: DemoPrompt | null; ourJudge: number | null }) {
  if (!demo) {
    return (
      <Column
        title="LLMLingua"
        tone="lingua"
        text="LLMLingua comparison is shown for the four reference examples. Custom prompts run on our system only."
      />
    );
  }
  const l = demo.llmlingua;
  const [, linguaCls] = judgeColours(ourJudge, l.judge_score ?? null);
  return (
    <Column
      title="LLMLingua (reference)"
      tone="lingua"
      text={l.compressed_text}
      headerRight={
        <span className={
          'text-[11px] px-2 py-0.5 rounded-full font-medium ' +
          (l.persona_preserved ? 'bg-emerald-100 text-emerald-800' : 'bg-fail/10 text-fail')
        }>
          {l.persona_preserved ? 'Persona kept' : 'Persona lost'}
        </span>
      }
      footer={
        <div className="space-y-0.5">
          <div>
            {`${(l.compression * 100).toFixed(1)}% reduction · output sim ${l.output_similarity.toFixed(3)}`}
          </div>
          {l.judge_score != null && (
            <div>
              LLM judge score:{' '}
              <span className={linguaCls}>{l.judge_score.toFixed(1)} / 100</span>
            </div>
          )}
        </div>
      }
    />
  );
}

function ResultSummary({ result, demo }: { result: LiveCompressionResult; demo: DemoPrompt | null }) {
  const failures = result.validator_failures.length
    ? `failed validator: ${result.validator_failures.join(', ')}`
    : '';

  // Judge comparison line only appears for the reference demos. For live
  // custom prompts we don't have a judge score — it's a benchmark metric,
  // not a real-time signal, so we don't fabricate one.
  const showJudge =
    demo && demo.ours.judge_score != null && demo.llmlingua.judge_score != null;
  const ourJudge = demo?.ours.judge_score ?? null;
  const linguaJudge = demo?.llmlingua.judge_score ?? null;
  const [oursCls, linguaCls] = judgeColours(ourJudge, linguaJudge);

  return (
    <section className="rounded-lg border border-line bg-white shadow-card px-5 py-4 space-y-2">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="text-sm">
          Saved <span className="font-semibold">{result.tokens_saved}</span> tokens
          {' '}(
          <span className="font-semibold">{(result.compression_ratio * 100).toFixed(1)}%</span> compression
          ) in <span className="font-semibold">{result.time_seconds.toFixed(2)}</span>s.
        </div>
        {result.safe_to_use ? (
          <span className="text-[11px] px-2 py-1 rounded-md bg-ours/10 text-ours font-medium">
            ✓ Safe to deploy
          </span>
        ) : (
          <span className="text-[11px] px-2 py-1 rounded-md bg-fail/10 text-fail font-medium">
            ✗ {failures || 'Validator rejected — returning original'}
          </span>
        )}
        <span className="text-xs text-muted ml-auto">
          Tier {result.tier} · α {result.alpha ?? 'n/a'} · {result.n_evaluations ?? 0} evals
        </span>
      </div>
      {showJudge && (
        <div className="text-sm text-muted">
          LLM judge: <span className={oursCls}>{ourJudge!.toFixed(1)}</span>
          {' '}vs LLMLingua's <span className={linguaCls}>{linguaJudge!.toFixed(1)}</span>
        </div>
      )}
    </section>
  );
}
