import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ScatterPoint, Summary } from '../types';

export default function Benchmark() {
  const [summary, setSummary] = useState<Summary | null>(null);
  useEffect(() => {
    fetch('/summary.json').then(r => r.json()).then(setSummary).catch(() => {});
  }, []);

  if (!summary) return <div className="text-muted text-sm">Loading benchmark…</div>;

  const h = summary.headline;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Matched-subset benchmark</h1>
        <p className="text-muted text-sm mt-1">
          {summary.metadata.n_matched_subset} prompts where both systems successfully compressed and produced
          probing responses, out of {summary.metadata.n_prompts_total} total.
          α = {summary.metadata.alpha}, validator threshold {summary.metadata.validator_threshold}.
        </p>
      </div>

      {/* Headline cards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <HeadlineCard
          label="Compression"
          ours={`${(h.ours.compression * 100).toFixed(1)}%`}
          lingua={`${(h.llmlingua.compression * 100).toFixed(1)}%`}
          higherIsBetter
          numericOurs={h.ours.compression}
          numericLingua={h.llmlingua.compression}
        />
        <HeadlineCard
          label="LLM judge (0–100)"
          ours={h.ours.judge_score.toFixed(2)}
          lingua={h.llmlingua.judge_score.toFixed(2)}
          higherIsBetter
          numericOurs={h.ours.judge_score}
          numericLingua={h.llmlingua.judge_score}
        />
        <HeadlineCard
          label="Persona preserved"
          ours={`${(h.ours.persona_preserved * 100).toFixed(0)}%`}
          lingua={`${(h.llmlingua.persona_preserved * 100).toFixed(0)}%`}
          higherIsBetter
          numericOurs={h.ours.persona_preserved}
          numericLingua={h.llmlingua.persona_preserved}
        />
        <HeadlineCard
          label="Compression efficiency"
          ours={h.ours.compression_efficiency.toFixed(3)}
          lingua={h.llmlingua.compression_efficiency.toFixed(3)}
          higherIsBetter
          numericOurs={h.ours.compression_efficiency}
          numericLingua={h.llmlingua.compression_efficiency}
          hint="compression × output_similarity"
        />
      </section>

      {/* Scatter */}
      <section className="rounded-lg border border-line bg-white shadow-card p-5">
        <div className="flex items-end justify-between mb-3">
          <div>
            <h2 className="text-base font-semibold">Compression vs output quality</h2>
            <p className="text-xs text-muted mt-0.5">
              One point per matched prompt. Quality floor at 0.85 dashed.
              <span className="ml-3 inline-flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-ours" /> ours</span>
              <span className="ml-3 inline-flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-lingua" /> LLMLingua</span>
            </p>
          </div>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 16, bottom: 24, left: 0 }}>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="2 4" />
              <XAxis
                type="number"
                dataKey="x"
                name="compression"
                domain={[0, 1]}
                tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                label={{ value: 'compression ratio', position: 'insideBottom', offset: -10, style: { fontSize: 11, fill: '#64748b' } }}
                tick={{ fontSize: 11, fill: '#64748b' }}
                stroke="#cbd5e1"
              />
              <YAxis
                type="number"
                dataKey="y"
                name="output similarity"
                domain={[0, 1]}
                tickFormatter={v => v.toFixed(2)}
                label={{ value: 'output similarity', angle: -90, position: 'insideLeft', offset: 12, style: { fontSize: 11, fill: '#64748b' } }}
                tick={{ fontSize: 11, fill: '#64748b' }}
                stroke="#cbd5e1"
              />
              <ReferenceLine y={0.85} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: 'quality floor', fontSize: 10, fill: '#64748b', position: 'insideTopRight' }} />
              <Tooltip content={<ScatterTooltip />} cursor={{ stroke: '#94a3b8', strokeWidth: 1, strokeDasharray: '4 4' }} />
              <Scatter
                name="Ours"
                data={summary.scatter_data.map(d => ({ x: d.ours.compression, y: d.ours.output_similarity, label: d.label, system: 'ours', sim: d.ours.output_similarity, compression: d.ours.compression, persona: d.ours.persona_preserved }))}
                fill="#2d5a1f"
              />
              <Scatter
                name="LLMLingua"
                data={summary.scatter_data.map(d => ({ x: d.llmlingua.compression, y: d.llmlingua.output_similarity, label: d.label, system: 'llmlingua', sim: d.llmlingua.output_similarity, compression: d.llmlingua.compression, persona: d.llmlingua.persona_preserved }))}
                fill="#d97706"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DomainTable summary={summary} />
        <TierChart summary={summary} />
      </div>
    </div>
  );
}

function HeadlineCard({
  label,
  ours,
  lingua,
  higherIsBetter,
  numericOurs,
  numericLingua,
  hint,
}: {
  label: string;
  ours: string;
  lingua: string;
  higherIsBetter: boolean;
  numericOurs: number;
  numericLingua: number;
  hint?: string;
}) {
  const oursWins = higherIsBetter ? numericOurs >= numericLingua : numericOurs <= numericLingua;
  return (
    <div className="rounded-lg border border-line bg-white shadow-card p-4">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      {hint && <div className="text-[11px] text-muted/80 -mt-0.5">{hint}</div>}
      <div className="mt-3 grid grid-cols-2 gap-2 items-baseline">
        <div>
          <div className="text-[11px] text-muted">Ours</div>
          <div className={'text-xl font-semibold ' + (oursWins ? 'text-ours' : '')}>{ours}</div>
        </div>
        <div>
          <div className="text-[11px] text-muted">LLMLingua</div>
          <div className={'text-xl font-semibold ' + (!oursWins ? 'text-lingua' : 'text-muted')}>{lingua}</div>
        </div>
      </div>
    </div>
  );
}

interface ScatterPayload {
  payload: {
    label: string;
    system: 'ours' | 'llmlingua';
    sim: number;
    compression: number;
    persona: boolean;
  };
}
function ScatterTooltip({ active, payload }: { active?: boolean; payload?: ScatterPayload[] }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md bg-ink text-canvas text-xs px-3 py-2 shadow-card">
      <div className="font-semibold">{p.label}</div>
      <div className="opacity-70 capitalize">{p.system}</div>
      <div className="mt-1">compression {(p.compression * 100).toFixed(1)}%</div>
      <div>output sim {p.sim.toFixed(3)}</div>
      <div>persona {p.persona ? '✓' : '✗'}</div>
    </div>
  );
}

function DomainTable({ summary }: { summary: Summary }) {
  // Sort by |judge_ours − judge_lingua| descending
  const rows = useMemo(() => {
    return [...summary.domain_breakdown]
      .map(r => ({
        ...r,
        delta: (r.ours.judge_score ?? 0) - (r.llmlingua.judge_score ?? 0),
      }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  }, [summary]);

  return (
    <div className="rounded-lg border border-line bg-white shadow-card p-5">
      <h2 className="text-base font-semibold">By domain (n ≥ 3)</h2>
      <p className="text-xs text-muted mt-0.5">Sorted by judge-score gap (ours − LLMLingua).</p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-muted text-left">
            <tr>
              <th className="py-1.5 pr-3">Domain</th>
              <th className="py-1.5 pr-3 text-right">n</th>
              <th className="py-1.5 pr-3 text-right">Ours comp</th>
              <th className="py-1.5 pr-3 text-right">LLM comp</th>
              <th className="py-1.5 pr-3 text-right">Ours judge</th>
              <th className="py-1.5 pr-3 text-right">LLM judge</th>
              <th className="py-1.5 text-right">Δ judge</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.domain} className="border-t border-line">
                <td className="py-1.5 pr-3 capitalize">{r.domain.replace('_', ' ')}</td>
                <td className="py-1.5 pr-3 text-right">{r.n}</td>
                <td className="py-1.5 pr-3 text-right">{((r.ours.compression ?? 0) * 100).toFixed(1)}%</td>
                <td className="py-1.5 pr-3 text-right">{((r.llmlingua.compression ?? 0) * 100).toFixed(1)}%</td>
                <td className="py-1.5 pr-3 text-right">{(r.ours.judge_score ?? 0).toFixed(1)}</td>
                <td className="py-1.5 pr-3 text-right">{(r.llmlingua.judge_score ?? 0).toFixed(1)}</td>
                <td className={'py-1.5 text-right font-medium ' + (r.delta >= 0 ? 'text-ours' : 'text-lingua')}>
                  {r.delta >= 0 ? '+' : ''}{r.delta.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TierChart({ summary }: { summary: Summary }) {
  const total = Object.values(summary.tier_distribution).reduce((a, b) => a + b, 0);
  const entries = Object.entries(summary.tier_distribution).sort();
  const palette: Record<string, string> = { T1: '#2563eb', T2: '#d97706', T3: '#059669' };
  const labels: Record<string, string> = {
    T1: 'Bayesian BO (low density)',
    T2: 'TextRank (mid)',
    T3: 'Pass-through (dense)',
  };
  return (
    <div className="rounded-lg border border-line bg-white shadow-card p-5">
      <h2 className="text-base font-semibold">Tier distribution</h2>
      <p className="text-xs text-muted mt-0.5">
        How the {total} prompts in the full corpus routed through the pipeline.
      </p>
      <div className="mt-4 space-y-3">
        {entries.map(([tier, count]) => {
          const pct = (count / total) * 100;
          return (
            <div key={tier}>
              <div className="flex items-baseline justify-between text-xs mb-1">
                <span className="font-mono">{tier}</span>
                <span className="text-muted">{labels[tier] ?? ''}</span>
                <span className="font-mono text-muted">{count} · {pct.toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded-full bg-line overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${pct}%`, background: palette[tier] ?? '#64748b' }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
