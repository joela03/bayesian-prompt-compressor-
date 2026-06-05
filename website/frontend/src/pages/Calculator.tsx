import { useMemo, useState } from 'react';

// Model pricing in USD per 1M input tokens.
const MODELS = [
  { id: 'gpt-4o', label: 'GPT-4o', price_per_million: 2.5 },
  { id: 'gpt-4.1', label: 'GPT-4.1', price_per_million: 2.0 },
  { id: 'gpt-4o-mini', label: 'GPT-4o mini', price_per_million: 0.15 },
  { id: 'claude-sonnet', label: 'Claude Sonnet', price_per_million: 3.0 },
  { id: 'claude-haiku', label: 'Claude Haiku', price_per_million: 0.25 },
];

const RPD_PRESETS = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];

// Carbon coefficients (rough, see calculator explainer panel below)
const KWH_PER_1K_TOKENS = 0.001;
const KG_CO2_PER_KWH = 0.233;
const KG_CO2_PER_TREE_YEAR = 21;

function fmtUSD(n: number) {
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(2)}`;
}

function fmtInt(n: number) {
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

export default function Calculator() {
  const [promptLen, setPromptLen] = useState(300);
  const [compressionPct, setCompressionPct] = useState(24);
  const [rpdIdx, setRpdIdx] = useState(2); // default 10K
  const [modelId, setModelId] = useState('gpt-4o');
  const [showMath, setShowMath] = useState(false);

  const model = MODELS.find(m => m.id === modelId)!;
  const rpd = RPD_PRESETS[rpdIdx];

  const calc = useMemo(() => {
    const compressionRatio = compressionPct / 100;
    const tokensSavedPerCall = promptLen * compressionRatio;
    const callsPerYear = rpd * 365;
    const tokensSavedPerYear = tokensSavedPerCall * callsPerYear;
    const costSavedPerYear = (tokensSavedPerYear / 1_000_000) * model.price_per_million;
    const kwhSaved = (tokensSavedPerYear / 1000) * KWH_PER_1K_TOKENS;
    const kgCO2 = kwhSaved * KG_CO2_PER_KWH;
    const treeYears = kgCO2 / KG_CO2_PER_TREE_YEAR;
    return { tokensSavedPerYear, costSavedPerYear, kgCO2, treeYears };
  }, [promptLen, compressionPct, rpd, model]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Cost & carbon</h1>
        <p className="text-muted text-sm mt-1">
          Estimated annual savings assuming every prompt compresses at the indicated rate and only the
          system prompt counts. Output tokens, retries, and tool calls are not included.
        </p>
      </div>

      <section className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: controls */}
        <div className="lg:col-span-2 space-y-5 rounded-lg border border-line bg-white shadow-card p-5">
          <Slider
            label="Prompt length"
            unit="tokens"
            value={promptLen}
            min={100}
            max={1000}
            step={50}
            display={fmtInt(promptLen)}
            onChange={setPromptLen}
          />
          <Slider
            label="Compression rate"
            unit="%"
            value={compressionPct}
            min={0}
            max={60}
            step={1}
            display={`${compressionPct}%`}
            onChange={setCompressionPct}
          />

          <div>
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-sm font-medium">Requests per day</span>
              <span className="text-sm font-mono text-muted">{fmtInt(rpd)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={RPD_PRESETS.length - 1}
              step={1}
              value={rpdIdx}
              onChange={e => setRpdIdx(parseInt(e.target.value, 10))}
            />
            <div className="flex justify-between text-[10px] text-muted mt-1">
              {RPD_PRESETS.map(v => (
                <span key={v}>{v >= 1_000_000 ? `${v / 1_000_000}M` : v >= 1000 ? `${v / 1000}K` : v}</span>
              ))}
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">Model</label>
            <select
              value={modelId}
              onChange={e => setModelId(e.target.value)}
              className="w-full rounded-md border border-line bg-white px-3 py-2 text-sm"
            >
              {MODELS.map(m => (
                <option key={m.id} value={m.id}>
                  {m.label} — ${m.price_per_million.toFixed(2)}/1M input tokens
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Right: outputs */}
        <div className="lg:col-span-3 grid grid-cols-2 gap-3">
          <Stat label="Tokens saved / year" value={fmtInt(calc.tokensSavedPerYear)} accent="ink" />
          <Stat label="Annual cost saving" value={fmtUSD(calc.costSavedPerYear)} accent="ours" />
          <Stat label="CO₂ avoided" value={`${calc.kgCO2.toFixed(1)} kg`} accent="ink" hint="per year" />
          <Stat label="Tree-years equivalent" value={calc.treeYears.toFixed(1)} accent="ours" hint={`× ${KG_CO2_PER_TREE_YEAR} kg CO₂ per tree per year`} />
        </div>
      </section>

      <section className="rounded-lg border border-line bg-white shadow-card">
        <button
          className="w-full px-5 py-3 text-left text-sm font-medium flex items-center justify-between"
          onClick={() => setShowMath(s => !s)}
        >
          How this is calculated
          <span className="text-muted text-xs">{showMath ? 'hide' : 'show'}</span>
        </button>
        {showMath && (
          <div className="px-5 pb-5 text-sm text-ink/90 space-y-2 font-mono leading-relaxed">
            <div>tokens_saved_per_call  = prompt_length × compression_ratio</div>
            <div>calls_per_year         = requests_per_day × 365</div>
            <div>tokens_saved_per_year  = tokens_saved_per_call × calls_per_year</div>
            <div>cost_saved_per_year    = tokens_saved_per_year / 1,000,000 × model_price</div>
            <div>kWh_saved              = tokens_saved_per_year / 1,000 × {KWH_PER_1K_TOKENS}</div>
            <div>kg_CO₂_avoided         = kWh_saved × {KG_CO2_PER_KWH}</div>
            <div>tree_years             = kg_CO₂_avoided / {KG_CO2_PER_TREE_YEAR}</div>
            <div className="text-muted text-xs mt-3 font-sans">
              Energy and carbon coefficients are rough industry estimates and intended for back-of-envelope use,
              not formal sustainability reporting. Output tokens are not included; if you account for output
              token costs in your own budget, multiply accordingly.
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function Slider({
  label,
  unit,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm font-mono text-muted">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseInt(e.target.value, 10))}
      />
      <div className="flex justify-between text-[10px] text-muted mt-1">
        <span>{min} {unit}</span>
        <span>{max} {unit}</span>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
  hint,
}: {
  label: string;
  value: string;
  accent: 'ink' | 'ours';
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-line bg-white shadow-card p-4">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className={'mt-1 text-2xl font-semibold ' + (accent === 'ours' ? 'text-ours' : 'text-ink')}>
        {value}
      </div>
      {hint && <div className="text-[11px] text-muted mt-1">{hint}</div>}
    </div>
  );
}
