import { useEffect, useState } from 'react'
import { getHealth, getProviders, getScenarios, recommend } from './api'
import MealDiff from './components/MealDiff'
import Metrics from './components/Metrics'
import Stepper from './components/Stepper'
import Verification, { Badge } from './components/Verification'

// One page, four stages, no routing. The demo is a single linear story -- pick a
// scenario, watch it run, inspect what happened -- and a router would add a
// second thing that can be broken in front of an audience.

function StagePanel({ stage, trace }) {
  if (!trace) return null

  if (stage === 'stage1') {
    const s = trace.stage1
    const ev = s.rule_evaluation
    return (
      <div className="space-y-3">
        <h3 className="font-semibold text-slate-800">Stage 1 &mdash; the planned meal, scored</h3>
        <p className="text-sm text-slate-600">
          Suitability <strong>{s.suitability.toFixed(3)}</strong> from{' '}
          <span className="font-mono text-xs">{s.model_name}</span>; the rule engine scores it{' '}
          <strong>{ev.score.toFixed(3)}</strong>.
        </p>
        {ev.violations?.length > 0 ? (
          <ul className="space-y-1">
            {ev.violations.map((v, i) => (
              <li
                key={i}
                className={`text-sm border rounded p-2 ${
                  v.severity === 'hard'
                    ? 'bg-rose-50 border-rose-200'
                    : 'bg-amber-50 border-amber-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Badge tone={v.severity === 'hard' ? 'fail' : 'warn'}>{v.severity}</Badge>
                  <span className="font-mono text-xs text-slate-600">{v.rule_id}</span>
                </div>
                <div className="text-slate-800 mt-1">{v.message}</div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500 italic">This meal breaks no rule.</p>
        )}
      </div>
    )
  }

  if (stage === 'stage2') {
    const s = trace.stage2
    if (!s) return <p className="text-sm text-slate-500">Stage 2 did not run.</p>
    return (
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-semibold text-slate-800">Stage 2 &mdash; the counterfactual</h3>
          {s.valid ? (
            <Badge tone="pass">reached the target</Badge>
          ) : (
            <Badge tone="warn">best safe edit found</Badge>
          )}
        </div>
        <p className="text-sm text-slate-600">
          {s.n_generations} generations, {s.n_evaluations.toLocaleString()} evaluations over a
          search space of {s.search_space_size} foods &mdash; every one of them from the planned
          meal or the pantry.
        </p>
        <MealDiff diff={s.diff} />
      </div>
    )
  }

  if (stage === 'stage3') {
    const s = trace.stage3
    return (
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-semibold text-slate-800">Stage 3 &mdash; the explanation</h3>
          <Badge tone="quiet">{s.provider}</Badge>
          {s.fallback_used && <Badge tone="warn">fell back to template</Badge>}
        </div>
        {s.fallback_used && s.fallback_reason && (
          // Degraded, never silently. The reason is on screen so "the LLM was
          // skipped" reads as a missing key rather than as a broken demo.
          <p className="text-sm bg-amber-50 border border-amber-200 rounded p-2 text-amber-900">
            {s.fallback_reason}
          </p>
        )}
        <p className="text-slate-800 whitespace-pre-wrap">{s.text}</p>
        {s.rationale?.length > 0 && (
          <ul className="text-sm text-slate-700 list-disc list-inside space-y-0.5">
            {s.rationale.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return <Verification report={trace.stage4} />
}

export default function App() {
  const [scenarios, setScenarios] = useState([])
  const [providers, setProviders] = useState([])
  const [health, setHealth] = useState(null)
  const [scenario, setScenario] = useState('toddler_choking')
  const [provider, setProvider] = useState('template')
  const [seed, setSeed] = useState(42)
  const [trace, setTrace] = useState(null)
  const [stage, setStage] = useState('stage2')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [showJson, setShowJson] = useState(false)

  useEffect(() => {
    Promise.all([getScenarios(), getProviders(), getHealth()])
      .then(([s, p, h]) => {
        setScenarios(s)
        setProviders(p)
        setHealth(h)
      })
      .catch((e) => setError(e.message))
  }, [])

  async function run() {
    setBusy(true)
    setError(null)
    try {
      const result = await recommend({ scenario, provider, seed: Number(seed) })
      setTrace(result)
      setStage('stage2')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const selected = scenarios.find((s) => s.key === scenario)

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="bg-slate-900 text-white">
        <div className="max-w-5xl mx-auto px-4 py-5">
          <h1 className="text-xl font-bold">FoodSense</h1>
          <p className="text-slate-300 text-sm">
            Availability-aware, verification-guided counterfactual food recommendation
          </p>
          {health && (
            <p className="text-slate-400 text-xs mt-2 font-mono">
              v{health.version} &middot; {health.n_foods.toLocaleString()} USDA foods &middot;{' '}
              {health.model_loaded ? 'model loaded' : 'NO MODEL — run make train'} &middot; runs
              offline
            </p>
          )}
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6 space-y-5">
        <section className="bg-white border border-slate-200 rounded-lg p-4">
          <div className="grid md:grid-cols-[2fr_1fr_auto_auto] gap-3 items-end">
            <label className="block">
              <span className="text-xs uppercase tracking-wide text-slate-500">Scenario</span>
              <select
                className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5"
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
              >
                {scenarios.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.title}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="text-xs uppercase tracking-wide text-slate-500">
                Stage-3 provider
              </span>
              <select
                className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                {providers.map((p) => (
                  // Unavailable providers stay selectable but are marked, because
                  // choosing one and watching it degrade gracefully is itself
                  // worth demonstrating.
                  <option key={p.name} value={p.name} disabled={!p.available}>
                    {p.name}
                    {p.available ? (p.is_default ? ' (offline default)' : '') : ' — unavailable'}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="text-xs uppercase tracking-wide text-slate-500">Seed</span>
              <input
                type="number"
                min="0"
                className="mt-1 w-24 border border-slate-300 rounded px-2 py-1.5 font-mono"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </label>

            <button
              type="button"
              onClick={run}
              disabled={busy}
              className="bg-slate-900 text-white rounded px-5 py-2 font-semibold disabled:opacity-50"
            >
              {busy ? 'Running…' : 'Run pipeline'}
            </button>
          </div>

          {selected && <p className="text-sm text-slate-600 mt-3">{selected.description}</p>}

          {providers.some((p) => !p.available) && (
            <p className="text-xs text-slate-500 mt-2">
              {providers
                .filter((p) => !p.available)
                .map((p) => `${p.name}: ${p.reason}`)
                .join(' · ')}
            </p>
          )}
        </section>

        {error && (
          <div className="bg-rose-50 border border-rose-300 rounded-lg p-3 text-rose-900">
            <strong>Request failed.</strong> {error}
          </div>
        )}

        {trace && (
          <>
            <Metrics trace={trace} />

            {trace.warnings?.length > 0 && (
              <div className="bg-amber-50 border border-amber-300 rounded-lg p-3">
                <h3 className="text-sm font-semibold text-amber-900 mb-1">Warnings</h3>
                <ul className="text-sm text-amber-900 list-disc list-inside space-y-0.5">
                  {trace.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            <Stepper trace={trace} active={stage} onSelect={setStage} />

            <section className="bg-white border border-slate-200 rounded-lg p-4">
              <StagePanel stage={stage} trace={trace} />
            </section>

            <section className="bg-white border border-slate-200 rounded-lg">
              <button
                type="button"
                onClick={() => setShowJson((v) => !v)}
                className="w-full text-left px-4 py-3 font-semibold text-slate-800 flex justify-between"
              >
                <span>Raw PipelineTrace</span>
                <span className="text-slate-400 font-mono text-sm">
                  seed {trace.seed} · {showJson ? '−' : '+'}
                </span>
              </button>
              {showJson && (
                <pre className="px-4 pb-4 text-xs overflow-x-auto text-slate-700 max-h-96 overflow-y-auto">
                  {JSON.stringify(trace, null, 2)}
                </pre>
              )}
            </section>
          </>
        )}

        {!trace && !error && (
          <p className="text-slate-500 text-sm">
            Pick a scenario and run it. Everything happens locally &mdash; no network, no API key.
          </p>
        )}
      </main>
    </div>
  )
}
