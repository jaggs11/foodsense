// The numbers that decide whether the recommendation was any good.
//
// Suitability before -> after is the headline, but it is shown next to the edit
// count and the distance on purpose: a large improvement bought by rebuilding
// the plate is not the same achievement as a small one bought with two edits,
// and showing only the first would flatter the system.

function Metric({ label, value, sub, tone = 'plain' }) {
  const tones = {
    plain: 'text-slate-900',
    good: 'text-emerald-700',
    bad: 'text-rose-700',
  }
  return (
    <div className="px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`text-lg font-semibold ${tones[tone]}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export default function Metrics({ trace }) {
  const before = trace?.stage1?.rule_evaluation?.score
  const after = trace?.final_rule_evaluation?.score
  const diff = trace?.stage2?.diff
  const safe = trace?.final_rule_evaluation?.is_safe
  const improved = before != null && after != null && after > before

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 divide-x divide-slate-200 border border-slate-200 rounded-lg bg-white">
      <Metric
        label="Suitability"
        value={
          before != null && after != null
            ? `${before.toFixed(2)} → ${after.toFixed(2)}`
            : '—'
        }
        sub={improved ? 'improved' : 'unchanged'}
        tone={improved ? 'good' : 'plain'}
      />
      <Metric label="Edits" value={diff?.n_items_changed ?? '—'} sub="items changed" />
      <Metric
        label="Distance"
        value={diff ? `${Math.round(diff.l1_distance_g)} g` : '—'}
        sub="L1 from the plan"
      />
      <Metric
        label="Safety"
        value={safe ? 'safe' : 'unsafe'}
        sub={safe ? 'no hard rule broken' : 'see warnings'}
        tone={safe ? 'good' : 'bad'}
      />
      <Metric
        label="Total time"
        value={trace ? `${(trace.total_runtime_s * 1000).toFixed(0)} ms` : '—'}
        sub="all four stages"
      />
    </div>
  )
}
