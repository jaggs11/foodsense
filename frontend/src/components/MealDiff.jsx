// The Stage-2 before/after, which is the thing a viewer actually looks at.
//
// Colour carries meaning and is never the only carrier: every row also has a
// word ("removed", "added") and a symbol, because a projector in a lecture
// theatre washes out exactly the distinction the colours are making.

const STYLES = {
  removed: {
    row: 'bg-rose-50 border-rose-200',
    label: 'text-rose-700',
    text: 'line-through text-rose-900/70',
    tag: 'removed',
    mark: '−',
  },
  added: {
    row: 'bg-emerald-50 border-emerald-200',
    label: 'text-emerald-700',
    text: 'text-emerald-900',
    tag: 'added',
    mark: '+',
  },
  modified: {
    row: 'bg-amber-50 border-amber-200',
    label: 'text-amber-700',
    text: 'text-amber-900',
    tag: 'changed',
    mark: '→',
  },
  unchanged: {
    row: 'bg-white border-slate-200',
    label: 'text-slate-400',
    text: 'text-slate-600',
    tag: 'kept',
    mark: '·',
  },
}

function Quantity({ change }) {
  const { change_type: kind, old_quantity_g: before, new_quantity_g: after } = change
  if (kind === 'removed') return <span>{before?.toFixed(0)} g &rarr; none</span>
  if (kind === 'added') return <span>{after?.toFixed(0)} g</span>
  if (before != null && after != null && Math.abs(after - before) > 0.5) {
    return (
      <span>
        {before.toFixed(0)} g &rarr; <strong>{after.toFixed(0)} g</strong>
      </span>
    )
  }
  return <span>{(after ?? before ?? 0).toFixed(0)} g</span>
}

function FormChange({ change }) {
  const { old_form: before, new_form: after } = change
  if (!before || !after || before === after) {
    return after ? <span className="text-slate-500">{after.replace(/_/g, ' ')}</span> : null
  }
  // The headline repair of the whole project: a hazard fixed by re-forming a
  // food rather than deleting it. Worth making impossible to miss.
  return (
    <span className="font-medium">
      <span className="line-through text-slate-400">{before.replace(/_/g, ' ')}</span>{' '}
      &rarr; <span className="text-amber-800">{after.replace(/_/g, ' ')}</span>
    </span>
  )
}

export default function MealDiff({ diff }) {
  if (!diff) return null
  const changes = diff.changes ?? []
  const edits = changes.filter((c) => c.change_type !== 'unchanged')
  const kept = changes.filter((c) => c.change_type === 'unchanged')

  return (
    <div>
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="font-semibold text-slate-800">What changed</h3>
        <p className="text-sm text-slate-500">
          {diff.n_items_changed} edit{diff.n_items_changed === 1 ? '' : 's'} &middot;{' '}
          {Math.round(diff.l1_distance_g)} g moved
        </p>
      </div>

      {edits.length === 0 && (
        <p className="text-sm text-slate-500 italic mb-3">
          Nothing was changed &mdash; the planned meal already met the guidelines.
        </p>
      )}

      <ul className="space-y-2">
        {[...edits, ...kept].map((change, i) => {
          const style = STYLES[change.change_type] ?? STYLES.unchanged
          return (
            <li key={`${change.food_id}-${i}`} className={`border rounded-lg p-3 ${style.row}`}>
              <div className="flex items-start gap-3">
                <span className={`font-mono text-lg leading-none pt-0.5 ${style.label}`}>
                  {style.mark}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <span className={`font-medium ${style.text}`}>{change.name}</span>
                    <span
                      className={`text-[11px] uppercase tracking-wide font-semibold ${style.label}`}
                    >
                      {style.tag}
                    </span>
                  </div>
                  <div className="text-sm text-slate-600 mt-0.5 flex flex-wrap gap-x-3">
                    <Quantity change={change} />
                    <FormChange change={change} />
                  </div>
                  {change.reason && (
                    // The rule that caused the edit, carried from the engine. This
                    // is what makes the recommendation an explanation.
                    <p className="text-xs text-slate-600 mt-1.5 border-l-2 border-slate-300 pl-2">
                      {change.reason}
                    </p>
                  )}
                </div>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
