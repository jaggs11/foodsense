// The four-stage pipeline, with each stage's own runtime.
//
// The stepper is not decoration: the proposal's contribution is a *four-stage*
// pipeline, and a viewer should be able to see all four ran, in order, and what
// each cost. Runtimes come from the trace, per stage, because "it was fast" is
// not a claim anyone should have to take on trust.

const STAGES = [
  { key: 'stage1', n: 1, name: 'Prediction', blurb: 'Score the planned meal' },
  { key: 'stage2', n: 2, name: 'Counterfactual', blurb: 'Edit it, minimally and safely' },
  { key: 'stage3', n: 3, name: 'Translation', blurb: 'Say what changed, in words' },
  { key: 'stage4', n: 4, name: 'Verification', blurb: 'Re-check every claim against USDA' },
]

export default function Stepper({ trace, active, onSelect }) {
  return (
    <ol className="grid grid-cols-2 md:grid-cols-4 gap-2">
      {STAGES.map((stage) => {
        const result = trace?.[stage.key]
        const ran = Boolean(result)
        const isActive = active === stage.key
        return (
          <li key={stage.key}>
            <button
              type="button"
              onClick={() => onSelect(stage.key)}
              disabled={!ran}
              aria-current={isActive ? 'step' : undefined}
              className={[
                'w-full text-left rounded-lg border p-3 transition',
                isActive
                  ? 'border-slate-800 bg-slate-800 text-white shadow'
                  : ran
                    ? 'border-slate-300 bg-white hover:border-slate-500'
                    : 'border-dashed border-slate-300 bg-slate-50 text-slate-400 cursor-not-allowed',
              ].join(' ')}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`flex-none w-6 h-6 rounded-full grid place-items-center text-xs font-bold ${
                    isActive
                      ? 'bg-white text-slate-800'
                      : ran
                        ? 'bg-slate-800 text-white'
                        : 'bg-slate-200 text-slate-400'
                  }`}
                >
                  {ran ? stage.n : '·'}
                </span>
                <span className="font-semibold text-sm">{stage.name}</span>
              </div>
              <p
                className={`text-xs mt-1.5 ${isActive ? 'text-slate-200' : 'text-slate-500'}`}
              >
                {stage.blurb}
              </p>
              <p
                className={`text-[11px] mt-1 font-mono ${isActive ? 'text-slate-300' : 'text-slate-400'}`}
              >
                {ran ? `${(result.runtime_s * 1000).toFixed(0)} ms` : 'did not run'}
              </p>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

export { STAGES }
