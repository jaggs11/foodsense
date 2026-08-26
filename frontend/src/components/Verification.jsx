// Stage 4, rendered as the headline it is.
//
// This layer is the project's third extension and the reason the system can
// claim anything a generator says is checked. Tucking it behind a toggle would
// make the demo assert the claim instead of showing it, so everything the
// verifier did is on screen: what it corrected, what it re-formed or removed,
// what it could not match, and whether the final meal passed.

function Badge({ tone, children }) {
  const tones = {
    pass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    fail: 'bg-rose-100 text-rose-800 border-rose-300',
    warn: 'bg-amber-100 text-amber-800 border-amber-300',
    quiet: 'bg-slate-100 text-slate-700 border-slate-300',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 border rounded-full px-2.5 py-0.5 text-xs font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  )
}

function Count({ n, label, tone }) {
  return (
    <div className="text-center">
      <div
        className={`text-2xl font-semibold ${n > 0 ? (tone === 'bad' ? 'text-rose-700' : 'text-amber-700') : 'text-slate-300'}`}
      >
        {n}
      </div>
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
    </div>
  )
}

export default function Verification({ report }) {
  if (!report) return null
  const corrections = report.corrected ?? []
  const fixes = report.safety_fixes ?? []
  const unmatched = report.unmatched ?? []
  const flagged = report.flagged ?? []
  const clean = corrections.length === 0 && fixes.length === 0 && unmatched.length === 0

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h3 className="font-semibold text-slate-800">Stage 4 &mdash; USDA verification</h3>
        {report.final_pass ? (
          <Badge tone="pass">&#10003; verified safe</Badge>
        ) : (
          <Badge tone="fail">&#10007; could not be made safe</Badge>
        )}
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4 bg-slate-50 border border-slate-200 rounded-lg py-3">
        <Count n={report.checked ?? 0} label="checked" />
        <Count n={corrections.length} label="corrected" />
        <Count n={fixes.length} label="safety fixes" tone="bad" />
        <Count n={unmatched.length} label="unmatched" tone="bad" />
      </div>

      {clean && (
        <p className="text-sm text-slate-600 mb-3">
          Nothing needed correcting. On the offline path Stage 3 emits the optimiser&rsquo;s own
          items, so the figures above being zero is the expected result &mdash; it means the
          reported meal already <em>is</em> the database recomputation, not that the check was
          skipped.
        </p>
      )}

      {flagged.length > 0 && (
        <section className="mb-4">
          <h4 className="text-sm font-semibold text-rose-800 mb-1.5">
            Hazards found in what Stage 3 produced
          </h4>
          <ul className="space-y-1">
            {flagged.map((v, i) => (
              <li key={i} className="text-sm bg-rose-50 border border-rose-200 rounded p-2">
                <span className="font-mono text-xs text-rose-700">{v.rule_id}</span>
                <div className="text-rose-900">{v.message}</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {fixes.length > 0 && (
        <section className="mb-4">
          <h4 className="text-sm font-semibold text-slate-800 mb-1.5">Repairs applied</h4>
          <ul className="space-y-1">
            {fixes.map((fix, i) => (
              <li key={i} className="text-sm bg-amber-50 border border-amber-200 rounded p-2">
                <div className="flex items-center gap-2">
                  <Badge tone="warn">{fix.action}</Badge>
                  <span className="font-medium text-slate-800">{fix.name}</span>
                </div>
                <div className="text-slate-700 mt-1">{fix.message}</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {corrections.length > 0 && (
        <section className="mb-4">
          <h4 className="text-sm font-semibold text-slate-800 mb-1.5">
            Claims corrected to the USDA value
          </h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="pb-1">Field</th>
                <th className="pb-1">Claimed</th>
                <th className="pb-1">Actual</th>
                <th className="pb-1">Error</th>
              </tr>
            </thead>
            <tbody>
              {corrections.map((c, i) => (
                <tr key={i} className="border-t border-slate-200">
                  <td className="py-1 pr-2">
                    <span className="text-slate-800">{c.name}</span>{' '}
                    <span className="text-slate-500">{c.field}</span>
                  </td>
                  <td className="py-1 pr-2 text-rose-700 line-through">{c.claimed}</td>
                  <td className="py-1 pr-2 text-emerald-800 font-medium">{c.corrected}</td>
                  <td className="py-1 text-slate-600">
                    {(c.relative_error * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {unmatched.length > 0 && (
        <section>
          <h4 className="text-sm font-semibold text-slate-800 mb-1.5">
            Named foods that do not exist in the database
          </h4>
          <ul className="text-sm text-rose-800 list-disc list-inside">
            {unmatched.map((name, i) => (
              <li key={i}>{name}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

export { Badge }
