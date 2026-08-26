import { useEffect, useRef, useState } from 'react'
import { searchFoods } from '../api'

// Build an arbitrary case: who is eating, what they planned, what they have.
//
// This is what `/api/foods?q=` was always for. The endpoint shipped before any
// consumer existed, which meant the one part of the system a curious examiner
// would reach for -- "what does it do with *my* meal?" -- was reachable only by
// hand-writing JSON.
//
// Entirely local: the autocomplete queries the same curated USDA database the
// pipeline scores against, so a custom run needs no network and no key, exactly
// like the preloaded scenarios.

const AGE_GROUPS = [
  { value: 'toddler', label: 'Toddler', months: 18, weight: 11 },
  { value: 'adult', label: 'Adult', months: 360, weight: 70 },
  { value: 'older_adult', label: 'Older adult', months: 900, weight: 68 },
]

const GOALS = [
  { value: 'balanced_nutrition', label: 'Balanced nutrition' },
  { value: 'glycemic_control', label: 'Glycemic control' },
  { value: 'weight_management', label: 'Weight management' },
]

const FLAGS = [
  'hypertension',
  'diabetes',
  'dysphagia',
  'iron_focus',
  'strict_no_added_sugar',
  'warfarin',
  'maoi',
  'ace_inhibitor_or_k_sparing_diuretic',
  'statin',
  'metformin',
]

/** Debounced type-ahead over the curated database. */
function FoodSearch({ onPick, placeholder }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const timer = useRef(null)

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current)
    if (query.trim().length < 2) {
      setResults([])
      return
    }
    // Debounced: a keystroke-per-request would hammer the matcher, which does
    // real token work rather than a substring scan.
    timer.current = setTimeout(() => {
      setBusy(true)
      searchFoods(query, 8)
        .then((r) => {
          setResults(r)
          setOpen(true)
        })
        .catch(() => setResults([]))
        .finally(() => setBusy(false))
    }, 220)
    return () => timer.current && clearTimeout(timer.current)
  }, [query])

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        placeholder={placeholder}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="w-full border border-slate-300 rounded px-2 py-1.5 text-sm"
      />
      {busy && (
        <span className="absolute right-2 top-2 text-xs text-slate-400">searching…</span>
      )}
      {open && results.length > 0 && (
        <ul className="absolute z-20 mt-1 w-full bg-white border border-slate-300 rounded shadow-lg max-h-64 overflow-y-auto">
          {results.map((food) => (
            <li key={food.food_id}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onPick(food)
                  setQuery('')
                  setResults([])
                  setOpen(false)
                }}
                className="w-full text-left px-2 py-1.5 text-sm hover:bg-slate-100"
              >
                <div className="text-slate-800">{food.name}</div>
                <div className="text-xs text-slate-500">
                  {food.category} &middot; {food.energy_kcal_per_100g} kcal/100 g
                  {food.hazard_class && (
                    <span className="text-rose-600"> &middot; {food.hazard_class} hazard</span>
                  )}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** One editable list of items — used for both the planned meal and the pantry. */
function ItemList({ title, hint, items, setItems, withQuantity }) {
  const add = (food) =>
    setItems([
      ...items,
      {
        food_id: food.food_id,
        name: food.name,
        // Pantry items are possibilities rather than portions, so they start at
        // zero: the optimiser decides whether one is worth its distance cost.
        quantity_g: withQuantity ? 100 : 0,
        form: food.allowed_forms[0],
        allowed_forms: food.allowed_forms,
      },
    ])

  const update = (i, patch) =>
    setItems(items.map((item, index) => (index === i ? { ...item, ...patch } : item)))

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <h4 className="text-sm font-semibold text-slate-800">{title}</h4>
        <span className="text-xs text-slate-500">{items.length} items</span>
      </div>
      <p className="text-xs text-slate-500 mb-2">{hint}</p>
      <FoodSearch onPick={add} placeholder="Type at least 2 letters…" />
      <ul className="mt-2 space-y-1">
        {items.map((item, i) => (
          <li
            key={`${item.food_id}-${i}`}
            className="flex items-center gap-2 border border-slate-200 rounded px-2 py-1.5 bg-white"
          >
            <span className="flex-1 text-sm text-slate-800 truncate" title={item.name}>
              {item.name}
            </span>
            {withQuantity && (
              <label className="flex items-center gap-1">
                <input
                  type="number"
                  min="0"
                  max="2000"
                  value={item.quantity_g}
                  onChange={(e) => update(i, { quantity_g: Number(e.target.value) })}
                  className="w-20 border border-slate-300 rounded px-1 py-0.5 text-sm text-right"
                />
                <span className="text-xs text-slate-500">g</span>
              </label>
            )}
            <select
              value={item.form}
              onChange={(e) => update(i, { form: e.target.value })}
              className="border border-slate-300 rounded px-1 py-0.5 text-sm"
            >
              {item.allowed_forms.map((f) => (
                <option key={f} value={f}>
                  {f.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setItems(items.filter((_, index) => index !== i))}
              className="text-slate-400 hover:text-rose-600 px-1"
              aria-label={`Remove ${item.name}`}
            >
              &times;
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function CustomBuilder({ value, onChange }) {
  const { profile, planned, pantry } = value
  const set = (patch) => onChange({ ...value, ...patch })
  const setProfile = (patch) => set({ profile: { ...profile, ...patch } })

  const toggleFlag = (flag) =>
    setProfile({
      health_flags: profile.health_flags.includes(flag)
        ? profile.health_flags.filter((f) => f !== flag)
        : [...profile.health_flags, flag],
    })

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold text-slate-800 mb-2">Who is eating</h4>
        <div className="grid sm:grid-cols-4 gap-3">
          <label className="block">
            <span className="text-xs uppercase tracking-wide text-slate-500">Age group</span>
            <select
              className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5 text-sm"
              value={profile.age_group}
              onChange={(e) => {
                // Age in months drives the choking rules, so it moves with the
                // group rather than being left at a value from another life stage.
                const g = AGE_GROUPS.find((a) => a.value === e.target.value)
                setProfile({
                  age_group: g.value,
                  age_months: g.months,
                  weight_kg: g.weight,
                })
              }}
            >
              {AGE_GROUPS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-xs uppercase tracking-wide text-slate-500">Age (months)</span>
            <input
              type="number"
              min="1"
              max="1400"
              value={profile.age_months}
              onChange={(e) => setProfile({ age_months: Number(e.target.value) })}
              className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5 text-sm"
            />
          </label>

          <label className="block">
            <span className="text-xs uppercase tracking-wide text-slate-500">Weight (kg)</span>
            <input
              type="number"
              min="1"
              max="300"
              value={profile.weight_kg}
              onChange={(e) => setProfile({ weight_kg: Number(e.target.value) })}
              className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5 text-sm"
            />
          </label>

          <label className="block">
            <span className="text-xs uppercase tracking-wide text-slate-500">Goal</span>
            <select
              className="mt-1 w-full border border-slate-300 rounded px-2 py-1.5 text-sm"
              value={profile.goal}
              onChange={(e) => setProfile({ goal: e.target.value })}
            >
              {GOALS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mt-3">
          <span className="text-xs uppercase tracking-wide text-slate-500">Health flags</span>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {FLAGS.map((flag) => {
              const on = profile.health_flags.includes(flag)
              return (
                <button
                  key={flag}
                  type="button"
                  onClick={() => toggleFlag(flag)}
                  aria-pressed={on}
                  className={`text-xs rounded-full border px-2.5 py-1 ${
                    on
                      ? 'bg-slate-800 text-white border-slate-800'
                      : 'bg-white text-slate-600 border-slate-300 hover:border-slate-500'
                  }`}
                >
                  {flag.replace(/_/g, ' ')}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <ItemList
          title="Planned meal"
          hint="What they were going to eat. The optimiser edits this."
          items={planned}
          setItems={(items) => set({ planned: items })}
          withQuantity
        />
        <ItemList
          title="Pantry"
          hint="What else they have. Nothing outside these two lists can be recommended."
          items={pantry}
          setItems={(items) => set({ pantry: items })}
          withQuantity={false}
        />
      </div>
    </div>
  )
}

export { AGE_GROUPS, GOALS, FLAGS }
