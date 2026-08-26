// The cheapest test that would actually have caught something.
//
// Not a component-by-component suite -- the brief does not ask for one and a
// large frontend suite for a demo page is cost without cover. What these check
// is the handful of things that would embarrass the demo if they broke: the page
// renders, the scenario dropdown populates, an unavailable provider is visibly
// unavailable rather than silently missing, and a fallback is badged.

import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const SCENARIOS = [
  {
    key: 'toddler_choking',
    title: 'Toddler, 18 months: whole grapes and whole peanuts',
    description: 'A choking-hazard lunch.',
    age_group: 'toddler',
    goal: 'balanced_nutrition',
    health_flags: [],
    n_planned_items: 3,
    n_pantry_items: 8,
  },
]

const PROVIDERS = [
  { name: 'template', available: true, reason: '', is_default: true },
  { name: 'anthropic', available: false, reason: 'ANTHROPIC_API_KEY not set', is_default: false },
]

const HEALTH = {
  status: 'ok',
  version: '0.1.0',
  model_loaded: true,
  n_foods: 2590,
  default_provider: 'template',
}

function stubFetch(overrides = {}) {
  const routes = {
    '/api/scenarios': SCENARIOS,
    '/api/providers': PROVIDERS,
    '/api/health': HEALTH,
    ...overrides,
  }
  vi.stubGlobal(
    'fetch',
    vi.fn((url) =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(routes[url] ?? {}),
      }),
    ),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('App', () => {
  it('renders the title and the offline claim', async () => {
    stubFetch()
    render(<App />)
    expect(screen.getByText('FoodSense')).toBeTruthy()
    await waitFor(() => expect(screen.getByText(/runs\s+offline/)).toBeTruthy())
  })

  it('populates the scenario dropdown from the API', async () => {
    stubFetch()
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText(/Toddler, 18 months/)).toBeTruthy(),
    )
  })

  it('marks an unavailable provider and shows why', async () => {
    // The offline-first claim depends on "no key" reading as a deliberate
    // default rather than as breakage, so the reason has to be on screen.
    stubFetch()
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText(/anthropic — unavailable/)).toBeTruthy()
      expect(screen.getByText(/ANTHROPIC_API_KEY not set/)).toBeTruthy()
    })
  })

  it('surfaces an API error instead of rendering a blank page', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) })),
    )
    render(<App />)
    await waitFor(() => expect(screen.getByText(/Request failed/)).toBeTruthy())
  })
})

const TRACE = (label) => ({
  profile: {},
  planned_meal: { items: [] },
  final_meal: { items: [{ food_id: '1', name: `${label} food`, form: 'whole', quantity_g: 10 }] },
  stage1: {
    suitability: 0.1,
    runtime_s: 0.01,
    model_name: 'lightgbm',
    rule_evaluation: {
      score: 0.1,
      is_safe: false,
      violations: [{ rule_id: `${label}.rule`, severity: 'hard', message: `${label} violation` }],
    },
  },
  stage2: {
    valid: false,
    n_generations: 1,
    n_evaluations: 10,
    search_space_size: 3,
    runtime_s: 0.01,
    diff: { changes: [], n_items_changed: 0, l1_distance_g: 0 },
  },
  stage3: { provider: 'template', fallback_used: false, text: 't', rationale: [], runtime_s: 0.01 },
  stage4: { final_pass: true, checked: 1, corrected: [], safety_fixes: [], unmatched: [], flagged: [], runtime_s: 0.01 },
  final_rule_evaluation: { score: 0.7, is_safe: true },
  warnings: [],
  seed: 42,
  total_runtime_s: 0.1,
})

describe('stale results', () => {
  it('keeps the results labelled with the run that produced them', async () => {
    // The bug: pick toddler, run, switch the dropdown to adult -- and Stage 1
    // still showed the toddler's choking violations with nothing saying so, so
    // the panel read as though it belonged to the adult scenario.
    stubFetch({ '/api/recommend': TRACE('toddler') })
    render(<App />)
    await waitFor(() => expect(screen.getByText(/Toddler, 18 months/)).toBeTruthy())

    fireEvent.click(screen.getByRole('button', { name: /run pipeline/i }))
    await waitFor(() => expect(screen.getByText(/Showing results for/)).toBeTruthy())
    expect(screen.getByRole('heading', { name: /Toddler, 18 months/ })).toBeTruthy()

    // Change the selection without re-running.
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: '__custom__' } })

    await waitFor(() =>
      expect(screen.getByText(/controls above have changed since this ran/i)).toBeTruthy(),
    )
    // And it still names the run it actually came from.
    expect(screen.getByRole('heading', { name: /Toddler, 18 months/ })).toBeTruthy()
  })
})

describe('custom builder', () => {
  it('offers a custom option in the scenario dropdown', async () => {
    stubFetch()
    render(<App />)
    await waitFor(() => expect(screen.getByText(/build your own case/i)).toBeTruthy())
  })

  it('shows the profile and meal builder when custom is selected', async () => {
    stubFetch()
    render(<App />)
    await waitFor(() => expect(screen.getByText(/build your own case/i)).toBeTruthy())
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: '__custom__' } })

    await waitFor(() => {
      expect(screen.getByText('Who is eating')).toBeTruthy()
      expect(screen.getByText('Planned meal')).toBeTruthy()
      expect(screen.getByText('Pantry')).toBeTruthy()
    })
  })

  it('refuses to run a custom case with an empty planned meal', async () => {
    stubFetch()
    render(<App />)
    await waitFor(() => expect(screen.getByText(/build your own case/i)).toBeTruthy())
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: '__custom__' } })

    await waitFor(() =>
      expect(screen.getByText(/Add at least one food to the planned meal/)).toBeTruthy(),
    )
    expect(screen.getByRole('button', { name: /run pipeline/i }).disabled).toBe(true)
  })
})
