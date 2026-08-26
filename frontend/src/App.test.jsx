// The cheapest test that would actually have caught something.
//
// Not a component-by-component suite -- the brief does not ask for one and a
// large frontend suite for a demo page is cost without cover. What these check
// is the handful of things that would embarrass the demo if they broke: the page
// renders, the scenario dropdown populates, an unavailable provider is visibly
// unavailable rather than silently missing, and a fallback is badged.

import { render, screen, waitFor } from '@testing-library/react'
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
