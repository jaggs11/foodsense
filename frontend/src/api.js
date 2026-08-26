// Thin fetch layer. Same origin in the shipped shape -- FastAPI serves this
// bundle -- so these are relative paths with no base URL and no CORS.

/** Read the one error shape the API emits, or invent a readable one. */
async function unwrap(response) {
  if (response.ok) return response.json()
  let detail = `HTTP ${response.status}`
  try {
    const body = await response.json()
    detail = body.detail || body.error || detail
  } catch {
    // A non-JSON error body is itself the useful signal; keep the status.
  }
  const error = new Error(detail)
  error.status = response.status
  throw error
}

export const getHealth = () => fetch('/api/health').then(unwrap)
export const getScenarios = () => fetch('/api/scenarios').then(unwrap)
export const getProviders = () => fetch('/api/providers').then(unwrap)

/**
 * Run the pipeline. `body` is either {scenario} or {profile, planned_meal, pantry}.
 * A provider with no key is NOT an error here -- it comes back 200 with
 * stage3.fallback_used set, and the UI badges it.
 */
export const recommend = (body) =>
  fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(unwrap)
