"""Stage-3 provider abstraction.

``TemplateProvider`` is the default and is fully deterministic and offline -- the
faculty demo never depends on a network call. ``AnthropicProvider``
(claude-sonnet-4-6), ``OpenAIProvider`` and ``OllamaProvider`` honour the same
strict-JSON contract ``{items: [{name, food_id, quantity_g, form}], text, rationale}``
at temperature 0.2, retry once on invalid JSON, then fall back to the template.
A provider failure must never break the pipeline.

TODO(Phase 4): implement.
"""
