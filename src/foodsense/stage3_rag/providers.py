"""Stage-3 providers: turn an optimised meal into language, under one JSON contract.

Every provider answers the same request with the same shape::

    {"items": [{"name", "food_id", "quantity_g", "form"}], "text", "rationale"}

``TemplateProvider`` is the default and is deterministic, offline and dependency-free.
That ordering is deliberate and is the project's central robustness claim: the LLM is
an *enhancement layer*, never a dependency. A faculty demo must not fail because of
Wi-Fi, an expired key or a rate limit, so the path that always works is the default
path, and the interesting path is opt-in.

The LLM providers all: send the retrieved USDA candidates so the model picks real
foods rather than inventing them, demand strict JSON, validate what comes back,
retry once on invalid JSON, and fall back to the template if that fails. A provider
failure degrades the output; it never breaks the pipeline.

Nothing a provider returns is trusted. Stage 4 re-matches every name, recomputes
every nutrient from the database and re-runs the safety scan -- which is the whole
reason a generative step is safe to have here at all.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from foodsense.constraints.age_rules import load_age_config
from foodsense.schemas import Form, Meal, MealItem, UserProfile

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderResponse",
    "TemplateProvider",
    "TranslationRequest",
    "get_provider",
]

#: The JSON contract every provider honours. Stated once, used in every prompt.
JSON_CONTRACT = (
    '{"items": [{"name": str, "food_id": str, "quantity_g": number, "form": str}], '
    '"text": str, "rationale": [str]}'
)


@dataclass(slots=True)
class TranslationRequest:
    """Everything a provider needs to describe one recommendation."""

    profile: UserProfile
    planned_meal: Meal
    optimized_meal: Meal
    changes: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    texture_notes: dict[str, str] = field(default_factory=dict)
    violations_fixed: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProviderResponse:
    """A provider's answer, before Stage 4 has checked any of it."""

    items: list[MealItem]
    text: str
    rationale: list[str]
    provider: str
    fallback_used: bool = False
    raw: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Phrasing helpers, shared by every provider
# ---------------------------------------------------------------------------

#: Plain-English phrasing per form, used when the age config has nothing to say.
DEFAULT_TEXTURE_NOTES: dict[Form, str] = {
    Form.WHOLE: "as is",
    Form.QUARTERED: "cut into quarters, lengthwise",
    Form.CHOPPED: "chopped small",
    Form.SLICED: "thinly sliced",
    Form.SLICED_ROUNDS: "cut into rounds",
    Form.MASHED: "mashed",
    Form.PUREED: "pureed smooth",
    Form.MINCED: "finely minced",
    Form.GROUND: "ground",
    Form.SOFT_COOKED: "cooked until soft",
    Form.THIN_SPREAD: "spread thinly",
    Form.SPOONFUL: "by the spoonful",
}


def phrase_for(form: Form, texture_notes: dict[str, str]) -> str:
    """How to say a preparation form to the person doing the preparing."""
    return texture_notes.get(form.value) or DEFAULT_TEXTURE_NOTES.get(form, form.value)


def _short(name: str) -> str:
    """The head of a USDA description -- what a person would actually call the food."""
    head = name.split(",")[0].strip()
    return head or name


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """One way of turning a meal edit into language."""

    name: str = "abstract"

    @property
    def available(self) -> bool:
        """Whether this provider can run right now (key present, package installed)."""
        return True

    @abstractmethod
    def generate(self, request: TranslationRequest) -> ProviderResponse: ...

    def unavailable_reason(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# The default: deterministic, offline
# ---------------------------------------------------------------------------


class TemplateProvider(LLMProvider):
    """Deterministic rendering. No network, no key, no variance.

    Produces the same text for the same input every time, which is what makes it
    safe to demo and possible to unit-test. It is also the fallback for every
    other provider, so its output quality sets the floor for the whole system.
    """

    name = "template"

    def generate(self, request: TranslationRequest) -> ProviderResponse:
        notes = request.texture_notes
        lines: list[str] = []
        rationale: list[str] = []

        for change in request.changes:
            kind = change.get("change_type")
            food = _short(str(change.get("name", "")))
            reason = change.get("reason") or ""

            if kind == "removed":
                lines.append(f"Leave out the {food.lower()}.")
            elif kind == "added":
                form = change.get("new_form")
                phrase = phrase_for(Form(form), notes) if form else ""
                grams = change.get("new_quantity_g") or 0
                suffix = f", {phrase}" if phrase and phrase != "as is" else ""
                lines.append(f"Add {grams:.0f} g of {food.lower()}{suffix}.")
            elif kind == "modified":
                old_form = change.get("old_form")
                new_form = change.get("new_form")
                old_q = change.get("old_quantity_g") or 0
                new_q = change.get("new_quantity_g") or 0
                if new_form and old_form and new_form != old_form:
                    phrase = phrase_for(Form(new_form), notes)
                    lines.append(f"Serve the {food.lower()} {phrase}.")
                if abs(new_q - old_q) > 1:
                    verb = "Increase" if new_q > old_q else "Reduce"
                    lines.append(f"{verb} the {food.lower()} from {old_q:.0f} g to {new_q:.0f} g.")
            if reason:
                rationale.append(f"{food}: {reason}")

        if not lines:
            lines.append("This meal already meets the targets; no changes are needed.")

        items = list(request.optimized_meal.items)
        summary = _summarise(items, notes)
        text = " ".join(lines) + ("\n\n" + summary if summary else "")

        return ProviderResponse(
            items=items,
            text=text,
            rationale=rationale or ["No changes were required."],
            provider=self.name,
        )


def _summarise(items: list[MealItem], notes: dict[str, str]) -> str:
    if not items:
        return ""
    parts = []
    for item in items:
        phrase = phrase_for(item.form, notes)
        suffix = f" ({phrase})" if phrase and phrase != "as is" else ""
        parts.append(f"{item.quantity_g:.0f} g {_short(item.name).lower()}{suffix}")
    return "The meal becomes: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def build_prompt(request: TranslationRequest) -> str:
    """The prompt every LLM provider sends. One place, so they stay comparable."""
    profile = request.profile
    config = load_age_config(profile.age_group)

    planned = (
        "; ".join(
            f"{i.quantity_g:.0f} g {i.name} ({i.form.value})" for i in request.planned_meal.items
        )
        or "(empty)"
    )
    optimized = (
        "; ".join(
            f"{i.quantity_g:.0f} g {i.name} [food_id={i.food_id}] ({i.form.value})"
            for i in request.optimized_meal.items
        )
        or "(empty)"
    )

    candidate_lines = (
        "\n".join(f"  {query}: {', '.join(names)}" for query, names in request.candidates.items())
        or "  (none)"
    )

    fixed = "; ".join(request.violations_fixed) or "none"
    flags = ", ".join(f.value for f in profile.health_flags) or "none"

    return f"""You are writing a short, practical meal recommendation for a caregiver.

WHO IS EATING
  Life stage: {config.label}{f" ({profile.age_months} months old)" if profile.age_months and profile.age_group.value == "toddler" else ""}
  Goal: {profile.goal.value.replace("_", " ")}
  Health flags: {flags}

WHAT THEY PLANNED TO EAT
  {planned}

WHAT THE OPTIMISER PRODUCED (this is the answer; do not change the foods or amounts)
  {optimized}

SAFETY ISSUES THAT WERE REPAIRED
  {fixed}

REAL USDA FOODS RETRIEVED FOR GROUNDING (use these exact names and ids)
{candidate_lines}

RULES
1. Return ONLY the optimised meal above. Do not invent foods, change quantities,
   or change preparation forms. You are describing a decision, not making one.
2. Use the exact food_id values given.
3. `form` must be one of: {", ".join(f.value for f in Form)}.
4. `text` is 2-4 short sentences addressed to the caregiver, saying what to change
   and, where a safety issue was repaired, why.
5. `rationale` is one short string per change.

Respond with ONE JSON object and nothing else -- no markdown fence, no preamble:
{JSON_CONTRACT}"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str, request: TranslationRequest, provider: str) -> ProviderResponse | None:
    """Validate a provider's JSON against the contract, or return ``None``.

    Deliberately strict about *structure* and forgiving about *packaging*: models
    wrap JSON in prose or a markdown fence often enough that refusing those would
    waste a retry on a cosmetic problem, but a missing field or an unknown form
    is a real contract breach and Stage 4 should never see it.
    """
    if not raw or not raw.strip():
        return None
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None

    valid_forms = {f.value for f in Form}
    items: list[MealItem] = []
    for entry in payload["items"]:
        if not isinstance(entry, dict):
            return None
        try:
            form = str(entry.get("form", "whole"))
            items.append(
                MealItem(
                    food_id=str(entry["food_id"]),
                    name=str(entry["name"]),
                    quantity_g=float(entry["quantity_g"]),
                    form=Form(form) if form in valid_forms else Form.WHOLE,
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    if not items:
        return None

    rationale = payload.get("rationale") or []
    if isinstance(rationale, str):
        rationale = [rationale]

    return ProviderResponse(
        items=items,
        text=str(payload.get("text") or "").strip(),
        rationale=[str(r) for r in rationale],
        provider=provider,
        raw=raw,
    )


#: Wall-clock ceiling on a single provider call. Every network path must have
#: one: a demo that hangs because the venue's Wi-Fi is captive-portalled is worse
#: than a demo that quietly uses the offline template, and without an explicit
#: timeout the SDKs will wait far longer than anyone standing in front of an
#: audience will tolerate.
PROVIDER_TIMEOUT_S = 20.0


class _RetryingLLMProvider(LLMProvider):
    """Shared control flow: try, validate, retry once, fall back to the template."""

    max_retries = 1

    def generate(self, request: TranslationRequest) -> ProviderResponse:
        prompt = build_prompt(request)
        last_error = self.unavailable_reason() if not self.available else None

        if self.available:
            for attempt in range(self.max_retries + 1):
                try:
                    raw = self._complete(prompt, attempt)
                except Exception as exc:
                    # Deliberately not retried. The retry exists for a model that
                    # replied with something unparseable, where asking again
                    # plausibly helps. An exception here is transport -- no key, no
                    # route, a timeout -- and asking again just spends the timeout
                    # a second time before reaching the same fallback. Bounding the
                    # wait matters more than a second chance the network will not
                    # give us.
                    last_error = f"{type(exc).__name__}: {exc}"
                    break
                parsed = parse_response(raw, request, self.name)
                if parsed is not None:
                    return parsed
                last_error = "invalid JSON response"

        fallback = TemplateProvider().generate(request)
        fallback.provider = self.name
        fallback.fallback_used = True
        fallback.error = last_error
        return fallback

    @abstractmethod
    def _complete(self, prompt: str, attempt: int) -> str: ...


#: Model strings checked against the published model list at
#: https://platform.claude.com/docs/en/docs/about-claude/models/overview on
#: 2026-08-26. Recorded rather than remembered: a wrong model string is a 404 in
#: front of an audience, and this file is the kind that gets written once and
#: read a year later.
#:
#: Current at that date: claude-fable-5, claude-opus-5, claude-sonnet-5,
#: claude-haiku-4-5(-20251001). Legacy but still served: claude-opus-4-8/4-7/4-6,
#: claude-sonnet-4-6, claude-opus-4-5, claude-sonnet-4-5.
ANTHROPIC_MODELS_CHECKED_ON = "2026-08-26"


class AnthropicProvider(_RetryingLLMProvider):
    """Claude via the official Anthropic SDK.

    Note on ``temperature``. The Messages API deprecated the sampling parameters:
    models released after Claude Opus 4.6 reject any temperature other than 1.0
    with a 400. The design brief asks for 0.2, so it is sent only to models known
    to accept it and omitted everywhere else.

    The direction of that test matters and is the reason it is an allow-list
    rather than a deny-list. Omitting the parameter is accepted by every model
    that ever existed; sending it to a model that has dropped it is a hard
    failure. An unrecognised model string -- a newer one, or a typo -- therefore
    has to fall on the side that cannot 400.
    """

    name = "anthropic"

    #: Models that still accept sampling parameters, i.e. those released up to and
    #: including Claude Opus 4.6. Matched as a prefix so dated snapshots
    #: ("claude-haiku-4-5-20251001") resolve the same way as their aliases.
    #: Anything not listed here is sent *without* a temperature.
    _SAMPLING_MODELS = (
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    )

    #: Default is a current model rather than the newest or the cheapest. Stage 3
    #: writes a short strict-JSON object from a diff that is already computed, so
    #: frontier reasoning buys nothing here; what it needs is a model that is not
    #: about to be retired. Sonnet 5 does not take a temperature, which is why the
    #: guard above exists and is exercised by the shipped default.
    #:
    #: **Docs-verified, not runtime-verified.** This id was checked against the
    #: published model list on the date above and the request shape is pinned by
    #: stubbed-client tests, but no live call has been made from this repository --
    #: there is no API key in the development environment. The offline template
    #: path is unaffected and is what every command, test and experiment uses. If
    #: a live check is wanted, set ANTHROPIC_API_KEY and run
    #: `foodsense recommend --provider anthropic`; a wrong id would surface as a
    #: 404 recorded in `trace.warnings`, with the template answer still returned.
    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY")) and _installed("anthropic")

    def unavailable_reason(self) -> str:
        if not _installed("anthropic"):
            return "anthropic package not installed (pip install -r requirements-optional.txt)"
        return "ANTHROPIC_API_KEY not set"

    def _complete(self, prompt: str, attempt: int) -> str:
        import anthropic

        # max_retries=0: the SDK's own retry budget would multiply the ceiling
        # above, and this class already owns the retry policy.
        client = anthropic.Anthropic(timeout=PROVIDER_TIMEOUT_S, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.model.startswith(self._SAMPLING_MODELS):
            kwargs["temperature"] = self.temperature
        if attempt:
            kwargs["messages"][0]["content"] = (
                prompt + "\n\nYour previous reply was not valid JSON. "
                "Reply with the JSON object only."
            )
        response = client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIProvider(_RetryingLLMProvider):
    """GPT via the official OpenAI SDK, using JSON mode where available."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY")) and _installed("openai")

    def unavailable_reason(self) -> str:
        if not _installed("openai"):
            return "openai package not installed (pip install -r requirements-optional.txt)"
        return "OPENAI_API_KEY not set"

    def _complete(self, prompt: str, attempt: int) -> str:
        from openai import OpenAI

        content = (
            prompt
            if not attempt
            else (prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only.")
        )
        response = OpenAI(timeout=PROVIDER_TIMEOUT_S, max_retries=0).chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content or ""


class OllamaProvider(_RetryingLLMProvider):
    """A local model over Ollama's HTTP API. No key, no cloud."""

    name = "ollama"

    def __init__(
        self, model: str | None = None, host: str | None = None, temperature: float = 0.2
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature

    @property
    def available(self) -> bool:
        """Whether an Ollama server is actually reachable, not merely configured."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2):
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def unavailable_reason(self) -> str:
        return f"no Ollama server reachable at {self.host}"

    def _complete(self, prompt: str, attempt: int) -> str:
        import urllib.request

        content = (
            prompt
            if not attempt
            else (prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only.")
        )
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": content,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_S) as response:
            return json.loads(response.read()).get("response", "")


def _installed(package: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(package) is not None


PROVIDERS: dict[str, type[LLMProvider]] = {
    "template": TemplateProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str = "template", **kwargs: Any) -> LLMProvider:
    """Construct a provider by name."""
    try:
        return PROVIDERS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown provider {name!r}. Available: {', '.join(PROVIDERS)}") from None
