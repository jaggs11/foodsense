"""Tests for Stage 3 (retrieval + providers) and Stage 4 (verification).

The Stage-4 tests are adversarial by design. Verification is only worth having if
it catches things, so most of these hand it output that is already wrong -- a
hallucinated food, an impossible preparation, an inflated nutrient claim, a
re-introduced hazard -- and check that none of it survives.
"""

from __future__ import annotations

import json

import pytest

from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import get_food_db
from foodsense.schemas import (
    AgeGroup,
    Form,
    Goal,
    HealthFlag,
    Meal,
    MealItem,
    UserProfile,
)
from foodsense.stage3_rag.providers import (
    PROVIDERS,
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    TemplateProvider,
    TranslationRequest,
    build_prompt,
    get_provider,
    parse_response,
)
from foodsense.stage3_rag.retriever import get_retriever
from foodsense.stage3_rag.translate import build_diff
from foodsense.stage4_verification.verifier import DEFAULT_TOLERANCE, verify


@pytest.fixture(scope="module")
def db():
    return get_food_db()


@pytest.fixture(scope="module")
def engine(db):
    return RuleEngine(db=db)


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


@pytest.fixture(scope="module")
def toddler():
    return UserProfile(
        age_group=AgeGroup.TODDLER, age_months=18, weight_kg=11.0, goal=Goal.BALANCED_NUTRITION
    )


@pytest.fixture(scope="module")
def adult():
    return UserProfile(age_group=AgeGroup.ADULT, weight_kg=70.0, goal=Goal.BALANCED_NUTRITION)


def _find(db, query):
    return db.search(query, limit=1)[0][0]


# ---------------------------------------------------------------------------
# Stage 3: retrieval
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_it_indexes_the_whole_database(self, retriever, db):
        assert len(retriever) == len(db)

    def test_it_finds_the_obvious_food(self, retriever):
        names = [hit.name.lower() for hit in retriever.search("grapes", k=5)]
        assert any("grape" in name for name in names)

    def test_results_are_ordered_by_score(self, retriever):
        scores = [hit.score for hit in retriever.search("chicken breast", k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_it_retrieves_on_category_and_tags_too(self, retriever):
        """Names alone would not answer 'low sodium'; the index includes tags."""
        assert retriever.search("low sodium", k=5)

    def test_an_empty_query_returns_nothing(self, retriever):
        assert retriever.search("", k=5) == []
        assert retriever.search("   ", k=5) == []

    def test_ground_returns_a_real_food_or_none(self, retriever, db):
        record = retriever.ground("something like brown rice")
        assert record is None or db.find(record.fdc_id) is not None

    def test_candidates_are_recorded_per_query(self, retriever):
        candidates = retriever.candidates_for(["grapes", "chicken"], k=3)
        assert set(candidates) == {"grapes", "chicken"}
        assert all(len(v) <= 3 for v in candidates.values())

    def test_retrieval_differs_from_stage4_matching(self, retriever, db):
        """They answer different questions, so they must not be the same code.

        BM25 ranks by term overlap; the Stage-4 matcher weighs precision, recall
        and preparation qualifiers. Sharing one would make verification partly
        circular -- grading the retriever against the retriever's own notion of
        similarity.
        """
        assert retriever.search.__qualname__ != db.match.__qualname__


# ---------------------------------------------------------------------------
# Stage 3: providers
# ---------------------------------------------------------------------------


@pytest.fixture
def request_for(db, toddler):
    grapes = db.by_hazard("grape")[0]
    rice = _find(db, "rice white long grain cooked")
    planned = Meal(items=[grapes.as_item(40.0, Form.WHOLE), rice.as_item(50.0)])
    optimized = Meal(items=[grapes.as_item(40.0, Form.QUARTERED), rice.as_item(70.0)])
    diff = build_diff(planned, optimized, [])
    return TranslationRequest(
        profile=toddler,
        planned_meal=planned,
        optimized_meal=optimized,
        changes=[c.model_dump(mode="json") for c in diff.edits],
        texture_notes={"quartered": "cut into small quarters, lengthwise"},
    )


class TestTemplateProvider:
    def test_it_needs_nothing(self):
        assert TemplateProvider().available

    def test_it_returns_the_optimised_items_unchanged(self, request_for):
        response = TemplateProvider().generate(request_for)
        assert response.items == request_for.optimized_meal.items
        assert not response.fallback_used

    def test_it_is_deterministic(self, request_for):
        first = TemplateProvider().generate(request_for)
        second = TemplateProvider().generate(request_for)
        assert first.text == second.text and first.rationale == second.rationale

    def test_it_uses_the_age_appropriate_phrasing(self, request_for):
        text = TemplateProvider().generate(request_for).text.lower()
        assert "quarter" in text

    def test_it_says_so_when_nothing_changed(self, toddler, db):
        meal = Meal(items=[_find(db, "carrots cooked").as_item(80.0)])
        response = TemplateProvider().generate(
            TranslationRequest(profile=toddler, planned_meal=meal, optimized_meal=meal)
        )
        assert "no changes" in response.text.lower()

    def test_it_never_returns_empty_text(self, request_for):
        assert TemplateProvider().generate(request_for).text.strip()


class TestProviderContract:
    @pytest.mark.parametrize("name", list(PROVIDERS))
    def test_every_provider_is_constructible(self, name):
        provider = get_provider(name)
        assert provider.name == name

    @pytest.mark.parametrize("name", ["anthropic", "openai", "ollama"])
    def test_unavailable_providers_explain_themselves(self, name):
        provider = get_provider(name)
        if not provider.available:
            assert provider.unavailable_reason()

    @pytest.mark.parametrize("name", ["anthropic", "openai", "ollama"])
    def test_an_unavailable_provider_falls_back_rather_than_failing(self, name, request_for):
        """The central robustness claim: a provider failure degrades, never breaks."""
        provider = get_provider(name)
        if provider.available:
            pytest.skip(f"{name} is configured; the fallback path is not exercised")
        response = provider.generate(request_for)
        assert response.fallback_used
        assert response.provider == name
        assert response.items == request_for.optimized_meal.items
        assert response.text.strip()
        assert response.error

    def test_a_broken_provider_falls_back(self, request_for, monkeypatch):
        """Even a provider that raises mid-call must not take the pipeline down."""
        provider = OpenAIProvider()
        monkeypatch.setattr(type(provider), "available", property(lambda self: True))
        monkeypatch.setattr(
            provider,
            "_complete",
            lambda prompt, attempt: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        response = provider.generate(request_for)
        assert response.fallback_used and "RuntimeError" in (response.error or "")

    def test_a_provider_returning_junk_falls_back(self, request_for, monkeypatch):
        provider = OllamaProvider()
        monkeypatch.setattr(type(provider), "available", property(lambda self: True))
        monkeypatch.setattr(provider, "_complete", lambda prompt, attempt: "I'm afraid I can't.")
        response = provider.generate(request_for)
        assert response.fallback_used and response.error == "invalid JSON response"

    def test_a_valid_json_reply_is_accepted(self, request_for, monkeypatch):
        provider = AnthropicProvider()
        monkeypatch.setattr(type(provider), "available", property(lambda self: True))
        payload = json.dumps(
            {
                "items": [
                    {
                        "name": i.name,
                        "food_id": i.food_id,
                        "quantity_g": i.quantity_g,
                        "form": i.form.value,
                    }
                    for i in request_for.optimized_meal.items
                ],
                "text": "Quarter the grapes.",
                "rationale": ["grapes: choking hazard"],
            }
        )
        monkeypatch.setattr(provider, "_complete", lambda prompt, attempt: payload)
        response = provider.generate(request_for)
        assert not response.fallback_used
        assert response.text == "Quarter the grapes."

    def test_it_retries_once_before_falling_back(self, request_for, monkeypatch):
        provider = AnthropicProvider()
        monkeypatch.setattr(type(provider), "available", property(lambda self: True))
        attempts: list[int] = []

        def _complete(prompt, attempt):
            attempts.append(attempt)
            return "not json"

        monkeypatch.setattr(provider, "_complete", _complete)
        provider.generate(request_for)
        assert attempts == [0, 1]

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("gemini")


class TestTheAnthropicSamplingGuard:
    """``temperature`` is sent to models that take it and to no others.

    The Messages API deprecated the sampling parameters: a model released after
    Claude Opus 4.6 rejects any temperature but 1.0 with a 400. The design brief
    asks for 0.2. So the request has to be built differently per model, and the
    consequence of getting it wrong is a hard failure in front of an audience --
    which makes it worth a test that never touches the network.

    The stub stands in for ``anthropic.Anthropic`` and records the kwargs it was
    handed, so these assert the *request that would be sent*, not a response.
    """

    @staticmethod
    def _sent(monkeypatch, model: str) -> dict:
        import sys
        import types

        captured: dict = {}

        class _Messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                block = types.SimpleNamespace(
                    type="text", text='{"items": [], "text": "", "rationale": []}'
                )
                return types.SimpleNamespace(content=[block])

        class _Anthropic:
            def __init__(self, *args, **kwargs):
                self.messages = _Messages()

        stub = types.ModuleType("anthropic")
        stub.Anthropic = _Anthropic
        monkeypatch.setitem(sys.modules, "anthropic", stub)

        AnthropicProvider(model=model, temperature=0.2)._complete("prompt", 0)
        return captured

    def test_it_is_omitted_for_a_model_that_rejects_it(self, monkeypatch):
        """Sonnet 5 is the shipped default, so the guard is exercised by default."""
        sent = self._sent(monkeypatch, "claude-sonnet-5")
        assert "temperature" not in sent
        assert sent["model"] == "claude-sonnet-5"

    def test_it_is_included_for_a_model_that_accepts_it(self, monkeypatch):
        sent = self._sent(monkeypatch, "claude-sonnet-4-6")
        assert sent["temperature"] == pytest.approx(0.2)

    def test_a_dated_snapshot_matches_its_alias(self, monkeypatch):
        """Model ids are matched as a prefix so pinned snapshots behave identically."""
        sent = self._sent(monkeypatch, "claude-haiku-4-5-20251001")
        assert sent["temperature"] == pytest.approx(0.2)

    def test_an_unrecognised_model_omits_it(self, monkeypatch):
        """The safe direction. Omitting always works; sending can 400.

        A model string this file has never heard of is far more likely to be newer
        than the list than older, so the unknown case must not send a parameter.
        """
        sent = self._sent(monkeypatch, "claude-something-9")
        assert "temperature" not in sent

    def test_the_shipped_default_is_a_model_the_docs_still_list(self):
        """Guards against the default quietly rotting into a retired model id.

        Checked against the published list on the date recorded in the module; if
        this ever needs changing, re-check the list rather than editing the tuple.
        """
        from foodsense.stage3_rag.providers import ANTHROPIC_MODELS_CHECKED_ON

        current = {"claude-fable-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"}
        assert AnthropicProvider.DEFAULT_MODEL in current
        assert ANTHROPIC_MODELS_CHECKED_ON


class TestPromptAndParsing:
    def test_the_prompt_grounds_the_model_in_real_ids(self, request_for):
        prompt = build_prompt(request_for)
        for item in request_for.optimized_meal.items:
            assert item.food_id in prompt
        assert "JSON" in prompt

    def test_the_prompt_lists_the_valid_forms(self, request_for):
        prompt = build_prompt(request_for)
        assert all(form.value in prompt for form in (Form.QUARTERED, Form.MINCED))

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "no json here", "{broken", '{"text": "hi"}', '{"items": []}', '{"items": "x"}'],
    )
    def test_malformed_replies_are_rejected(self, raw, request_for):
        assert parse_response(raw, request_for, "test") is None

    def test_json_wrapped_in_prose_is_accepted(self, request_for):
        """Models fence and preface constantly; that is cosmetic, not a contract breach."""
        body = json.dumps(
            {
                "items": [
                    {"name": "Grapes", "food_id": "1", "quantity_g": 40, "form": "quartered"}
                ],
                "text": "ok",
                "rationale": [],
            }
        )
        parsed = parse_response(f"Sure! ```json\n{body}\n``` hope that helps", request_for, "t")
        assert parsed is not None and parsed.items[0].form is Form.QUARTERED

    def test_an_unknown_form_degrades_rather_than_failing(self, request_for):
        body = json.dumps(
            {
                "items": [{"name": "G", "food_id": "1", "quantity_g": 40, "form": "julienned"}],
                "text": "ok",
                "rationale": [],
            }
        )
        parsed = parse_response(body, request_for, "t")
        assert parsed is not None and parsed.items[0].form is Form.WHOLE

    def test_a_string_rationale_is_normalised_to_a_list(self, request_for):
        body = json.dumps(
            {
                "items": [{"name": "G", "food_id": "1", "quantity_g": 40, "form": "whole"}],
                "text": "ok",
                "rationale": "one reason",
            }
        )
        assert parse_response(body, request_for, "t").rationale == ["one reason"]


# ---------------------------------------------------------------------------
# Stage 4: verification
# ---------------------------------------------------------------------------


class TestVerificationBasics:
    def test_a_clean_meal_passes_untouched(self, db, engine, adult):
        items = [_find(db, "carrots cooked").as_item(100.0), _find(db, "rice brown").as_item(120.0)]
        final, report = verify(items, adult, db=db, engine=engine)
        assert report.final_pass
        assert report.matched == 2 and not report.corrected and not report.safety_fixes
        assert final.food_ids() == {i.food_id for i in items}

    def test_nutrients_are_recomputed_not_taken_on_trust(self, db, engine, adult):
        items = [_find(db, "carrots cooked").as_item(100.0)]
        _, report = verify(items, adult, db=db, engine=engine)
        assert report.verified_nutrients.as_tuple() == db.nutrients_for(items).as_tuple()

    def test_an_empty_list_is_handled(self, db, engine, adult):
        final, report = verify([], adult, db=db, engine=engine)
        assert final.items == [] and report.checked == 0

    def test_the_report_records_its_own_cost(self, db, engine, adult):
        _, report = verify([_find(db, "carrots cooked").as_item(80.0)], adult, db=db, engine=engine)
        assert report.runtime_s >= 0.0


class TestVerificationCatchesHallucinations:
    def test_a_nonexistent_food_is_flagged_and_replaced(self, db, engine, adult):
        """The classic LLM failure: a food that sounds real and is not."""
        ghost = MealItem(
            food_id="9999999", name="Artisanal quinoa-kale power blend", quantity_g=80.0
        )
        final, report = verify([ghost], adult, db=db, engine=engine)
        assert report.unmatched, "a fabricated food was accepted"
        assert "9999999" not in final.food_ids()
        assert all(db.find(i.food_id) is not None for i in final.items)

    def test_a_wrong_id_with_a_real_name_is_recovered_by_name(self, db, engine, adult):
        """A model can name a real food and attach the wrong id; the name saves it."""
        carrots = _find(db, "carrots cooked")
        item = MealItem(food_id="0000000", name=carrots.name, quantity_g=100.0)
        final, report = verify([item], adult, db=db, engine=engine)
        assert final.items[0].food_id == carrots.fdc_id
        assert not report.unmatched

    def test_an_inflated_nutrient_claim_is_corrected(self, db, engine, adult):
        items = [_find(db, "rice brown").as_item(150.0)]
        truth = db.nutrients_for(items)
        _, report = verify(items, adult, claimed_nutrients=truth.scaled(1.9), db=db, engine=engine)
        fields = {c.field for c in report.corrected}
        assert "energy_kcal" in fields
        for correction in report.corrected:
            if correction.field == "energy_kcal":
                assert correction.corrected == pytest.approx(truth.energy_kcal, rel=1e-3)

    def test_a_claim_inside_tolerance_is_left_alone(self, db, engine, adult):
        items = [_find(db, "rice brown").as_item(150.0)]
        truth = db.nutrients_for(items)
        _, report = verify(
            items,
            adult,
            claimed_nutrients=truth.scaled(1.0 + DEFAULT_TOLERANCE * 0.5),
            db=db,
            engine=engine,
        )
        assert not [c for c in report.corrected if c.field != "form"]

    def test_an_impossible_form_is_corrected(self, db, engine, adult):
        """Forms are text to a model; nothing stops it writing 'pureed crackers'."""
        record = _find(db, "crackers whole-wheat")
        bad = next(f for f in Form if f not in record.allowed_forms)
        final, report = verify(
            [MealItem(food_id=record.fdc_id, name=record.name, quantity_g=30.0, form=bad)],
            adult,
            db=db,
            engine=engine,
        )
        assert final.items[0].form in record.allowed_forms
        assert any(c.field == "form" for c in report.corrected)


class TestVerificationCatchesSafety:
    def test_a_reintroduced_hazard_is_repaired(self, db, engine, toddler):
        """The case that matters most: a rewrite silently undoing a safety fix."""
        grapes = db.by_hazard("grape")[0]
        final, report = verify([grapes.as_item(40.0, Form.WHOLE)], toddler, db=db, engine=engine)
        assert report.flagged, "a whole grape reached a toddler unflagged"
        assert report.safety_fixes
        assert final.items[0].form is Form.QUARTERED
        assert report.final_pass

    def test_a_hazard_with_no_safe_form_is_removed(self, db, engine, toddler):
        popcorn = db.by_hazard("popcorn")[0]
        staple = _find(db, "rice white long grain cooked")
        final, report = verify(
            [popcorn.as_item(20.0), staple.as_item(60.0)], toddler, db=db, engine=engine
        )
        assert popcorn.fdc_id not in final.food_ids()
        assert any(f.action == "remove" for f in report.safety_fixes)
        assert report.final_pass

    def test_the_same_item_is_fine_for_an_adult(self, db, engine, adult):
        """The repair belongs to the profile, not to the food."""
        grapes = db.by_hazard("grape")[0]
        final, report = verify([grapes.as_item(40.0, Form.WHOLE)], adult, db=db, engine=engine)
        assert final.items[0].form is Form.WHOLE
        assert not report.safety_fixes and report.final_pass

    def test_a_medication_exclusion_is_caught(self, db, engine):
        profile = UserProfile(
            age_group=AgeGroup.OLDER_ADULT, weight_kg=70.0, health_flags=[HealthFlag.STATIN]
        )
        grapefruit = db.by_tag("grapefruit")[0]
        final, report = verify([grapefruit.as_item(150.0)], profile, db=db, engine=engine)
        assert report.flagged
        assert grapefruit.fdc_id not in final.food_ids()

    def test_multiple_hazards_are_all_repaired(self, db, engine, toddler):
        grapes = db.by_hazard("grape")[0]
        popcorn = db.by_hazard("popcorn")[0]
        final, report = verify(
            [grapes.as_item(40.0, Form.WHOLE), popcorn.as_item(15.0)],
            toddler,
            db=db,
            engine=engine,
        )
        assert report.final_pass
        assert engine.evaluate(final, toddler).is_safe
        assert len(report.safety_fixes) >= 2


class TestDiff:
    def test_it_classifies_every_kind_of_change(self, db):
        carrots = _find(db, "carrots cooked")
        rice = _find(db, "rice brown")
        yogurt = _find(db, "yogurt plain")
        planned = Meal(items=[carrots.as_item(100.0), rice.as_item(80.0)])
        optimized = Meal(items=[carrots.as_item(150.0), yogurt.as_item(60.0)])
        diff = build_diff(planned, optimized, [])
        kinds = {c.food_id: c.change_type for c in diff.changes}
        assert kinds[carrots.fdc_id] == "modified"
        assert kinds[rice.fdc_id] == "removed"
        assert kinds[yogurt.fdc_id] == "added"
        assert diff.n_items_changed == 3

    def test_an_identical_meal_has_no_edits(self, db):
        meal = Meal(items=[_find(db, "carrots cooked").as_item(100.0)])
        diff = build_diff(meal, meal, [])
        assert diff.edits == [] and diff.l1_distance_g == 0.0

    def test_a_form_change_alone_counts(self, db):
        grapes = db.by_hazard("grape")[0]
        planned = Meal(items=[grapes.as_item(40.0, Form.WHOLE)])
        optimized = Meal(items=[grapes.as_item(40.0, Form.QUARTERED)])
        diff = build_diff(planned, optimized, [])
        assert diff.n_items_changed == 1 and diff.l1_distance_g == 0.0

    def test_a_violation_becomes_the_reason_for_its_edit(self, db, engine, toddler):
        grapes = db.by_hazard("grape")[0]
        planned = Meal(items=[grapes.as_item(40.0, Form.WHOLE)])
        optimized = Meal(items=[grapes.as_item(40.0, Form.QUARTERED)])
        violations = engine.evaluate(planned, toddler).violations
        diff = build_diff(planned, optimized, violations)
        change = next(c for c in diff.changes if c.food_id == grapes.fdc_id)
        assert change.reason and "choking" in change.reason.lower()


class TestTheSecondRepairPass:
    """The pass that exists because one item can break two rules at once.

    Re-forming answers the rule about *preparation*. It cannot answer a rule about
    the food itself, so an item can survive the first pass still unsafe. These
    assert the three things that make the Stage-4 headline claim mean anything:
    the offending item is genuinely gone from the returned meal, the safety-fix
    log describes that same meal rather than an earlier draft of it, and
    ``final_pass`` is only True when nothing unsafe survived.
    """

    @pytest.fixture(scope="class")
    def double_bind(self, db):
        """A frankfurter for someone with dysphagia *and* an MAOI prescription.

        Mincing it satisfies the texture rule and does nothing about the tyramine,
        which is exactly the shape the second pass is for. Both foods are real
        rows in the curated database, pinned by id.
        """
        profile = UserProfile(
            age_group=AgeGroup.OLDER_ADULT,
            age_months=70 * 12,
            weight_kg=68.0,
            goal=Goal.BALANCED_NUTRITION,
            health_flags=[HealthFlag.DYSPHAGIA, HealthFlag.MAOI],
        )
        frankfurter = db.find("174614")
        rice = db.find("168878")
        assert frankfurter is not None and rice is not None
        items = [
            MealItem(
                food_id=frankfurter.fdc_id,
                name=frankfurter.name,
                quantity_g=60.0,
                form=Form.WHOLE,
            ),
            MealItem(food_id=rice.fdc_id, name=rice.name, quantity_g=120.0, form=Form.SOFT_COOKED),
        ]
        return profile, items, frankfurter.fdc_id

    def test_the_setup_really_does_break_two_rules_at_once(self, double_bind, engine):
        """Guard: if only one rule fired, the first pass would settle it."""
        profile, items, _ = double_bind
        rules = {v.rule_id for v in engine.evaluate(items, profile).hard_violations}
        assert "flag.dysphagia.texture" in rules
        assert "flag.maoi.excluded_food" in rules

    def test_the_removed_item_is_absent_from_the_returned_meal(self, double_bind, db, engine):
        profile, items, offender = double_bind
        final, _ = verify(items, profile, db=db, engine=engine)
        assert offender not in final.food_ids()
        assert final.items, "the repair removed the whole meal"

    def test_the_log_and_the_meal_describe_the_same_plate(self, double_bind, db, engine):
        """A log claiming a food was re-formed, for a food that is not there, is a lie.

        The first pass minces the frankfurter and records it; the second pass then
        removes it. Leaving both entries in would tell a reader the meal contains
        minced frankfurter, so the superseded entry is dropped.
        """
        profile, items, _ = double_bind
        final, report = verify(items, profile, db=db, engine=engine)
        served = final.food_ids()
        for fix in report.safety_fixes:
            if fix.action == "remove":
                assert fix.food_id not in served, f"{fix.name} logged removed but still served"
            elif fix.action == "reform":
                assert fix.food_id in served, f"{fix.name} logged re-formed but not served"

    def test_final_pass_is_true_only_when_nothing_unsafe_survived(self, double_bind, db, engine):
        profile, items, _ = double_bind
        final, report = verify(items, profile, db=db, engine=engine)
        assert report.final_pass is engine.evaluate(final, profile).is_safe

    def test_an_unfixable_meal_reports_failure_rather_than_looping(self, db, engine):
        """Numeric hard rules have no offending item, so no repair can reach them.

        The honest outcome is ``final_pass=False`` with the violation flagged --
        not a silent pass, and not an unbounded repair loop.
        """
        profile = UserProfile(
            age_group=AgeGroup.OLDER_ADULT,
            age_months=70 * 12,
            weight_kg=68.0,
            goal=Goal.BALANCED_NUTRITION,
            health_flags=[HealthFlag.HYPERTENSION],
        )
        soup = db.find("172882")
        assert soup is not None
        items = [
            MealItem(food_id=soup.fdc_id, name=soup.name, quantity_g=400.0, form=Form.SOFT_COOKED)
        ]
        final, report = verify(items, profile, db=db, engine=engine)
        assert not report.final_pass
        assert any(v.rule_id == "flag.hypertension.sodium_mg" for v in report.flagged)
        assert final.items, "an unrepairable numeric rule must not empty the plate"
