"""Tests for the HTTP layer, and for the demo behaviour it exposes.

Two kinds of test live here and they fail for different reasons on purpose.

**API contract tests** check the shapes: status codes, error envelopes, that the
trace comes back whole, that a missing API key degrades instead of erroring.
These should only fail if the API changes.

**Golden-trace tests** pin what the three demo scenarios actually produce at the
fixed seed -- final items and forms, validity, verification outcome, safety
fixes. They are *supposed* to fail if pipeline semantics change. A failure here
is not a bug report about this file; it is the demo telling you that the thing a
faculty member will see on screen is no longer the thing that was signed off.
When that happens, look at what moved and decide deliberately, then update the
constant.

Everything is offline. No test here touches the network, reads a key, or needs
one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foodsense.api.main import app
from foodsense.stage1_prediction.predict import LIGHTGBM_PATH

needs_model = pytest.mark.skipif(
    not LIGHTGBM_PATH.exists(), reason="Stage-1 model not trained; run `make train`"
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_health_reports_what_a_demo_actually_needs(self, client):
        """Not just 'ok' -- whether the model and the database are really there."""
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["n_foods"] > 1000
        assert body["default_provider"] == "template"
        assert isinstance(body["model_loaded"], bool)

    def test_scenarios_lists_the_three_demo_cases(self, client):
        body = client.get("/api/scenarios").json()
        assert {s["key"] for s in body} == {"toddler_choking", "elderly_sodium", "adult_weight"}
        for scenario in body:
            assert scenario["title"] and scenario["description"]
            assert scenario["n_planned_items"] > 0

    def test_providers_says_why_an_unavailable_one_is_unavailable(self, client):
        """The UI greys these out and shows the reason; a blank reason is a bug."""
        body = client.get("/api/providers").json()
        by_name = {p["name"]: p for p in body}
        assert by_name["template"]["available"] is True
        assert by_name["template"]["is_default"] is True
        for name, info in by_name.items():
            if not info["available"]:
                assert info["reason"], f"{name} is unavailable with no reason given"

    def test_no_provider_reason_leaks_a_secret(self, client):
        """Reasons are shown in the browser, so they must not quote key material."""
        for info in client.get("/api/providers").json():
            assert "sk-" not in info["reason"]
            assert "Bearer" not in info["reason"]

    def test_foods_search_ranks_the_obvious_match_first(self, client):
        body = client.get("/api/foods", params={"q": "grape", "limit": 5}).json()
        assert body and "grape" in body[0]["name"].lower()
        assert body[0]["allowed_forms"]

    def test_foods_rejects_an_absurd_limit(self, client):
        assert client.get("/api/foods", params={"q": "rice", "limit": 5000}).status_code == 422


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


@needs_model
class TestRecommendContract:
    def test_a_scenario_returns_the_whole_trace(self, client):
        """The trace *is* the response. No wrapper, no reshaping."""
        body = client.post("/api/recommend", json={"scenario": "toddler_choking"}).json()
        for stage in ("stage1", "stage2", "stage3", "stage4"):
            assert body[stage] is not None, f"{stage} missing from the response"
        assert body["final_meal"]["items"]
        assert body["final_rule_evaluation"] is not None
        assert "warnings" in body
        assert body["total_runtime_s"] > 0

    def test_every_stage_reports_its_own_runtime(self, client):
        """The UI shows per-stage timing; it has to be in the trace to be shown."""
        body = client.post("/api/recommend", json={"scenario": "elderly_sodium"}).json()
        for stage in ("stage1", "stage2", "stage3", "stage4"):
            assert body[stage]["runtime_s"] >= 0

    def test_an_unknown_scenario_is_404_not_500(self, client):
        response = client.post("/api/recommend", json={"scenario": "brunch"})
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"
        assert "brunch" in response.json()["detail"]

    def test_an_unknown_field_is_rejected_rather_than_ignored(self, client):
        """extra='forbid' on the request models, so a typo is caught not swallowed."""
        response = client.post(
            "/api/recommend", json={"scenario": "toddler_choking", "provder": "template"}
        )
        assert response.status_code == 422
        assert "provder" in response.json()["detail"]

    def test_an_unknown_provider_is_422(self, client):
        response = client.post(
            "/api/recommend", json={"scenario": "toddler_choking", "provider": "gemini"}
        )
        assert response.status_code == 422

    def test_an_empty_body_reports_the_missing_field(self, client):
        response = client.post("/api/recommend", json={})
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_request"

    def test_a_custom_payload_runs(self, client):
        from foodsense.data.fdc import get_food_db
        from foodsense.scenarios import load_scenario

        db = get_food_db()
        scenario = load_scenario("toddler_choking")
        payload = {
            "profile": scenario.profile.model_dump(mode="json", exclude={"age_years"}),
            "planned_meal": {
                "items": [i.model_dump(mode="json") for i in scenario.planned_meal(db).items]
            },
            "pantry": [i.model_dump(mode="json") for i in scenario.pantry_meal(db).items],
        }
        response = client.post("/api/recommend", json=payload)
        assert response.status_code == 200, response.json()
        assert response.json()["final_meal"]["items"]

    def test_a_custom_payload_with_a_fake_food_is_422(self, client):
        payload = {
            "profile": {"age_group": "adult", "age_months": 360, "goal": "balanced_nutrition"},
            "planned_meal": {
                "items": [
                    {
                        "food_id": "0000000",
                        "name": "Unobtainium",
                        "quantity_g": 100,
                        "form": "whole",
                    }
                ]
            },
        }
        response = client.post("/api/recommend", json=payload)
        assert response.status_code == 422
        assert "0000000" in response.json()["detail"]

    def test_an_empty_planned_meal_is_422(self, client):
        payload = {
            "profile": {"age_group": "adult", "age_months": 360, "goal": "balanced_nutrition"},
            "planned_meal": {"items": []},
        }
        assert client.post("/api/recommend", json=payload).status_code == 422


@needs_model
class TestOfflineFirst:
    """The demo's central claim, asserted at the HTTP boundary."""

    def test_the_default_run_uses_no_provider_at_all(self, client):
        body = client.post("/api/recommend", json={"scenario": "toddler_choking"}).json()
        assert body["stage3"]["provider"] == "template"
        assert body["stage3"]["fallback_used"] is False

    def test_an_unavailable_provider_degrades_rather_than_failing(self, client):
        """No key is not an error. It is a 200 with the fallback flag set.

        This is the behaviour a demo depends on: the Wi-Fi being down must change
        the badge in the corner, not the outcome on screen.
        """
        from foodsense.stage3_rag.providers import get_provider

        if get_provider("anthropic").available:  # pragma: no cover - needs a key
            pytest.skip("anthropic is configured; the fallback path is not exercised")

        response = client.post(
            "/api/recommend", json={"scenario": "toddler_choking", "provider": "anthropic"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["stage3"]["fallback_used"] is True
        assert body["stage3"]["fallback_reason"], "the UI needs a reason to show"
        assert any("anthropic" in w for w in body["warnings"])
        assert body["final_meal"]["items"], "the fallback still has to produce a meal"

    def test_the_fallback_answer_is_still_safe(self, client):
        """Degrading must not degrade safety."""
        from foodsense.stage3_rag.providers import get_provider

        if get_provider("openai").available:  # pragma: no cover - needs a key
            pytest.skip("openai is configured")
        body = client.post(
            "/api/recommend", json={"scenario": "toddler_choking", "provider": "openai"}
        ).json()
        assert body["stage4"]["final_pass"] is True


@needs_model
class TestDeterminism:
    def test_the_same_seed_gives_the_same_meal(self, client):
        """Faculty must be able to reproduce the exact run they are looking at."""
        first = client.post("/api/recommend", json={"scenario": "adult_weight", "seed": 42}).json()
        second = client.post("/api/recommend", json={"scenario": "adult_weight", "seed": 42}).json()
        assert first["final_meal"] == second["final_meal"]
        assert first["stage3"]["text"] == second["stage3"]["text"]

    def test_the_seed_is_echoed_back_so_the_ui_can_show_it(self, client):
        body = client.post("/api/recommend", json={"scenario": "adult_weight", "seed": 7}).json()
        assert body["seed"] == 7

    def test_a_negative_seed_is_rejected(self, client):
        response = client.post("/api/recommend", json={"scenario": "adult_weight", "seed": -1})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Golden traces -- these pin the demo
# ---------------------------------------------------------------------------

#: What each scenario produces at seed 42 on the phase-4.5 pipeline.
#:
#: Regenerate deliberately, never reflexively: run the scenario, read the trace,
#: satisfy yourself the change is an improvement, and only then edit. A diff here
#: is the demo changing under you.
GOLDEN = {
    "toddler_choking": {
        "items": [
            ("Grapes, muscadine, raw", "quartered"),
            ("Rice, white, long-grain, regular, enriched, cooked", "soft_cooked"),
            ("Chicken, ground, crumbles, cooked, pan-browned", "ground"),
        ],
        "final_pass": True,
        "safe": True,
        "stage2_valid": True,
        "n_safety_fixes": 0,
        "n_corrections": 0,
    },
    "elderly_sodium": {
        "items": [
            ("Chicken, broilers or fryers, breast, meat only, cooked, roasted", "soft_cooked"),
            ("Crackers, whole-wheat", "whole"),
        ],
        "final_pass": True,
        "safe": True,
        # False on purpose. The meal is safe and hugely improved but does not
        # clear the 0.70 composite, because per-meal micronutrient floors that a
        # single meal cannot meet are part of that score. Pinning it as False is
        # the honest snapshot; pinning it as True would be a wish.
        "stage2_valid": False,
        "n_safety_fixes": 0,
        "n_corrections": 0,
    },
    "adult_weight": {
        "items": [
            ("Fast foods, hamburger; single, regular patty; plain", "whole"),
            (
                "Potatoes, french fried, steak fries, salt added in processing, "
                "frozen, as purchased",
                "whole",
            ),
            ("Beverages, carbonated, cola, regular", "whole"),
            ("Broccoli, raw", "whole"),
        ],
        "final_pass": True,
        "safe": True,
        # Also False, and for a different reason worth keeping straight: here the
        # surrogate believes the target was met (0.7007) and the rule engine
        # scores 0.6793. That 0.021 gap is Stage-1 calibration at the decision
        # boundary, quantified in results/surrogate_boundary.md.
        "stage2_valid": False,
        "n_safety_fixes": 0,
        "n_corrections": 0,
    },
}


@needs_model
@pytest.mark.parametrize("key", sorted(GOLDEN))
class TestGoldenTraces:
    @pytest.fixture(scope="class")
    def traces(self, client):
        return {
            key: client.post("/api/recommend", json={"scenario": key, "seed": 42}).json()
            for key in GOLDEN
        }

    def test_verification_passes(self, traces, key):
        assert traces[key]["stage4"]["final_pass"] is GOLDEN[key]["final_pass"]

    def test_the_final_meal_is_safe(self, traces, key):
        assert traces[key]["final_rule_evaluation"]["is_safe"] is GOLDEN[key]["safe"]

    def test_the_exact_items_and_forms(self, traces, key):
        """The strictest assertion in the suite, and deliberately so."""
        expected = [tuple(pair) for pair in GOLDEN[key]["items"]]
        actual = [(i["name"], i["form"]) for i in traces[key]["final_meal"]["items"]]
        assert actual == expected

    def test_stage2_validity_matches_the_snapshot(self, traces, key):
        """Two of the three are False, and that is recorded rather than hidden.

        A demo that only pins the cases it wins is not pinned at all.
        """
        assert traces[key]["stage2"]["valid"] is GOLDEN[key]["stage2_valid"]

    def test_the_verification_workload_matches_the_snapshot(self, traces, key):
        """Zero on the offline path, and that is the point.

        The template provider emits the optimiser's own items, so Stage 4 has
        nothing to correct. If these ever become non-zero on the template path,
        something upstream started producing claims that disagree with USDA and
        that is worth knowing immediately.
        """
        report = traces[key]["stage4"]
        assert len(report["safety_fixes"]) == GOLDEN[key]["n_safety_fixes"]
        assert report["n_corrections"] == GOLDEN[key]["n_corrections"]

    def test_nothing_unavailable_reached_the_plate(self, traces, key):
        """Extension #1, asserted through the HTTP boundary rather than in-process."""
        from foodsense.data.fdc import get_food_db
        from foodsense.scenarios import load_scenario

        db = get_food_db()
        scenario = load_scenario(key)
        available = scenario.planned_meal(db).food_ids() | scenario.pantry_meal(db).food_ids()
        assert {i["food_id"] for i in traces[key]["final_meal"]["items"]} <= available


@needs_model
class TestTheToddlerHeadline:
    """The single claim the demo is built around, over HTTP."""

    @pytest.fixture(scope="class")
    def trace(self, client):
        return client.post(
            "/api/recommend", json={"scenario": "toddler_choking", "seed": 42}
        ).json()

    def test_the_grapes_survive_quartered(self, trace):
        grapes = [i for i in trace["final_meal"]["items"] if "Grape" in i["name"]]
        assert grapes, "the grapes were removed rather than re-formed"
        assert grapes[0]["form"] == "quartered"

    def test_the_whole_peanuts_are_gone(self, trace):
        assert not [i for i in trace["final_meal"]["items"] if "eanut" in i["name"]]

    def test_the_planned_meal_was_genuinely_unsafe(self, trace):
        """Guard: a scenario that starts safe proves nothing."""
        assert trace["stage1"]["rule_evaluation"]["is_safe"] is False

    def test_the_explanation_names_the_hazard(self, trace):
        text = (trace["stage3"]["text"] + " ".join(trace["stage3"]["rationale"])).lower()
        assert "grape" in text


#: ANSI SGR sequences, which Rich emits whenever it believes it is writing to a
#: terminal.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text or "")


def _plain_output_env() -> dict[str, str]:
    """An environment in which Rich renders plain, wide, unstyled help.

    Without this the assertion below is really a test of the *terminal*, not of
    the CLI. Rich decides for itself whether to emit escape sequences and how
    wide to wrap, and it treats CI as a colour-capable terminal at 80 columns --
    so `serve --help` came back styled and hard-wrapped under GitHub Actions
    while rendering plain and wide on a developer machine. The literal being
    searched for then straddled a wrap and the test failed in CI only.

    Pinning NO_COLOR, TERM and COLUMNS makes the subprocess render the same way
    everywhere, and dropping the variables Rich uses to force terminal mode stops
    the host CI from overriding that.
    """
    env = dict(os.environ)
    for forcing in ("FORCE_COLOR", "GITHUB_ACTIONS", "TTY_COMPATIBLE", "CLICOLOR_FORCE"):
        env.pop(forcing, None)
    env.update(NO_COLOR="1", TERM="dumb", COLUMNS="200", LINES="50")
    return env


class TestTheApiIsImportableTheWayTheCliImportsIt:
    """The API must be reachable from the *installed distribution*, not the repo.

    This exists because it was not. `api/` lived at the repository root and was
    never packaged, while `pyproject.toml` put "." on pytest's `pythonpath` so
    the suite could still `from api.main import app`. The result was 538 passing
    tests against an import path the shipped CLI did not have: `foodsense serve`
    worked when the working directory happened to be the repo root and raised
    ModuleNotFoundError everywhere else.

    So these assert the property the old tests assumed: that the import works
    because the package is installed, not because of where the process started.
    """

    def test_the_api_module_lives_inside_the_installed_package(self):
        """A path under site-packages, or under src/ for an editable install.

        Either way it is inside `foodsense`, which is what makes it importable
        from any working directory.
        """
        import foodsense
        import foodsense.api.main

        package_root = Path(foodsense.__file__).resolve().parent
        module = Path(foodsense.api.main.__file__).resolve()
        assert package_root in module.parents

    def test_importing_it_does_not_depend_on_the_working_directory(self):
        """Import it in a subprocess started somewhere else entirely.

        A subprocess is the only honest way to check this: the module is already
        in `sys.modules` here, and this test process was started from the repo
        root with pytest's own path setup, so an in-process import proves
        nothing about the shipped CLI.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import foodsense.api.main as m; print(m.app.title)"],
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
        )
        assert result.returncode == 0, result.stderr
        assert "FoodSense" in result.stdout

    def test_the_cli_can_resolve_what_serve_needs_from_elsewhere(self):
        """`serve` imports FRONTEND_DIST at call time; that is the line that broke."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from foodsense.api.main import FRONTEND_DIST; print(FRONTEND_DIST)",
            ],
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()

    def test_serve_help_works_from_another_directory(self):
        """End to end through the CLI entry point, from a directory that is not ours."""
        result = subprocess.run(
            [sys.executable, "-m", "foodsense.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            # Decoded as UTF-8 explicitly. The CLI re-encodes its own streams so
            # Rich box-drawing survives a cp1252 console; a parent decoding with
            # the ambient Windows codepage then chokes on those same bytes and
            # returns a None stdout, which looks exactly like a crash and is not.
            encoding="utf-8",
            errors="replace",
            cwd=tempfile.gettempdir(),
            env=_plain_output_env(),
        )
        assert result.returncode == 0, result.stderr
        assert "--port" in _strip_ansi(result.stdout)

    def test_the_frontend_path_is_anchored_to_the_module_not_the_cwd(self):
        """Resolved by walking up from the module file, so cwd cannot change it."""
        from foodsense.api.main import _find_frontend_dist

        here = _find_frontend_dist()
        original = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            assert _find_frontend_dist() == here
        finally:
            os.chdir(original)
