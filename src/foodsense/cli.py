"""FoodSense command line.

    foodsense demo                                   # all three scenarios
    foodsense recommend --scenario toddler_choking   # one scenario, in detail
    foodsense recommend --scenario elderly_sodium --provider anthropic
    foodsense scenarios                              # what is available
    foodsense version

The demo path is deliberately the offline one: the default Stage-3 provider is the
deterministic template, so ``foodsense demo`` works with no key and no network.
``--provider`` opts into an LLM and falls back to the template if it is unreachable,
so even that path cannot fail in front of an audience.
"""

from __future__ import annotations

import contextlib
import json
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from foodsense.schemas import PipelineTrace


def _make_console_safe() -> None:
    """Stop a Windows console codepage from killing the demo.

    The default Windows codepage is cp1252, which cannot encode the box-drawing
    and spinner glyphs Rich emits -- ``foodsense recommend`` died with a
    UnicodeEncodeError on a Braille spinner character before this. Re-encoding
    with ``errors="replace"`` means the worst case is a substituted character
    rather than a stack trace in front of an audience.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


_make_console_safe()

app = typer.Typer(
    add_completion=False,
    help="FoodSense -- availability-aware counterfactual food recommendation.",
    no_args_is_help=True,
)
console = Console()

#: ASCII spinner, for the same reason.
SPINNER = "line"

_CHANGE_STYLE = {
    "added": ("green", "+"),
    "removed": ("red", "-"),
    "modified": ("yellow", "~"),
    "unchanged": ("dim", " "),
}


@app.callback()
def main() -> None:
    """Root callback, so Typer stays in multi-command mode."""


@app.command()
def version() -> None:
    """Print the installed FoodSense version."""
    from foodsense import __version__

    typer.echo(f"foodsense {__version__}")


@app.command()
def scenarios() -> None:
    """List the built-in demo scenarios."""
    from foodsense.scenarios import SCENARIOS

    table = Table(title="Demo scenarios", show_lines=True, title_justify="left")
    table.add_column("key", style="bold cyan", no_wrap=True)
    table.add_column("who / what")
    table.add_column("what it demonstrates")
    for scenario in SCENARIOS.values():
        table.add_row(scenario.key, scenario.title, scenario.expectation)
    console.print(table)


@app.command()
def recommend(
    scenario: str = typer.Option(
        None, "--scenario", "-s", help="Built-in scenario key; see `foodsense scenarios`."
    ),
    provider: str = typer.Option(
        "template",
        "--provider",
        "-p",
        help=(
            "Stage-3 text generator. Default 'template' is fully offline and needs "
            "no API key or network. 'anthropic' | 'openai' | 'ollama' opt into an "
            "LLM and fall back to the template if it is unreachable."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the raw PipelineTrace as JSON."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only the final recommendation."),
) -> None:
    """Run the four-stage pipeline on a scenario and show the whole trace."""
    from foodsense.pipeline import run_scenario
    from foodsense.scenarios import SCENARIOS, scenario_names
    from foodsense.stage3_rag.providers import get_provider

    if scenario is None:
        console.print(
            "[red]--scenario is required.[/red] Available: " + ", ".join(scenario_names())
        )
        raise typer.Exit(code=2)
    if scenario not in SCENARIOS:
        console.print(
            f"[red]Unknown scenario {scenario!r}.[/red] Available: {', '.join(scenario_names())}"
        )
        raise typer.Exit(code=2)

    try:
        chosen = get_provider(provider)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    if provider != "template" and not chosen.available:
        console.print(
            f"[yellow]{provider} is unavailable ({chosen.unavailable_reason()}); "
            f"the deterministic template will be used instead.[/yellow]"
        )

    with console.status(f"Running the pipeline on {scenario}...", spinner=SPINNER):
        trace = run_scenario(scenario, provider=chosen)

    if as_json:
        typer.echo(trace.model_dump_json(indent=2))
        return
    _render(trace, quiet=quiet)


@app.command()
def demo(
    provider: str = typer.Option(
        "template",
        "--provider",
        "-p",
        help="Offline by default; an LLM provider is opt-in. See `recommend --help`.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run all three demo scenarios end to end. Offline, no API key."""
    from foodsense.pipeline import run_scenario
    from foodsense.scenarios import SCENARIOS
    from foodsense.stage3_rag.providers import get_provider

    chosen = get_provider(provider)
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold]FoodSense[/bold] -- availability-aware, verification-guided\n"
                "counterfactual food recommendation\n\n"
                f"Stage-3 provider: [cyan]{provider}[/cyan]"
                + ("" if provider == "template" else " (falls back to template on failure)")
            ),
            border_style="cyan",
        )
    )

    failures = 0
    for scenario in SCENARIOS.values():
        console.rule(f"[bold cyan]{scenario.key}[/bold cyan] -- {scenario.title}")
        console.print(f"[dim]{scenario.expectation}[/dim]\n")
        trace = run_scenario(scenario.key, provider=chosen)
        _render(trace, quiet=quiet)
        if not trace.succeeded:
            failures += 1

    console.rule("[bold]Summary[/bold]")
    if failures:
        console.print(
            f"[red]{failures} of {len(SCENARIOS)} scenarios did not pass verification.[/red]"
        )
        raise typer.Exit(code=1)
    console.print(
        f"[green]All {len(SCENARIOS)} scenarios completed and passed Stage-4 verification.[/green]"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(trace: PipelineTrace, quiet: bool = False) -> None:
    if not quiet:
        _render_stage1(trace)
        _render_stage2(trace)
    _render_recommendation(trace)
    _render_stage4(trace)
    for warning in trace.warnings:
        console.print(f"[yellow]note:[/yellow] {warning}")
    console.print()


def _render_stage1(trace: PipelineTrace) -> None:
    stage1 = trace.stage1
    if stage1 is None:
        return
    evaluation = stage1.rule_evaluation
    verdict = "[green]safe[/green]" if evaluation and evaluation.is_safe else "[red]UNSAFE[/red]"
    console.print(
        f"[bold]Stage 1 - prediction[/bold]  suitability [cyan]{stage1.suitability:.3f}[/cyan]"
        f"  |  rule score {evaluation.score:.3f} ({verdict})"
        f"  |  {stage1.nutrients.energy_kcal:.0f} kcal, "
        f"{stage1.nutrients.sodium_mg:.0f} mg sodium"
    )
    if evaluation:
        for violation in evaluation.hard_violations:
            console.print(f"    [red]hazard[/red] {violation.message}")


def _render_stage2(trace: PipelineTrace) -> None:
    stage2 = trace.stage2
    if stage2 is None:
        console.print("[bold]Stage 2 - optimisation[/bold]  [yellow]skipped[/yellow]")
        return
    console.print(
        f"[bold]Stage 2 - counterfactual edit[/bold]  "
        f"{stage2.diff.n_items_changed} edits, L1 {stage2.diff.l1_distance_g:.0f} g, "
        f"searched {stage2.search_space_size} available foods "
        f"({stage2.n_evaluations} evaluations, {stage2.runtime_s:.2f}s)"
    )
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column(" ", width=1)
    table.add_column("food")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("why", style="dim")
    for change in stage2.diff.changes:
        style, marker = _CHANGE_STYLE[change.change_type]
        before = f"{change.old_quantity_g:.0f} g" if change.old_quantity_g else "-"
        after = f"{change.new_quantity_g:.0f} g" if change.new_quantity_g else "-"
        if change.new_form and change.old_form and change.new_form != change.old_form:
            after += f" ({change.new_form.value})"
        table.add_row(
            Text(marker, style=style),
            Text(change.name[:46], style=style),
            before,
            after,
            (change.reason or "")[:52],
        )
    console.print(table)


def _render_recommendation(trace: PipelineTrace) -> None:
    stage3 = trace.stage3
    if stage3 is None:
        return
    label = stage3.provider + (" (fell back to template)" if stage3.fallback_used else "")
    console.print(
        Panel(
            (stage3.text or "").strip() or "(no text produced)",
            title=f"Stage 3 - recommendation [dim]via {label}[/dim]",
            border_style="green",
            title_align="left",
        )
    )


def _render_stage4(trace: PipelineTrace) -> None:
    report = trace.stage4
    if report is None:
        return
    verdict = "[green]PASS[/green]" if report.final_pass else "[red]FAIL[/red]"
    console.print(
        f"[bold]Stage 4 - USDA verification[/bold]  {verdict}  "
        f"{report.matched}/{report.checked} matched, "
        f"{report.n_corrections} corrected, {len(report.safety_fixes)} safety fixes"
    )
    for correction in report.corrected:
        console.print(
            f"    [yellow]corrected[/yellow] {correction.name}: {correction.field} "
            f"{correction.claimed} -> {correction.corrected}"
        )
    for fix in report.safety_fixes:
        console.print(f"    [magenta]{fix.action}[/magenta] {fix.message}")
    for name in report.unmatched:
        console.print(f"    [red]unmatched[/red] {name!r} does not exist in the USDA database")

    table = Table(title="Verified meal", box=None, title_justify="left", title_style="bold")
    table.add_column("amount", justify="right")
    table.add_column("preparation")
    table.add_column("food")
    for item in trace.final_meal.items:
        table.add_row(f"{item.quantity_g:.0f} g", item.form.value, item.name[:52])
    console.print(table)

    nutrients = report.verified_nutrients
    console.print(
        f"    [dim]verified totals: {nutrients.energy_kcal:.0f} kcal, "
        f"{nutrients.protein_g:.0f} g protein, {nutrients.sodium_mg:.0f} mg sodium, "
        f"{nutrients.fiber_g:.1f} g fibre, {nutrients.iron_mg:.1f} mg iron[/dim]"
    )


@app.command()
def trace(
    scenario: str = typer.Option(..., "--scenario", "-s"),
    output: str = typer.Option("-", "--output", "-o", help="File to write, or - for stdout."),
) -> None:
    """Write the full PipelineTrace as JSON, for the report or the API fixtures."""
    from foodsense.pipeline import run_scenario

    payload = json.loads(run_scenario(scenario).model_dump_json())
    text = json.dumps(payload, indent=2)
    if output == "-":
        typer.echo(text)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)
        console.print(f"[green]wrote {output}[/green]")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
