"""FoodSense command line.

    foodsense recommend --scenario toddler_choking
    foodsense demo

TODO(Phase 4): implement. A placeholder Typer app exists so that
``pip install -e .`` registers the ``foodsense`` entry point from Phase 0 onward.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="FoodSense -- availability-aware counterfactual food recommendation.",
)


@app.callback()
def main() -> None:
    """Root callback.

    Its only job is to keep Typer in multi-command mode while ``version`` is the
    single registered command, so that ``foodsense recommend`` and ``foodsense demo``
    slot in unchanged in Phase 4.
    """


@app.command()
def version() -> None:
    """Print the installed FoodSense version."""
    from foodsense import __version__

    typer.echo(f"foodsense {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
