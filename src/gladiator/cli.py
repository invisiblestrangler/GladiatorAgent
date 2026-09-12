from __future__ import annotations

import typer
from rich.console import Console

from gladiator import __version__
from gladiator.config import config_path, load_config
from gladiator.setup.wizard import run_setup

app = typer.Typer(no_args_is_help=True, help="GladiatorAgent — a Telegram-first mini-swe-agent runtime")
console = Console()


@app.command()
def setup() -> None:
    """Interactive first-run/reconfiguration wizard."""
    run_setup()


@app.command()
def status() -> None:
    """Show local configuration status without exposing secrets."""
    path = config_path()
    if not path.exists():
        console.print("[yellow]Not configured.[/yellow] Run: gladiator setup")
        raise typer.Exit(code=1)
    config = load_config(path)
    console.print(f"Gladiator {__version__}")
    console.print(f"Provider: {config.provider.base_url}")
    console.print(f"Model: {config.provider.model}")
    console.print(f"Reasoning: {config.provider.reasoning_effort}")
    console.print(f"YOLO: {'on' if config.runtime.yolo else 'off'}")
    console.print(f"Escalation timeout: {config.runtime.escalation_timeout_seconds}s")
    console.print(f"Compact target: {config.runtime.compact_threshold_tokens:,} tokens")
    console.print(f"Search: {config.search.mode} ({config.search.searxng_url or 'disabled'})")
    console.print(f"Browser automation: {'enabled' if config.browser.enabled else 'not installed'}")


@app.command()
def run() -> None:
    """Start Gladiator (Telegram runtime wiring is under active development)."""
    path = config_path()
    if not path.exists():
        console.print("[yellow]Gladiator is not configured yet. Starting setup.[/yellow]")
        run_setup()
    console.print("[yellow]Telegram runtime wiring is not enabled in this bootstrap slice yet.[/yellow]")


@app.command()
def version() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()
