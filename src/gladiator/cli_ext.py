from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from gladiator import __version__
from gladiator.config import config_path, load_config
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.setup.wizard import run_setup

app = typer.Typer(no_args_is_help=True, help="GladiatorAgent - Telegram-first mini-swe-agent runtime")
console = Console()


@app.command()
def setup() -> None:
    run_setup()


@app.command()
def status() -> None:
    path = config_path()
    if not path.exists():
        console.print("Not configured. Run: gladiator setup")
        raise typer.Exit(code=1)
    config = load_config(path)
    console.print(f"Gladiator {__version__}")
    console.print(f"Provider: {config.provider.base_url}")
    console.print(f"Model: {config.provider.model}")
    console.print(f"Reasoning: {config.provider.reasoning_effort}")
    console.print(f"YOLO: {'on' if config.runtime.yolo else 'off'}")


@app.command()
def run(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    path = config_path()
    if not path.exists():
        run_setup()
    config = load_config(path)
    service = ExtendedGladiatorService(config=config, config_path=path, workspace=workspace)
    try:
        asyncio.run(service.run_forever())
    except KeyboardInterrupt:
        console.print("Stopped Gladiator.")


@app.command()
def version() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()
