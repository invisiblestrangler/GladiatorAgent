from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from gladiator import __version__
from gladiator.config import config_path, load_config
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.setup.service import BackgroundServiceManager
from gladiator.setup.wizard import run_setup
from gladiator.todo import TodoManager

app = typer.Typer(no_args_is_help=True, help="GladiatorAgent - Telegram-first mini-swe-agent runtime")
todo_app = typer.Typer(no_args_is_help=True, help="Manage the current workspace task ledger")
service_app = typer.Typer(no_args_is_help=True, help="Manage the persistent background Gladiator service")
app.add_typer(todo_app, name="todo")
app.add_typer(service_app, name="service")
console = Console()


def _workspace_todo() -> TodoManager:
    return TodoManager(Path.cwd() / ".gladiator" / "todo.json")


def _service_manager(workspace: Path) -> BackgroundServiceManager:
    return BackgroundServiceManager(workspace)


def _require_config() -> None:
    if config_path().exists():
        return
    console.print("[red]Gladiator is not configured.[/red] Run: gladiator setup")
    raise typer.Exit(code=1)


def _service_action(action) -> None:
    try:
        message = action()
    except Exception as exc:
        console.print(f"[red]Service operation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if message:
        console.print(message)


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


@service_app.command("install")
def service_install(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Install and start Gladiator as a supervised background service."""
    _require_config()
    manager = _service_manager(workspace)
    _service_action(manager.install)


@service_app.command("status")
def service_status(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Show background-service status."""
    manager = _service_manager(workspace)
    try:
        state = manager.status()
    except Exception as exc:
        console.print(f"[red]Service operation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Installed: {'yes' if state.installed else 'no'}")
    console.print(f"Active: {'yes' if state.active else 'no'}")
    console.print(f"Autostart: {'yes' if state.enabled else 'no'}")
    if state.detail:
        console.print(state.detail)


@service_app.command("start")
def service_start(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Start the installed Gladiator background service."""
    _service_action(_service_manager(workspace).start)


@service_app.command("stop")
def service_stop(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Stop the installed Gladiator background service."""
    _service_action(_service_manager(workspace).stop)


@service_app.command("restart")
def service_restart(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Restart the installed Gladiator background service."""
    _service_action(_service_manager(workspace).restart)


@service_app.command("uninstall")
def service_uninstall(workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w")) -> None:
    """Disable and remove the Gladiator background service."""
    _service_action(_service_manager(workspace).uninstall)


@todo_app.command("show")
def todo_show() -> None:
    """Show the current workspace task ledger."""
    console.print(_workspace_todo().render())


@todo_app.command("add")
def todo_add(text: str) -> None:
    """Add one concise TODO item."""
    item = _workspace_todo().add(text)
    console.print(f"Added TODO #{item.id}: {item.text}")


@todo_app.command("done")
def todo_done(item_id: int) -> None:
    """Mark one TODO item complete."""
    item = _workspace_todo().mark_done(item_id)
    console.print(f"Completed TODO #{item.id}: {item.text}")


@todo_app.command("clear")
def todo_clear() -> None:
    """Clear the current workspace task ledger."""
    _workspace_todo().clear()
    console.print("Cleared TODO ledger.")


@app.command()
def version() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()
