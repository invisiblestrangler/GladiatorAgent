from __future__ import annotations

from pathlib import Path

import httpx
import typer
from rich.console import Console

from gladiator.config import (
    REASONING_EFFORTS,
    BrowserConfig,
    GladiatorConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
    TelegramConfig,
    config_root,
    save_config,
)
from gladiator.setup.browser import install_browser_use
from gladiator.setup.docker import DockerManager
from gladiator.setup.searxng import SearxngManager

console = Console()


def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _fetch_models(base_url: str, api_key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{_normalize_base_url(base_url)}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    return [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]


def _verify_telegram(token: str) -> dict:
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"https://api.telegram.org/bot{token}/getMe")
        response.raise_for_status()
        body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram rejected bot token: {body}")
    return body.get("result", {})


def _choose_model(base_url: str, api_key: str) -> str:
    if typer.confirm("Fetch available models from the endpoint?", default=True):
        try:
            models = _fetch_models(base_url, api_key)
        except Exception as exc:  # setup should stay usable with nonstandard OpenAI-compatible endpoints
            console.print(f"[yellow]Could not list models: {exc}[/yellow]")
        else:
            if models:
                console.print(f"Found {len(models)} model(s).")
                preview = models[:20]
                for idx, name in enumerate(preview, start=1):
                    console.print(f"  {idx:>2}. {name}")
                raw = typer.prompt("Model number or model ID", default=preview[0])
                if raw.isdigit() and 1 <= int(raw) <= len(preview):
                    return preview[int(raw) - 1]
                return raw
    return typer.prompt("Default model")


def run_setup(*, destination: Path | None = None) -> Path:
    console.print("[bold]Gladiator Setup[/bold]\n")

    base_url = _normalize_base_url(typer.prompt("OpenAI-compatible endpoint (include /v1 if required)"))
    api_key = typer.prompt("API key", hide_input=True)
    model = _choose_model(base_url, api_key)
    reasoning_choices = "/".join(REASONING_EFFORTS)
    reasoning = typer.prompt(
        f"Default reasoning ({reasoning_choices})",
        default="high",
    ).lower()
    if reasoning not in REASONING_EFFORTS:
        raise typer.BadParameter(f"Unsupported reasoning setting. Choose one of: {reasoning_choices}")

    bot_token = typer.prompt("Telegram bot token", hide_input=True)
    try:
        bot = _verify_telegram(bot_token)
        console.print(f"[green]Telegram bot verified:[/green] @{bot.get('username', 'unknown')}")
    except Exception as exc:
        if not typer.confirm(f"Telegram verification failed ({exc}). Save configuration anyway?", default=False):
            raise typer.Abort()

    search_mode = typer.prompt(
        "Web search (none/local/existing)",
        default="local",
    ).strip().lower()
    if search_mode not in {"none", "local", "existing"}:
        raise typer.BadParameter("Choose none, local, or existing")

    search = SearchConfig()
    if search_mode == "existing":
        search = SearchConfig(mode="existing_searxng", searxng_url=typer.prompt("Existing SearXNG URL").rstrip("/"))
    elif search_mode == "local":
        if not DockerManager.available():
            if typer.confirm("Docker is not installed. Install Docker now?", default=True):
                result = DockerManager.install()
                if result.returncode != 0:
                    console.print("[red]Docker installation failed; continuing without local search.[/red]")
                elif not DockerManager.available():
                    console.print("[yellow]Docker was installed but is not yet available in this shell. Re-run setup later.[/yellow]")
            else:
                console.print("[yellow]Skipping local SearXNG because Docker is unavailable.[/yellow]")
        if DockerManager.available():
            manager = SearxngManager(config_root())
            try:
                manager.write_settings()
                manager.install_or_restart()
            except Exception as exc:
                console.print(f"[yellow]SearXNG setup failed: {exc}. Search will remain disabled.[/yellow]")
            else:
                search = SearchConfig(mode="local_searxng", searxng_url=manager.url)
                console.print(f"[green]Local-only SearXNG configured at {manager.url}[/green]")

    browser = BrowserConfig(enabled=False)
    if typer.confirm("Install optional local browser automation (browser-use)?", default=False):
        try:
            browser_command = install_browser_use()
        except Exception as exc:
            console.print(f"[yellow]Browser automation install failed: {exc}[/yellow]")
        else:
            browser = BrowserConfig(enabled=True, command=browser_command)
            console.print("[green]browser-use installed.[/green]")

    config = GladiatorConfig(
        provider=ProviderConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning,  # type: ignore[arg-type]
        ),
        telegram=TelegramConfig(bot_token=bot_token),
        search=search,
        browser=browser,
        runtime=RuntimeConfig(yolo=True, escalation_timeout_seconds=3600),
    )
    saved = save_config(config, path=destination)
    console.print(f"\n[green bold]Setup complete.[/green bold] Configuration saved securely to {saved}")
    console.print("Gladiator runs YOLO by default. It only pauses for genuinely consequential uncertainty;")
    console.print("if you do not answer within one hour it resumes using the conservative option.")
    return saved
