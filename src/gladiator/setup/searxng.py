from __future__ import annotations

import secrets
import subprocess
import time
from pathlib import Path

import httpx
import yaml


class SearxngManager:
    container_name = "gladiator-searxng"
    image = "searxng/searxng:latest"
    port = 8888

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.settings_path = config_dir / "searxng-settings.yml"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def write_settings(self) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "use_default_settings": True,
            "server": {
                "bind_address": "0.0.0.0",
                "port": 8080,
                "secret_key": secrets.token_hex(32),
                "limiter": False,
                "image_proxy": False,
            },
            "search": {"formats": ["html", "json"]},
        }
        self.settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        return self.settings_path

    def install_or_restart(self) -> None:
        if not self.settings_path.exists():
            self.write_settings()
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--restart",
            "unless-stopped",
            "-p",
            f"127.0.0.1:{self.port}:8080",
            "-v",
            f"{self.settings_path.resolve()}:/etc/searxng/settings.yml:ro",
            self.image,
        ]
        subprocess.run(command, check=True)
        self.wait_until_ready()

    def wait_until_ready(self, timeout_seconds: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.url}/search", params={"q": "gladiator setup check", "format": "json"}, timeout=3.0
                )
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
        raise RuntimeError(f"SearXNG did not become ready within {timeout_seconds:.0f}s: {last_error or 'no HTTP 200'}")
