from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field, SecretStr

ReasoningEffort = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
TraceMode = Literal["off", "milestones", "verbose"]
SearchMode = Literal["none", "local_searxng", "existing_searxng"]


class ProviderConfig(BaseModel):
    name: str = "default"
    base_url: str
    api_key: SecretStr
    model: str
    reasoning_effort: ReasoningEffort = "high"
    context_window: int | None = None


class TelegramConfig(BaseModel):
    bot_token: SecretStr
    allowed_user_ids: list[int] = Field(default_factory=list)


class SearchConfig(BaseModel):
    mode: SearchMode = "none"
    searxng_url: str | None = None


class BrowserConfig(BaseModel):
    enabled: bool = False
    backend: Literal["browser-use"] = "browser-use"


class RuntimeConfig(BaseModel):
    yolo: bool = True
    escalation_timeout_seconds: int = 3600
    compact_threshold_tokens: int = 300_000
    compact_fraction_of_model_window: float = 0.82
    trace_mode: TraceMode = "milestones"
    shell_observation_char_limit: int = 12_000
    web_observation_char_limit: int = 12_000
    search_result_limit: int = 5


class GladiatorConfig(BaseModel):
    provider: ProviderConfig
    telegram: TelegramConfig
    search: SearchConfig = Field(default_factory=SearchConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def config_root() -> Path:
    return Path(user_config_dir("gladiator", appauthor=False))


def data_root() -> Path:
    return Path(user_data_dir("gladiator", appauthor=False))


def config_path() -> Path:
    return config_root() / "config.json"


def save_config(config: GladiatorConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["provider"]["api_key"] = config.provider.api_key.get_secret_value()
    payload["telegram"]["bot_token"] = config.telegram.bot_token.get_secret_value()
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def load_config(path: Path | None = None) -> GladiatorConfig:
    target = path or config_path()
    return GladiatorConfig.model_validate_json(target.read_text(encoding="utf-8"))
