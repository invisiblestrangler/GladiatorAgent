from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field, SecretStr

ReasoningEffort = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)
TraceMode = Literal["off", "milestones", "verbose"]
SearchMode = Literal["none", "local_searxng", "existing_searxng"]
ProviderMode = Literal["openai_compatible", "codex_oauth"]


class ProviderConfig(BaseModel):
    name: str = "default"
    base_url: str
    api_key: SecretStr
    model: str
    reasoning_effort: ReasoningEffort = "high"
    # context_window is the provider-advertised usable/current window used for
    # compaction safety. max_context_window records a distinct advertised ceiling
    # when the model catalog exposes one.
    context_window: int | None = None
    max_context_window: int | None = None
    context_window_source: str | None = None
    mode: ProviderMode = "openai_compatible"
    codex_access_token: SecretStr = SecretStr("")
    codex_account_id: str | None = None
    codex_responses_url: str = "https://chatgpt.com/backend-api/codex/responses"


class MentorConfig(BaseModel):
    """Rare, read-only advisory model configuration.

    Mentor policy/context is assembled only when the agent explicitly invokes the
    mentor pseudo-command, keeping the main agent's normal prompt/cache prefix small.
    """

    enabled: bool = False
    model: str | None = None
    reasoning_effort: ReasoningEffort = "high"
    max_files: int = Field(default=8, ge=1, le=24)
    max_file_chars: int = Field(default=24_000, ge=1_000, le=200_000)
    max_total_context_chars: int = Field(default=80_000, ge=4_000, le=500_000)
    max_advice_chars: int = Field(default=12_000, ge=1_000, le=50_000)


class TelegramConfig(BaseModel):
    bot_token: SecretStr
    allowed_user_ids: list[int] = Field(default_factory=list)


class SearchConfig(BaseModel):
    mode: SearchMode = "none"
    searxng_url: str | None = None


class BrowserConfig(BaseModel):
    enabled: bool = False
    backend: Literal["browser-use"] = "browser-use"
    command: str | None = None


class RuntimeConfig(BaseModel):
    yolo: bool = True
    escalation_timeout_seconds: int = 3600
    compact_threshold_tokens: int = 300_000
    compact_fraction_of_model_window: float = 0.82
    trace_mode: TraceMode = "milestones"
    shell_observation_char_limit: int = 12_000
    web_observation_char_limit: int = 12_000
    search_result_limit: int = 5
    telegram_input_debounce_seconds: float = Field(default=1.5, ge=0.05, le=10.0)
    telegram_input_max_burst_seconds: float = Field(default=5.0, ge=0.1, le=30.0)


class GladiatorConfig(BaseModel):
    provider: ProviderConfig
    telegram: TelegramConfig
    search: SearchConfig = Field(default_factory=SearchConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    mentor: MentorConfig = Field(default_factory=MentorConfig)


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
    payload["provider"]["codex_access_token"] = config.provider.codex_access_token.get_secret_value()
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
