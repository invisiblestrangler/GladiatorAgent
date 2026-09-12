from pathlib import Path

import pytest
from pydantic import ValidationError

from gladiator.config import REASONING_EFFORTS, GladiatorConfig, ProviderConfig, TelegramConfig, load_config, save_config


def test_config_round_trip(tmp_path: Path):
    target = tmp_path / "config.json"
    original = GladiatorConfig(
        provider=ProviderConfig(base_url="http://localhost:8000/v1", api_key="secret", model="model-x"),
        telegram=TelegramConfig(bot_token="telegram-secret"),
    )
    save_config(original, target)
    loaded = load_config(target)
    assert loaded.provider.api_key.get_secret_value() == "secret"
    assert loaded.telegram.bot_token.get_secret_value() == "telegram-secret"
    assert loaded.runtime.yolo is True
    assert loaded.runtime.escalation_timeout_seconds == 3600


def test_reasoning_levels_include_max_and_ultra():
    assert REASONING_EFFORTS == ("off", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    assert ProviderConfig(base_url="http://localhost/v1", api_key="secret", model="m", reasoning_effort="max").reasoning_effort == "max"
    assert ProviderConfig(base_url="http://localhost/v1", api_key="secret", model="m", reasoning_effort="ultra").reasoning_effort == "ultra"


def test_unknown_reasoning_level_is_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig(base_url="http://localhost/v1", api_key="secret", model="m", reasoning_effort="extreme")
