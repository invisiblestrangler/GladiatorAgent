import base64
import json
from pathlib import Path

import pytest

from gladiator.config import GladiatorConfig, ProviderConfig, TelegramConfig, load_config
from gladiator.models import CodexOAuthStreamingModel, OpenAICompatibleStreamingModel
from gladiator.service import GladiatorService
from gladiator.telegram.bot import TelegramBotRuntime


def _jwt(payload: dict) -> str:
    def encode(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


class _FakeTelegramClient:
    def __init__(self):
        self.sent: list[str] = []
        self.deleted: list[tuple[int, int]] = []

    async def send_message(self, _chat_id: int, text: str, **_kwargs):
        self.sent.append(text)
        return {"message_id": len(self.sent)}

    async def delete_message(self, chat_id: int, message_id: int):
        self.deleted.append((chat_id, message_id))
        return True


def _config() -> GladiatorConfig:
    return GladiatorConfig(
        provider=ProviderConfig(
            base_url="https://api.example.test/v1",
            api_key="api-secret",
            model="gpt-test",
        ),
        telegram=TelegramConfig(bot_token="telegram-secret", allowed_user_ids=[123]),
    )


@pytest.mark.asyncio
async def test_telegram_codex_connect_deletes_token_message_and_persists_secret(tmp_path: Path):
    runtime = object.__new__(TelegramBotRuntime)
    runtime.config = _config()
    runtime.config_path = tmp_path / "config.json"
    runtime.client = _FakeTelegramClient()
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})

    await runtime._handle_codex_command(123, {"message_id": 77}, f"connect {token}")

    assert runtime.client.deleted == [(123, 77)]
    assert runtime.config.provider.mode == "codex_oauth"
    assert runtime.config.provider.codex_account_id == "acct-123"
    assert runtime.config.provider.codex_access_token.get_secret_value() == token
    persisted = load_config(runtime.config_path)
    assert persisted.provider.codex_access_token.get_secret_value() == token
    assert "acct-123" not in runtime.client.sent[-1]
    assert token not in runtime.client.sent[-1]


@pytest.mark.asyncio
async def test_telegram_codex_off_keeps_credential_for_later(tmp_path: Path):
    runtime = object.__new__(TelegramBotRuntime)
    runtime.config = _config()
    runtime.config.provider.mode = "codex_oauth"
    runtime.config.provider.codex_access_token = "stored-token"
    runtime.config.provider.codex_account_id = "acct-123"
    runtime.config_path = tmp_path / "config.json"
    runtime.client = _FakeTelegramClient()

    await runtime._handle_codex_command(123, {"message_id": 88}, "off")

    assert runtime.config.provider.mode == "openai_compatible"
    assert load_config(runtime.config_path).provider.codex_access_token.get_secret_value() == "stored-token"


def test_service_uses_codex_transport_without_replacing_gladiator_harness(tmp_path: Path):
    config = _config()
    config.provider.mode = "codex_oauth"
    config.provider.codex_access_token = "oauth-token"
    config.provider.codex_account_id = "acct-123"
    service = GladiatorService(config=config, config_path=tmp_path / "config.json", workspace=tmp_path / "workspace")

    assert isinstance(service.model, CodexOAuthStreamingModel)
    assert service.agent.model is service.model
    assert service.environment.workspace_root == (tmp_path / "workspace").resolve()

    config.provider.mode = "openai_compatible"
    service._refresh_model_settings()
    assert isinstance(service.model, OpenAICompatibleStreamingModel)
    assert not isinstance(service.model, CodexOAuthStreamingModel)
    assert service.agent.model is service.model
