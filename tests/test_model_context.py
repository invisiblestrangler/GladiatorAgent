from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from gladiator.config import GladiatorConfig, ProviderConfig, TelegramConfig
from gladiator.model_context import (
    apply_model_context,
    clear_model_context,
    discover_model_context,
    extract_model_context,
)
from gladiator.service_goal import GoalAwareGladiatorService


def _provider() -> ProviderConfig:
    return ProviderConfig(
        base_url="https://provider.example/v1",
        api_key=SecretStr("secret"),
        model="model-x",
    )


def test_extracts_hermes_style_context_length():
    info = extract_model_context(
        {"id": "model-x", "context_length": 131_072},
        model="model-x",
        source="api:/models",
    )

    assert info is not None
    assert info.context_window == 131_072
    assert info.max_context_window == 131_072


def test_preserves_distinct_current_and_maximum_context():
    info = extract_model_context(
        {"slug": "model-x", "context_window": 272_000, "max_context_window": 872_000},
        model="model-x",
        source="api:/models",
    )

    assert info is not None
    assert info.context_window == 272_000
    assert info.max_context_window == 872_000
    assert info.has_distinct_maximum is True


def test_reads_nested_provider_metadata_without_using_ambiguous_max_tokens():
    info = extract_model_context(
        {
            "id": "model-x",
            "max_tokens": 8_192,
            "metadata": {"limits": {"max_model_len": "262144"}},
        },
        model="model-x",
        source="api:/models",
    )

    assert info is not None
    assert info.context_window == 262_144
    assert info.max_context_window == 262_144


class _Response:
    def __init__(self, body, status_code: int = 200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


class _GenericClient:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url, *, headers, params=None):
        del headers, params
        if url.endswith("/models/model-x"):
            return _Response({}, 404)
        assert url.endswith("/models")
        return _Response({"data": [{"id": "model-x", "context_length": 196_608}]})


class _CodexClient(_GenericClient):
    def get(self, url, *, headers, params=None):
        assert headers["Authorization"] == "Bearer oauth-token"
        assert headers["ChatGPT-Account-ID"] == "acct-1"
        assert url.endswith("/models")
        assert params and "client_version" in params
        return _Response(
            {
                "models": [
                    {
                        "slug": "model-x",
                        "context_window": 272_000,
                        "max_context_window": 872_000,
                    }
                ]
            }
        )


def test_discovers_openai_compatible_models_list(monkeypatch):
    monkeypatch.setattr("gladiator.model_context.httpx.Client", _GenericClient)
    info = discover_model_context(_provider())

    assert info is not None
    assert info.context_window == 196_608
    assert info.max_context_window == 196_608
    assert info.source == "api:/models"


def test_discovers_codex_catalog_maximum(monkeypatch):
    monkeypatch.setattr("gladiator.model_context.httpx.Client", _CodexClient)
    provider = _provider()
    provider.mode = "codex_oauth"
    provider.codex_access_token = SecretStr("oauth-token")
    provider.codex_account_id = "acct-1"

    info = discover_model_context(provider)

    assert info is not None
    assert info.context_window == 272_000
    assert info.max_context_window == 872_000


def test_clear_prevents_stale_window_from_following_provider_switch():
    provider = _provider()
    info = extract_model_context(
        {"id": "model-x", "context_window": 500_000},
        model="model-x",
        source="api:/models",
    )
    assert info is not None
    apply_model_context(provider, info)

    clear_model_context(provider)

    assert provider.context_window is None
    assert provider.max_context_window is None
    assert provider.context_window_source is None


@pytest.mark.asyncio
async def test_model_command_refreshes_limits_and_updates_agent(monkeypatch, tmp_path):
    provider = _provider()
    config = GladiatorConfig(
        provider=provider,
        telegram=TelegramConfig(bot_token=SecretStr("telegram")),
    )
    sent: list[str] = []
    service = object.__new__(GoalAwareGladiatorService)
    service.config = config
    service.config_path = tmp_path / "config.json"
    service.agent = SimpleNamespace(config=SimpleNamespace(model_context_window=None))
    service.bot = SimpleNamespace(client=SimpleNamespace(send_message=None))

    async def send_message(_chat_id, text, **_kwargs):
        sent.append(text)
        return {"message_id": len(sent)}

    service.bot.client.send_message = send_message
    service._refresh_model_settings = lambda: setattr(
        service.agent.config, "model_context_window", service.config.provider.context_window
    )

    def fake_discover(selected):
        assert selected.model == "new-model"
        return extract_model_context(
            {"id": "new-model", "context_window": 400_000, "max_context_window": 1_000_000},
            model="new-model",
            source="api:/models",
        )

    monkeypatch.setattr("gladiator.service_goal.discover_model_context", fake_discover)

    await service._handle_model_command(1, "new-model")

    assert provider.model == "new-model"
    assert provider.context_window == 400_000
    assert provider.max_context_window == 1_000_000
    assert service.agent.config.model_context_window == 400_000
    assert "1,000,000" in sent[-1]
