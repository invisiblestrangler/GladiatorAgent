from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from gladiator.config import GladiatorConfig, MentorConfig, ProviderConfig, TelegramConfig
from gladiator.environment import GladiatorLocalEnvironment
from gladiator.mentor import MENTOR_SYSTEM_PROMPT, MentorClient, MentorRequest
from gladiator.service_goal import GoalAwareGladiatorService


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_lines(self):
        chunk = {"choices": [{"delta": {"content": "Use the indexed lookup; keep the fallback path."}}]}
        yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class _FakeCodexResponse(_FakeResponse):
    def iter_lines(self):
        yield "data: " + json.dumps(
            {"type": "response.output_text.delta", "delta": "Prefer the bounded ring buffer."}
        )
        yield "data: [DONE]"


class _CaptureClient:
    payload = None
    headers = None
    url = None
    response_class = _FakeResponse

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stream(self, _method, url, *, headers, json):
        type(self).url = url
        type(self).headers = headers
        type(self).payload = json
        return type(self).response_class()


class _FakeTelegramClient:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, _chat_id: int, text: str, **_kwargs):
        self.sent.append(text)
        return {"message_id": len(self.sent)}


def _provider() -> ProviderConfig:
    return ProviderConfig(
        base_url="https://provider.example/v1",
        api_key=SecretStr("secret"),
        model="main-model",
    )


def test_mentor_openai_request_is_tool_free_and_only_contains_selected_evidence(tmp_path: Path, monkeypatch):
    selected = tmp_path / "algorithm.py"
    unselected = tmp_path / "secrets.txt"
    selected.write_text("def lookup(index, key): return index.get(key)\n", encoding="utf-8")
    unselected.write_text("THIS MUST NOT BE SENT", encoding="utf-8")
    mentor = MentorConfig(enabled=True, model="mentor-model", reasoning_effort="high")
    _CaptureClient.response_class = _FakeResponse
    monkeypatch.setattr("gladiator.mentor.httpx.Client", _CaptureClient)

    client = MentorClient(provider=_provider(), mentor=mentor, workspace_root=tmp_path)
    advice = client.consult(MentorRequest(question="Review lookup complexity", files=(selected,)))

    assert advice.startswith("Use the indexed lookup")
    payload = _CaptureClient.payload
    assert payload["model"] == "mentor-model"
    assert payload["messages"][0]["content"] == MENTOR_SYSTEM_PROMPT
    assert "tools" not in payload
    assert "algorithm.py" in payload["messages"][1]["content"]
    assert "def lookup" in payload["messages"][1]["content"]
    assert "THIS MUST NOT BE SENT" not in payload["messages"][1]["content"]


def test_mentor_codex_request_is_direct_tool_free_advice(tmp_path: Path, monkeypatch):
    provider = _provider()
    provider.mode = "codex_oauth"
    provider.codex_access_token = SecretStr("oauth-token")
    provider.codex_account_id = "acct-123"
    provider.codex_responses_url = "https://chatgpt.example/codex/responses"
    mentor = MentorConfig(enabled=True, model="mentor-codex", reasoning_effort="xhigh")
    _CaptureClient.response_class = _FakeCodexResponse
    monkeypatch.setattr("gladiator.mentor.httpx.Client", _CaptureClient)

    client = MentorClient(provider=provider, mentor=mentor, workspace_root=tmp_path)
    advice = client.consult(MentorRequest(question="Review queue design"))

    assert advice == "Prefer the bounded ring buffer."
    assert _CaptureClient.url == provider.codex_responses_url
    assert _CaptureClient.headers["ChatGPT-Account-ID"] == "acct-123"
    assert _CaptureClient.headers["Authorization"] == "Bearer oauth-token"
    assert _CaptureClient.payload["model"] == "mentor-codex"
    assert _CaptureClient.payload["instructions"] == MENTOR_SYSTEM_PROMPT
    assert "tools" not in _CaptureClient.payload


def test_mentor_pseudo_command_is_advice_only_and_receives_explicit_files(tmp_path: Path):
    evidence = tmp_path / "bench.log"
    evidence.write_text("latency=42ms\n", encoding="utf-8")
    captured = {}
    env = object.__new__(GladiatorLocalEnvironment)
    env.workspace_root = tmp_path.resolve()
    env.event_sink = lambda _event: None

    def mentor_handler(request):
        captured["request"] = request
        return "The hot path is allocation-bound."

    env.mentor_handler = mentor_handler
    result = env._mentor_command(
        "gladiator mentor --question 'Review this hot path' --log bench.log",
        cwd=str(tmp_path),
    )

    assert result is not None
    assert result["returncode"] == 0
    assert "<mentor_advice>" in result["output"]
    assert captured["request"].question == "Review this hot path"
    assert captured["request"].logs == (evidence.resolve(),)


def test_compound_mentor_command_is_intercepted_not_executed_as_shell(tmp_path: Path):
    env = object.__new__(GladiatorLocalEnvironment)
    env.workspace_root = tmp_path.resolve()
    env.event_sink = lambda _event: None
    env.mentor_handler = lambda _request: "unused"

    result = env._mentor_command("cd /tmp && gladiator mentor --question 'help'", cwd=str(tmp_path))

    assert result is not None
    assert result["returncode"] == 2
    assert "sole command" in result["output"]


@pytest.mark.asyncio
async def test_telegram_mentor_controls_toggle_and_select_model(tmp_path: Path):
    service = object.__new__(GoalAwareGladiatorService)
    service.config = GladiatorConfig(
        provider=_provider(),
        telegram=TelegramConfig(bot_token=SecretStr("telegram")),
    )
    service.config_path = tmp_path / "config.json"
    client = _FakeTelegramClient()
    service.bot = SimpleNamespace(client=client)

    await service._handle_mentor_command(1, "on")
    await service._handle_mentor_command(1, "model gpt-review")
    await service._handle_mentor_command(1, "reasoning xhigh")

    assert service.config.mentor.enabled is True
    assert service.config.mentor.model == "gpt-review"
    assert service.config.mentor.reasoning_effort == "xhigh"
    assert "gpt-review" in client.sent[-2]
