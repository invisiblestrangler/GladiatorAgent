from __future__ import annotations

import json

import pytest
from minisweagent.exceptions import FormatError, Submitted

from gladiator.cache_session import CacheStats
from gladiator.models.openai_streaming import OpenAICompatibleStreamingModel
from gladiator.runtime.session import load_or_create_session, rotate_session


class FakeResponse:
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "true"}),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10},
            },
        }
        yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class FakeClient:
    last_request = None

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stream(self, method, url, **kwargs):
        type(self).last_request = (method, url, kwargs)
        return FakeResponse()


class FakeTextResponse(FakeResponse):
    def iter_lines(self):
        chunk = {
            "choices": [{"delta": {"content": "Finished cleanly."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50},
        }
        yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class FakeTextClient(FakeClient):
    def stream(self, method, url, **kwargs):
        type(self).last_request = (method, url, kwargs)
        return FakeTextResponse()


def test_provider_request_has_no_vendor_specific_cache_controls(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", FakeClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )

    response = model.query(
        [
            {"role": "system", "content": "stable-prefix", "extra": {"timestamp": 123}},
            {"role": "user", "content": "do the work"},
        ]
    )

    assert response["extra"]["actions"][0]["command"] == "true"
    assert response["extra"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    _, _, request = FakeClient.last_request
    payload = request["json"]
    headers = request["headers"]

    assert payload["messages"][0] == {"role": "system", "content": "stable-prefix"}
    assert set(payload) == {"model", "messages", "tools", "tool_choice", "stream", "reasoning_effort"}
    assert set(headers) == {"Authorization", "Content-Type"}


def test_text_only_response_after_tool_work_is_final_submission(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", FakeTextClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )
    history = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "do the work"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            "extra": {"actions": [{"command": "true", "tool_call_id": "call-1"}]},
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]

    with pytest.raises(Submitted) as submitted:
        model.query(history)

    message = submitted.value.messages[0]
    assert message["role"] == "exit"
    assert message["extra"]["exit_status"] == "Submitted"
    assert message["extra"]["submission"] == "Finished cleanly."


def test_text_only_response_before_any_tool_work_remains_format_error(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", FakeTextClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )

    with pytest.raises(FormatError):
        model.query(
            [
                {"role": "system", "content": "stable-prefix"},
                {"role": "user", "content": "do the work"},
            ]
        )


def test_api_visible_prefix_is_stable_when_history_is_appended():
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )
    original = [
        {"role": "system", "content": "stable-prefix", "extra": {"ui": "ignored"}},
        {"role": "user", "content": "first"},
    ]
    first = model._prepare_messages_for_api(original)
    second = model._prepare_messages_for_api(original + [{"role": "assistant", "content": "later"}])
    assert second[: len(first)] == first


def test_session_persists_and_rotates_locally(tmp_path):
    path = tmp_path / "session.json"
    first = load_or_create_session(path)
    restored = load_or_create_session(path)
    assert restored.session_id == first.session_id

    replacement = rotate_session(path)
    assert replacement.session_id != first.session_id
    assert load_or_create_session(path).session_id == replacement.session_id


def test_cache_stats_accumulate_when_provider_reports_them():
    stats = CacheStats()
    stats.add_usage(
        {
            "prompt_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 150, "cache_write_tokens": 25},
        }
    )
    stats.add_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 100}})
    assert stats.prompt_tokens == 300
    assert stats.cached_tokens == 250
    assert stats.cache_write_tokens == 25
    assert stats.hit_ratio == 250 / 300
