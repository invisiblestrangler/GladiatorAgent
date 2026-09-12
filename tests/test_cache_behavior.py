from __future__ import annotations

import json

from gladiator.cache_session import CacheStats
from gladiator.models.openai_streaming import OpenAICompatibleStreamingModel
from gladiator.runtime.session import load_or_create_session, rotate_session


class FakeResponse:
    def __init__(self):
        self.headers = {"X-OpenRouter-Cache-Status": "MISS"}

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


def test_openrouter_cache_transport(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", FakeClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        model_name="test/model",
        session_id="gladiator-stable-session",
    )

    response = model.query(
        [
            {"role": "system", "content": "stable-prefix"},
            {"role": "user", "content": "do the work"},
        ]
    )

    assert response["extra"]["actions"][0]["command"] == "true"
    _, _, request = FakeClient.last_request
    assert request["json"]["session_id"] == "gladiator-stable-session"
    assert request["json"]["stream_options"] == {"include_usage": True}
    assert request["headers"]["x-session-id"] == "gladiator-stable-session"
    assert request["headers"]["X-OpenRouter-Cache"] == "true"
    assert request["headers"]["X-OpenRouter-Cache-TTL"] == "300"
    assert model.last_response_cache_status == "MISS"
    assert model.last_prompt_cache_ratio == 0.8
    assert model.cumulative_prompt_cache_ratio == 0.8
    assert model.total_cache_write_tokens == 10


def test_non_openrouter_does_not_add_openrouter_cache_controls(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", FakeClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
        session_id="same-session",
    )
    model.query([{"role": "user", "content": "work"}])
    _, _, request = FakeClient.last_request
    assert "session_id" not in request["json"]
    assert "stream_options" not in request["json"]
    assert "X-OpenRouter-Cache" not in request["headers"]
    assert "x-session-id" not in request["headers"]


def test_session_persists_and_rotates(tmp_path):
    path = tmp_path / "session.json"
    first = load_or_create_session(path)
    restored = load_or_create_session(path)
    assert restored.session_id == first.session_id

    replacement = rotate_session(path)
    assert replacement.session_id != first.session_id
    assert load_or_create_session(path).session_id == replacement.session_id


def test_cache_stats_accumulate():
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
