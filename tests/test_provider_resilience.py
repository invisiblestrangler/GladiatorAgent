from __future__ import annotations

import json
import time

import pytest
from minisweagent.exceptions import Submitted

from gladiator.events import AgentEvent, EventKind
from gladiator.models.openai_streaming import OpenAICompatibleStreamingModel, ProviderRequestError


class FakeResponse:
    headers: dict[str, str] = {}
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b""


class TextResponse(FakeResponse):
    def iter_lines(self):
        chunk = {"choices": [{"delta": {"content": "Recovered."}, "finish_reason": "stop"}]}
        yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class PreviousResponseIdError(FakeResponse):
    status_code = 400

    def read(self):
        return (
            b'{"code":11133,"extError":{"code":"invalid_value",'
            b'"message":"request parameters were rejected by the model provider",'
            b'"param":"previous_response_id"}}'
        )

    def iter_lines(self):
        raise AssertionError("HTTP 400 must be handled before SSE parsing")


class InvalidRequestError(FakeResponse):
    status_code = 400

    def read(self):
        return b'{"error":{"message":"tool_call_id is malformed","code":"invalid_request"}}'

    def iter_lines(self):
        raise AssertionError("HTTP 400 must be handled before SSE parsing")


class InBandUpstreamError(FakeResponse):
    def iter_lines(self):
        yield 'data: {"error":{"code":400,"message":"upstream error","type":"server_error"}}'
        yield "data: [DONE]"


class EmptyStreamResponse(FakeResponse):
    def iter_lines(self):
        yield 'data: {"usage":{"prompt_tokens":1234},"choices":[]}'
        yield "data: [DONE]"


class RetryThenSuccessClient:
    calls = 0

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return PreviousResponseIdError() if type(self).calls == 1 else TextResponse()


class AlwaysInvalidClient(RetryThenSuccessClient):
    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return InvalidRequestError()


class InBandRetryThenSuccessClient(RetryThenSuccessClient):
    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return InBandUpstreamError() if type(self).calls == 1 else TextResponse()


class AlwaysInBandErrorClient(RetryThenSuccessClient):
    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return InBandUpstreamError()


class AlwaysEmptyStreamClient(RetryThenSuccessClient):
    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return EmptyStreamResponse()


class SlowTextResponse(TextResponse):
    def iter_lines(self):
        time.sleep(0.04)
        yield from super().iter_lines()


class SlowClient(RetryThenSuccessClient):
    def stream(self, *_args, **_kwargs):
        type(self).calls += 1
        return SlowTextResponse()


def _history() -> list[dict]:
    return [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "hello"},
    ]


def test_previous_response_id_400_is_retried_before_stream(monkeypatch):
    RetryThenSuccessClient.calls = 0
    events: list[AgentEvent] = []
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", RetryThenSuccessClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        event_sink=events.append,
        retry_base_seconds=0,
        slow_stream_notice_seconds=0,
    )

    with pytest.raises(Submitted) as submitted:
        model.query(_history())

    assert RetryThenSuccessClient.calls == 2
    assert submitted.value.messages[-1]["extra"]["submission"] == "Recovered."
    status = [event.text for event in events if event.kind == EventKind.STATUS]
    assert any("retrying 1/2" in text for text in status)
    assert any("retry succeeded" in text for text in status)


def test_nonretryable_400_is_not_replayed(monkeypatch):
    AlwaysInvalidClient.calls = 0
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", AlwaysInvalidClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        retry_base_seconds=0,
        slow_stream_notice_seconds=0,
    )

    with pytest.raises(ProviderRequestError, match="tool_call_id is malformed"):
        model.query(_history())

    assert AlwaysInvalidClient.calls == 1


def test_in_band_sse_error_is_retried_and_can_recover(monkeypatch):
    InBandRetryThenSuccessClient.calls = 0
    events: list[AgentEvent] = []
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", InBandRetryThenSuccessClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        event_sink=events.append,
        retry_base_seconds=0,
        slow_stream_notice_seconds=0,
    )

    with pytest.raises(Submitted) as submitted:
        model.query(_history())

    assert InBandRetryThenSuccessClient.calls == 2
    assert submitted.value.messages[-1]["extra"]["submission"] == "Recovered."
    status = [event.text for event in events if event.kind == EventKind.STATUS]
    assert any("retrying 1/2" in text for text in status)
    assert any("retry succeeded" in text for text in status)


def test_in_band_sse_error_surfaces_after_bounded_retries(monkeypatch):
    AlwaysInBandErrorClient.calls = 0
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", AlwaysInBandErrorClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        retry_base_seconds=0,
        slow_stream_notice_seconds=0,
    )

    with pytest.raises(ProviderRequestError, match="upstream error") as failure:
        model.query(_history())

    assert failure.value.status_code == 400
    assert AlwaysInBandErrorClient.calls == 3


def test_empty_stream_is_provider_failure_not_format_error(monkeypatch):
    AlwaysEmptyStreamClient.calls = 0
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", AlwaysEmptyStreamClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        max_retries=0,
        retry_base_seconds=0,
        slow_stream_notice_seconds=0,
    )

    with pytest.raises(ProviderRequestError, match="empty streaming completion") as failure:
        model.query(_history())

    assert failure.value.status_code == 502
    assert "sse_chunks=1" in failure.value.detail
    assert "prompt_tokens=1234" in failure.value.detail
    assert AlwaysEmptyStreamClient.calls == 1


def test_slow_first_stream_notifies_then_recovers(monkeypatch):
    SlowClient.calls = 0
    events: list[AgentEvent] = []
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", SlowClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="key",
        model_name="model",
        event_sink=events.append,
        slow_stream_notice_seconds=0.01,
        retry_base_seconds=0,
    )

    with pytest.raises(Submitted):
        model.query(_history())

    status = [event for event in events if event.kind == EventKind.STATUS]
    assert any(event.data.get("slow_stream") for event in status)
    assert any(event.data.get("slow_stream_recovered") for event in status)
