import json

import pytest
from minisweagent.exceptions import Submitted

from gladiator.agent import SYSTEM_TEMPLATE
from gladiator.models.openai_streaming import OpenAICompatibleStreamingModel


class TextResponse:
    status_code = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b""

    def iter_lines(self):
        chunk = {
            "choices": [{"delta": {"content": "Hello! How can I help?"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
        yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class TextClient:
    last_request = None

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def stream(self, method, url, **kwargs):
        type(self).last_request = (method, url, kwargs)
        return TextResponse()


def test_direct_hello_is_a_real_assistant_turn_not_a_fake_tool_completion(monkeypatch):
    monkeypatch.setattr("gladiator.models.openai_streaming.httpx.Client", TextClient)
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )
    initial = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "hello"},
    ]

    with pytest.raises(Submitted) as submitted:
        model.query(initial)

    assistant, terminal = submitted.value.messages
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Hello! How can I help?"
    assert "tool_calls" not in assistant
    assert terminal["role"] == "exit"
    assert terminal["extra"]["submission"] == "Hello! How can I help?"

    # Gladiator removes only its local exit marker before the next Telegram turn.
    # The prior assistant answer must remain as ordinary chat history and must not
    # create an incomplete tool transcript.
    continued = [*initial, assistant, {"role": "user", "content": "Now inspect the project."}]
    prepared = model._prepare_messages_for_api(continued)
    assert [message["role"] for message in prepared] == ["system", "user", "assistant", "user"]
    assert prepared[2] == {"role": "assistant", "content": "Hello! How can I help?"}

    _, _, request = TextClient.last_request
    assert "previous_response_id" not in request["json"]


def test_system_prompt_does_not_force_tools_for_conversation():
    assert "do NOT manufacture a bash call" in SYSTEM_TEMPLATE
    assert "If the request requires workspace inspection" in SYSTEM_TEMPLATE
