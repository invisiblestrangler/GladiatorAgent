import base64
import json

import pytest
from minisweagent.exceptions import Submitted

from gladiator.models.codex_oauth import CodexOAuthStreamingModel, extract_chatgpt_account_id


def _jwt(payload: dict) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def _model() -> CodexOAuthStreamingModel:
    return CodexOAuthStreamingModel(
        access_token="token",
        account_id="acct-123",
        model_name="gpt-test",
        reasoning_effort="high",
    )


def test_extract_chatgpt_account_id_from_nested_oauth_claim():
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-nested"}})
    assert extract_chatgpt_account_id(token) == "acct-nested"


def test_extract_chatgpt_account_id_from_direct_claim():
    token = _jwt({"chatgpt_account_id": "acct-direct"})
    assert extract_chatgpt_account_id(token) == "acct-direct"


def test_codex_payload_keeps_gladiator_history_and_raw_reasoning_items():
    model = _model()
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "opaque",
        "summary": [],
    }
    function_item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "bash",
        "arguments": '{"command":"pwd"}',
    }
    messages = [
        {"role": "system", "content": "Gladiator system"},
        {"role": "user", "content": "Inspect the repo"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
            "extra": {"codex_response_items": [reasoning_item, function_item]},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "/workspace"},
        {"role": "user", "content": "Continue"},
    ]

    payload = model._build_responses_payload(messages)

    assert payload["instructions"] == "Gladiator system"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert payload["input"][0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Inspect the repo"}],
    }
    assert reasoning_item in payload["input"]
    assert function_item in payload["input"]
    assert {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "/workspace",
    } in payload["input"]
    assert payload["input"][-1]["content"][0]["text"] == "Continue"
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "bash"
    assert "function" not in payload["tools"][0]


def test_codex_response_function_call_returns_gladiator_bash_action():
    model = _model()
    raw_items = [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"command":"git status --short"}',
        },
    ]

    message = model._assemble_codex_result([], [], raw_items, {"input_tokens": 100, "output_tokens": 10})

    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["extra"]["actions"] == [
        {"command": "git status --short", "tool_call_id": "call_1"}
    ]
    assert message["extra"]["codex_response_items"] == raw_items


def test_codex_direct_final_preserves_raw_response_items_for_next_turn():
    model = _model()
    raw_items = [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done"}],
        },
    ]

    with pytest.raises(Submitted) as raised:
        model._assemble_codex_result(["Done"], ["reasoning"], raw_items, {"input_tokens": 10})

    assistant = raised.value.messages[0]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Done"
    assert assistant["extra"]["codex_response_items"] == raw_items


def test_codex_payload_converts_gladiator_images_to_responses_input(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"not-a-real-png-but-data-url-encoding-is-enough")
    model = _model()
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look"},
                {"type": "gladiator_image_path", "path": str(image)},
            ],
        },
    ]

    payload = model._build_responses_payload(messages)
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Look"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
