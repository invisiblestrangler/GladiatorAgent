from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from threading import Event
from typing import Any

import httpx
from jinja2 import StrictUndefined, Template

from gladiator.events import AgentEvent, EventKind, EventSink, null_event_sink
from minisweagent.exceptions import FormatError, Submitted, UserInterruption
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, format_toolcall_observation_messages


class ProviderRequestError(RuntimeError):
    """Sanitized provider HTTP failure suitable for Telegram and logs."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        request_id: str = "",
        provider_family: str = "",
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        self.provider_family = provider_family
        parts = [f"Provider HTTP {status_code}: {detail}"]
        metadata: list[str] = []
        if provider_family:
            metadata.append(f"provider={provider_family}")
        if request_id:
            metadata.append(f"request_id={request_id}")
        if metadata:
            parts.append("(" + ", ".join(metadata) + ")")
        super().__init__(" ".join(parts))


class ToolTranscriptError(RuntimeError):
    """Raised before a request when local tool-call history violates the OpenAI transcript contract."""


class OpenAICompatibleStreamingModel:
    """Small OpenAI-compatible chat-completions model with first-class streaming hooks.

    Streaming deltas are emitted to the UI event sink but only the final assembled
    assistant message is returned to mini-swe-agent and therefore enters context.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        reasoning_effort: str = "high",
        event_sink: EventSink = null_event_sink,
        timeout_seconds: float = 300.0,
        cancel_event: Event | None = None,
        observation_template: str | None = None,
        format_error_template: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.event_sink = event_sink
        self.timeout_seconds = timeout_seconds
        self.cancel_event = cancel_event
        self.observation_template = observation_template or (
            "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
            "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
        )
        self.format_error_template = format_error_template or "Format error: {{ error }}"
        self.last_usage: dict[str, Any] = {}

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared: list[dict] = []
        for message in messages:
            if message.get("role") == "exit":
                continue
            clean = {k: v for k, v in message.items() if k != "extra"}
            clean["content"] = self._expand_local_images(clean.get("content"))
            prepared.append(clean)
        self._validate_tool_transcript(prepared)
        return prepared

    def _expand_local_images(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content
        expanded: list[Any] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "gladiator_image_path":
                expanded.append(item)
                continue
            path = Path(str(item.get("path", ""))).expanduser()
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(raw).decode("ascii")
            expanded.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return expanded

    @staticmethod
    def _validate_tool_transcript(messages: list[dict]) -> None:
        pending: list[str] = []
        for index, message in enumerate(messages):
            role = str(message.get("role") or "")
            if pending:
                if role != "tool":
                    raise ToolTranscriptError(
                        f"Invalid local tool transcript before message {index}: expected tool result(s) for "
                        f"{', '.join(pending)}, got role={role!r}. Start /new if this session came from an older broken run."
                    )
                tool_call_id = str(message.get("tool_call_id") or "")
                if tool_call_id not in pending:
                    raise ToolTranscriptError(
                        f"Invalid local tool transcript at message {index}: unexpected tool_call_id={tool_call_id!r}; "
                        f"expected one of {pending}."
                    )
                pending.remove(tool_call_id)
                continue

            if role == "tool":
                raise ToolTranscriptError(
                    f"Invalid local tool transcript at message {index}: orphan tool result "
                    f"{message.get('tool_call_id')!r}. Start /new if this session came from an older broken run."
                )
            tool_calls = message.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
                ids = [str(call.get("id") or "") for call in tool_calls if isinstance(call, dict)]
                if len(ids) != len(tool_calls) or any(not call_id for call_id in ids):
                    raise ToolTranscriptError(f"Invalid local tool transcript at message {index}: missing tool call id.")
                if len(set(ids)) != len(ids):
                    raise ToolTranscriptError(f"Invalid local tool transcript at message {index}: duplicate tool call ids.")
                pending = ids

        if pending:
            raise ToolTranscriptError(
                "Invalid local tool transcript at end of history: missing tool result(s) for " + ", ".join(pending)
            )

    def query(self, messages: list[dict], **_kwargs) -> dict:
        self._raise_if_cancelled()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._prepare_messages_for_api(messages),
            "tools": [BASH_TOOL],
            "tool_choice": "auto",
            "stream": True,
        }
        if self.reasoning_effort != "off":
            payload["reasoning_effort"] = self.reasoning_effort

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(30.0, self.timeout_seconds))
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                self._raise_provider_http_error(response)
                for line in response.iter_lines():
                    self._raise_if_cancelled()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        self.event_sink(AgentEvent(EventKind.WARNING, "Provider emitted an invalid SSE JSON chunk."))
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                        self.event_sink(AgentEvent(EventKind.TEXT_DELTA, text))
                    reasoning = self._extract_reasoning_delta(delta)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        self.event_sink(AgentEvent(EventKind.REASONING_DELTA, reasoning))
                    self._merge_tool_call_deltas(tool_calls, delta.get("tool_calls") or [])

        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts)
        assembled_calls = [tool_calls[index] for index in sorted(tool_calls)]
        self.last_usage = usage
        self.event_sink(AgentEvent(EventKind.RESPONSE_FINISHED, data={"usage": usage, "finish_reason": finish_reason}))

        if not assembled_calls:
            if content and self._has_prior_tool_activity(messages):
                raise Submitted(
                    {
                        "role": "exit",
                        "content": content,
                        "extra": {
                            "exit_status": "Submitted",
                            "submission": content,
                            "cost": 0.0,
                            "usage": usage,
                            "reasoning": reasoning,
                            "timestamp": time.time(),
                            "finish_reason": finish_reason,
                        },
                    }
                )
            self._raise_format_error(
                "No tool calls found in the response. Working turns must call the bash tool; "
                "a text-only response is accepted only after tool work has occurred.",
                finish_reason=finish_reason,
            )

        actions = self._parse_actions(assembled_calls, finish_reason=finish_reason)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": assembled_calls,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "usage": usage,
                "reasoning": reasoning,
                "timestamp": time.time(),
                "finish_reason": finish_reason,
            },
        }
        return message

    @classmethod
    def _raise_provider_http_error(cls, response: Any) -> None:
        status_code = int(getattr(response, "status_code", 200) or 200)
        if status_code < 400:
            return
        body = ""
        try:
            raw = response.read()
            if isinstance(raw, bytes):
                body = raw.decode("utf-8", errors="replace")
            else:
                body = str(raw)
        except Exception:
            try:
                body = str(response.text)
            except Exception:
                body = ""
        headers = getattr(response, "headers", {}) or {}
        request_id = str(headers.get("x-request-id") or headers.get("request-id") or "")
        provider_family = str(
            headers.get("x-si-provider-family") or headers.get("x-si-served-by") or headers.get("x-provider") or ""
        )
        raise ProviderRequestError(
            status_code=status_code,
            detail=cls._provider_error_detail(body),
            request_id=request_id,
            provider_family=provider_family,
        )

    @staticmethod
    def _provider_error_detail(body: str) -> str:
        text = body.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                candidates: list[Any] = [parsed.get("message"), parsed.get("detail"), parsed.get("error")]
                error = parsed.get("error")
                if isinstance(error, dict):
                    candidates = [error.get("message"), error.get("detail"), error.get("code"), *candidates]
                for value in candidates:
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        break
            text = re.sub(r"\s+", " ", text)
            return text[:1200]
        return "request rejected without an error body"

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise UserInterruption(
                {
                    "role": "exit",
                    "content": "Cancelled by user",
                    "extra": {"exit_status": "Cancelled", "submission": "Cancelled by user."},
                }
            )

    @staticmethod
    def _has_prior_tool_activity(messages: list[dict]) -> bool:
        for message in messages:
            if message.get("tool_calls"):
                return True
            extra = message.get("extra")
            if isinstance(extra, dict) and extra.get("actions"):
                return True
        return False

    @staticmethod
    def _extract_reasoning_delta(delta: dict[str, Any]) -> str:
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _merge_tool_call_deltas(target: dict[int, dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
        for raw in deltas:
            index = int(raw.get("index", 0))
            call = target.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if raw.get("id"):
                call["id"] = str(raw["id"])
            function = raw.get("function") or {}
            incoming_name = str(function.get("name") or "")
            if incoming_name:
                current_name = str(call["function"].get("name") or "")
                if not current_name:
                    call["function"]["name"] = incoming_name
                elif incoming_name == current_name:
                    pass
                elif incoming_name.startswith(current_name):
                    call["function"]["name"] = incoming_name
                elif not current_name.endswith(incoming_name):
                    call["function"]["name"] += incoming_name
            if function.get("arguments"):
                call["function"]["arguments"] += str(function["arguments"])
        for call in target.values():
            if not call["id"]:
                # Some compatible streaming endpoints omit the tool-call id. Generate a
                # portable OpenAI-shaped id that is unique across turns and short enough
                # for strict providers instead of reusing a per-index constant.
                call["id"] = f"call_{uuid.uuid4().hex[:24]}"

    def _parse_actions(self, tool_calls: list[dict[str, Any]], *, finish_reason: str | None) -> list[dict]:
        if not tool_calls:
            self._raise_format_error(
                "No tool calls found in the response. Every working turn must call the bash tool.",
                finish_reason=finish_reason,
            )
        actions: list[dict] = []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            error = ""
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                error = f"Could not parse bash arguments: {exc}. "
            if function.get("name") != "bash":
                error += f"Unknown tool {function.get('name')!r}. "
            if not isinstance(args, dict) or "command" not in args:
                error += "Missing command argument."
            if error:
                self._raise_format_error(error.strip(), finish_reason=finish_reason)
            actions.append({"command": args["command"], "tool_call_id": tool_call["id"]})
        return actions

    def _raise_format_error(self, error: str, *, finish_reason: str | None) -> None:
        rendered = Template(self.format_error_template, undefined=StrictUndefined).render(
            error=error,
            finish_reason=finish_reason,
        )
        raise FormatError({"role": "user", "content": rendered, "extra": {"interrupt_type": "FormatError", "cost": 0.0}})

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **_kwargs) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "reasoning_effort": self.reasoning_effort,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "model": {
                        "model_name": self.model_name,
                        "base_url": self.base_url,
                        "reasoning_effort": self.reasoning_effort,
                    },
                }
            }
        }
