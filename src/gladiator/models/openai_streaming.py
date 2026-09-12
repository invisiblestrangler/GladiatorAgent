from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from threading import Event, Timer
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
        retry_after_seconds: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        self.provider_family = provider_family
        self.retry_after_seconds = retry_after_seconds
        parts = [f"Provider HTTP {status_code}: {detail}"]
        metadata: list[str] = []
        if provider_family:
            metadata.append(f"provider={provider_family}")
        if request_id:
            metadata.append(f"request_id={request_id}")
        if metadata:
            parts.append("(" + ", ".join(metadata) + ")")
        super().__init__(" ".join(parts))


class ProviderTransportError(RuntimeError):
    """Network/stream transport failure with enough state to decide whether replay is safe."""

    def __init__(self, detail: str, *, stream_started: bool) -> None:
        self.stream_started = stream_started
        super().__init__(detail)


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
        max_retries: int = 2,
        retry_base_seconds: float = 1.5,
        slow_stream_notice_seconds: float = 12.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.event_sink = event_sink
        self.timeout_seconds = timeout_seconds
        self.cancel_event = cancel_event
        self.max_retries = max(0, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.slow_stream_notice_seconds = max(0.0, float(slow_stream_notice_seconds))
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

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._raise_if_cancelled()
            try:
                result = self._query_once(payload, attempt=attempt)
            except ProviderRequestError as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._is_retryable_provider_error(exc):
                    raise
                delay = self._retry_delay(attempt, exc.retry_after_seconds)
                self.event_sink(
                    AgentEvent(
                        EventKind.STATUS,
                        f"↻ Provider rejected the request; retrying {attempt + 1}/{self.max_retries} in {delay:.1f}s…",
                        {"retry": attempt + 1, "max_retries": self.max_retries, "status_code": exc.status_code},
                    )
                )
                self._sleep_with_cancel(delay)
                continue
            except ProviderTransportError as exc:
                last_error = exc
                if exc.stream_started or attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt, None)
                self.event_sink(
                    AgentEvent(
                        EventKind.STATUS,
                        f"↻ Provider connection failed; retrying {attempt + 1}/{self.max_retries} in {delay:.1f}s…",
                        {"retry": attempt + 1, "max_retries": self.max_retries},
                    )
                )
                self._sleep_with_cancel(delay)
                continue

            if attempt:
                self.event_sink(
                    AgentEvent(
                        EventKind.STATUS,
                        f"✓ Provider retry succeeded on attempt {attempt + 1}",
                        {"retry_succeeded": True, "attempt": attempt + 1},
                    )
                )
            return self._assemble_result(messages, *result)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Provider request failed without an error")

    def _query_once(
        self,
        payload: dict[str, Any],
        *,
        attempt: int,
    ) -> tuple[list[str], list[str], dict[int, dict[str, Any]], str | None, dict[str, Any]]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        stream_started = False
        request_started = time.monotonic()
        slow_notice_sent = Event()

        def slow_notice() -> None:
            if stream_started or self.cancel_event is not None and self.cancel_event.is_set():
                return
            slow_notice_sent.set()
            elapsed = max(1, int(time.monotonic() - request_started))
            self.event_sink(
                AgentEvent(
                    EventKind.STATUS,
                    f"⏳ Still waiting for the model to start streaming… {elapsed}s",
                    {"slow_stream": True, "ephemeral": True, "attempt": attempt + 1, "elapsed_seconds": elapsed},
                )
            )

        timer: Timer | None = None
        if self.slow_stream_notice_seconds > 0:
            timer = Timer(self.slow_stream_notice_seconds, slow_notice)
            timer.daemon = True
            timer.start()

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(30.0, self.timeout_seconds))
        try:
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        self._raise_provider_http_error(response)
                        for line in response.iter_lines():
                            self._raise_if_cancelled()
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            if not stream_started:
                                stream_started = True
                                if timer is not None:
                                    timer.cancel()
                                if slow_notice_sent.is_set():
                                    elapsed = max(1, int(time.monotonic() - request_started))
                                    self.event_sink(
                                        AgentEvent(
                                            EventKind.STATUS,
                                            f"✓ Model started streaming after {elapsed}s",
                                            {"slow_stream_recovered": True, "replace_slow_stream": True},
                                        )
                                    )
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                self.event_sink(
                                    AgentEvent(EventKind.WARNING, "Provider emitted an invalid SSE JSON chunk.")
                                )
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
            except ProviderRequestError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                suffix = (
                    " Stream output had already begun, so Gladiator will not replay the request automatically."
                    if stream_started
                    else ""
                )
                raise ProviderTransportError(
                    f"Provider transport error: {type(exc).__name__}: {exc}.{suffix}",
                    stream_started=stream_started,
                ) from exc
        finally:
            if timer is not None:
                timer.cancel()

        return content_parts, reasoning_parts, tool_calls, finish_reason, usage

    def _assemble_result(
        self,
        messages: list[dict],
        content_parts: list[str],
        reasoning_parts: list[str],
        tool_calls: dict[int, dict[str, Any]],
        finish_reason: str | None,
        usage: dict[str, Any],
    ) -> dict:
        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts)
        assembled_calls = [tool_calls[index] for index in sorted(tool_calls)]
        self.last_usage = usage
        self.event_sink(AgentEvent(EventKind.RESPONSE_FINISHED, data={"usage": usage, "finish_reason": finish_reason}))

        if not assembled_calls:
            if content:
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "extra": {
                        "cost": 0.0,
                        "usage": usage,
                        "reasoning": reasoning,
                        "timestamp": time.time(),
                        "finish_reason": finish_reason,
                    },
                }
                exit_message = {
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
                raise Submitted(assistant_message, exit_message)
            self._raise_format_error(
                "Provider returned neither text nor tool calls.",
                finish_reason=finish_reason,
            )

        actions = self._parse_actions(assembled_calls, finish_reason=finish_reason)
        return {
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
        retry_after_seconds: float | None = None
        raw_retry_after = headers.get("retry-after")
        if raw_retry_after is not None:
            try:
                retry_after_seconds = max(0.0, float(raw_retry_after))
            except (TypeError, ValueError):
                retry_after_seconds = None
        raise ProviderRequestError(
            status_code=status_code,
            detail=cls._provider_error_detail(body),
            request_id=request_id,
            provider_family=provider_family,
            retry_after_seconds=retry_after_seconds,
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

    @staticmethod
    def _is_retryable_provider_error(error: ProviderRequestError) -> bool:
        if error.status_code in {408, 409, 425, 429} or 500 <= error.status_code <= 599:
            return True
        if error.status_code != 400:
            return False
        detail = error.detail.lower()
        transient_markers = (
            "previous_response_id",
            "temporarily",
            "try again",
            "upstream",
            "overloaded",
            "model provider",
        )
        return any(marker in detail for marker in transient_markers)

    def _retry_delay(self, attempt: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return min(retry_after_seconds, 30.0)
        return min(self.retry_base_seconds * (2**attempt), 8.0)

    def _sleep_with_cancel(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

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
        raise FormatError(
            {"role": "user", "content": rendered, "extra": {"interrupt_type": "FormatError", "cost": 0.0}}
        )

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
