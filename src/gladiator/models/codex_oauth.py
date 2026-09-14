from __future__ import annotations

import base64
import json
import time
import uuid
from threading import Event, Timer
from typing import Any

import httpx
from minisweagent.exceptions import Submitted
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from gladiator.events import AgentEvent, EventKind, EventSink, null_event_sink

from .openai_streaming import (
    OpenAICompatibleStreamingModel,
    ProviderRequestError,
    ProviderTransportError,
)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return claims if isinstance(claims, dict) else None


def extract_chatgpt_account_id(access_token: str) -> str | None:
    """Extract the ChatGPT account/workspace id carried by a Codex OAuth JWT."""
    claims = _jwt_claims(access_token)
    if claims is None:
        return None

    direct = claims.get("chatgpt_account_id")
    if isinstance(direct, str) and direct:
        return direct

    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        nested = auth.get("chatgpt_account_id")
        if isinstance(nested, str) and nested:
            return nested

    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first = organizations[0]
        if isinstance(first, dict):
            organization_id = first.get("id")
            if isinstance(organization_id, str) and organization_id:
                return organization_id
    return None


def oauth_token_expiration(access_token: str) -> int | None:
    claims = _jwt_claims(access_token)
    if claims is None:
        return None
    value = claims.get("exp")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


class CodexOAuthStreamingModel(OpenAICompatibleStreamingModel):
    """Direct Codex subscription transport while retaining Gladiator's own harness.

    Only the model transport changes. Tool calls are translated back into the same
    mini-swe/Gladiator bash transcript and are executed by GladiatorLocalEnvironment.
    """

    def __init__(
        self,
        *,
        access_token: str,
        account_id: str,
        model_name: str,
        responses_url: str = CODEX_RESPONSES_URL,
        reasoning_effort: str = "high",
        event_sink: EventSink = null_event_sink,
        timeout_seconds: float = 300.0,
        cancel_event: Event | None = None,
        max_retries: int = 2,
        retry_base_seconds: float = 1.5,
        slow_stream_notice_seconds: float = 12.0,
    ) -> None:
        if not access_token.strip():
            raise ValueError("Codex OAuth access token is empty")
        if not account_id.strip():
            raise ValueError("ChatGPT account id is required for Codex OAuth")
        super().__init__(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=access_token,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            event_sink=event_sink,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            slow_stream_notice_seconds=slow_stream_notice_seconds,
        )
        self.responses_url = responses_url
        self.account_id = account_id

    @property
    def access_token(self) -> str:
        return self.api_key

    @access_token.setter
    def access_token(self, value: str) -> None:
        self.api_key = value

    def query(self, messages: list[dict], **_kwargs) -> dict:
        self._raise_if_cancelled()
        payload = self._build_responses_payload(messages)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._raise_if_cancelled()
            try:
                result = self._query_codex_once(payload, attempt=attempt)
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
            return self._assemble_codex_result(*result)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Codex OAuth request failed without an error")

    def _prepare_codex_history(self, messages: list[dict]) -> list[dict]:
        prepared: list[dict] = []
        validation: list[dict] = []
        for message in messages:
            if message.get("role") == "exit":
                continue
            clean = dict(message)
            clean["content"] = self._expand_local_images(clean.get("content"))
            prepared.append(clean)
            validation.append({key: value for key, value in clean.items() if key != "extra"})
        self._validate_tool_transcript(validation)
        return prepared

    def _build_responses_payload(self, messages: list[dict]) -> dict[str, Any]:
        history = self._prepare_codex_history(messages)
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in history:
            role = str(message.get("role") or "")
            content = message.get("content")
            if role == "system":
                text = self._content_text(content)
                if text:
                    instructions.append(text)
                continue

            if role == "assistant":
                extra = message.get("extra")
                raw_items = extra.get("codex_response_items") if isinstance(extra, dict) else None
                if isinstance(raw_items, list) and raw_items:
                    input_items.extend(dict(item) for item in raw_items if isinstance(item, dict))
                    continue
                assistant_content = self._responses_content(content, assistant=True)
                if assistant_content:
                    input_items.append({"type": "message", "role": "assistant", "content": assistant_content})
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") or {}
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "arguments": str(function.get("arguments") or "{}"),
                        }
                    )
                continue

            if role == "tool":
                output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": output,
                    }
                )
                continue

            if role == "user":
                user_content = self._responses_content(content, assistant=False)
                if user_content:
                    input_items.append({"type": "message", "role": "user", "content": user_content})

        function = BASH_TOOL.get("function") or {}
        bash_tool = {
            "type": "function",
            "name": str(function.get("name") or "bash"),
            "description": str(function.get("description") or "Execute a shell command."),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": "\n\n".join(instructions),
            "input": input_items,
            "tools": [bash_tool],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if self.reasoning_effort != "off":
            payload["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        return payload

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _responses_content(content: Any, *, assistant: bool) -> list[dict[str, Any]]:
        text_type = "output_text" if assistant else "input_text"
        if isinstance(content, str):
            return [{"type": text_type, "text": content}] if content else []
        if not isinstance(content, list):
            return []

        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or "")
            if kind in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str) and text:
                    converted.append({"type": text_type, "text": text})
                continue
            if not assistant and kind == "image_url":
                image = item.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                if isinstance(url, str) and url:
                    converted.append({"type": "input_image", "image_url": url})
        return converted

    def _query_codex_once(
        self,
        payload: dict[str, Any],
        *,
        attempt: int,
    ) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        response_items: list[dict[str, Any]] = []
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

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "ChatGPT-Account-ID": self.account_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(30.0, self.timeout_seconds))
        try:
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", self.responses_url, headers=headers, json=payload) as response:
                        try:
                            self._raise_provider_http_error(response)
                        except ProviderRequestError as exc:
                            if exc.status_code in {401, 403}:
                                raise ProviderRequestError(
                                    status_code=exc.status_code,
                                    detail="Codex subscription OAuth token was rejected or expired. Reconnect with /codex connect <token>.",
                                    request_id=exc.request_id,
                                    retry_after_seconds=exc.retry_after_seconds,
                                ) from exc
                            raise

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
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                self.event_sink(AgentEvent(EventKind.WARNING, "Provider emitted an invalid SSE JSON chunk."))
                                continue
                            if not isinstance(event, dict):
                                continue
                            kind = str(event.get("type") or "")
                            if kind == "response.output_text.delta":
                                delta = event.get("delta")
                                if isinstance(delta, str) and delta:
                                    content_parts.append(delta)
                                    self.event_sink(AgentEvent(EventKind.TEXT_DELTA, delta))
                            elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                                delta = event.get("delta")
                                if isinstance(delta, str) and delta:
                                    reasoning_parts.append(delta)
                                    self.event_sink(AgentEvent(EventKind.REASONING_DELTA, delta))
                            elif kind == "response.output_item.done":
                                item = event.get("item")
                                if isinstance(item, dict):
                                    saved = dict(item)
                                    if saved.get("type") == "function_call" and not saved.get("call_id"):
                                        saved["call_id"] = str(saved.get("id") or f"call_{uuid.uuid4().hex[:24]}")
                                    response_items.append(saved)
                            elif kind == "response.completed":
                                response_data = event.get("response")
                                if isinstance(response_data, dict) and isinstance(response_data.get("usage"), dict):
                                    usage = dict(response_data["usage"])
                            elif kind in {"response.failed", "response.incomplete"}:
                                detail = self._responses_stream_error(event)
                                raise ProviderTransportError(
                                    f"Codex response failed after streaming began: {detail}",
                                    stream_started=True,
                                )
            except ProviderRequestError:
                raise
            except ProviderTransportError:
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

        return content_parts, reasoning_parts, response_items, usage

    @staticmethod
    def _responses_stream_error(event: dict[str, Any]) -> str:
        response = event.get("response")
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                for key in ("message", "code", "type"):
                    value = error.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()[:1200]
            incomplete = response.get("incomplete_details")
            if isinstance(incomplete, dict):
                reason = incomplete.get("reason")
                if isinstance(reason, str) and reason:
                    return reason[:1200]
        return str(event.get("type") or "unknown response failure")

    @staticmethod
    def _text_from_response_items(items: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in items:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    parts.append(str(content["text"]))
        return "".join(parts).strip()

    def _assemble_codex_result(
        self,
        content_parts: list[str],
        reasoning_parts: list[str],
        response_items: list[dict[str, Any]],
        usage: dict[str, Any],
    ) -> dict:
        content = "".join(content_parts).strip() or self._text_from_response_items(response_items)
        reasoning = "".join(reasoning_parts)
        tool_calls: list[dict[str, Any]] = []
        for item in response_items:
            if item.get("type") != "function_call":
                continue
            call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )

        finish_reason = "tool_calls" if tool_calls else "stop"
        self.last_usage = usage
        self.event_sink(AgentEvent(EventKind.RESPONSE_FINISHED, data={"usage": usage, "finish_reason": finish_reason}))
        protocol_extra = {"codex_response_items": response_items}

        if not tool_calls:
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
                        **protocol_extra,
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
            self._raise_format_error("Provider returned neither text nor tool calls.", finish_reason=finish_reason)

        actions = self._parse_actions(tool_calls, finish_reason=finish_reason)
        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "usage": usage,
                "reasoning": reasoning,
                "timestamp": time.time(),
                "finish_reason": finish_reason,
                **protocol_extra,
            },
        }

    def get_template_vars(self, **_kwargs) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.responses_url,
            "reasoning_effort": self.reasoning_effort,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "model": {
                        "model_name": self.model_name,
                        "transport": "codex_oauth",
                        "reasoning_effort": self.reasoning_effort,
                    },
                }
            }
        }
