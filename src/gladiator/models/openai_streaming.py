from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from threading import Event
from typing import Any

import httpx
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError, UserInterruption
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, format_toolcall_observation_messages

from gladiator.events import AgentEvent, EventKind, EventSink, null_event_sink


class OpenAICompatibleStreamingModel:
    """Provider-agnostic OpenAI-compatible chat-completions model with streaming hooks."""

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
            "{	 if output.exception_info %}<exception>{{output.exception_info}}</exception>\n%{ endif %}"
            "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
        )
        self.format_error_template = format_error_template or "Format error: {{ error }}"
        self.last_usage: dict[str, Any] = {}
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_cache_write_tokens = 0

    @property
    def last_prompt_cache_ratio(self) -> float | None:
        prompt = int(self.last_usage.get("prompt_tokens") or 0)
        details = self.last_usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        return (cached / prompt) if prompt > 0 else None

    @property
    def cumulative_prompt_cache_ratio(self) -> float | None:
        return (self.total_cached_tokens / self.total_prompt_tokens) if self.total_prompt_tokens > 0 else None

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Return only provider-visible state while preserving a byte-stable conversation prefix."""
        prepared: list[dict] = []
        for message in messages:
            if message.get("role") == "exit":
                continue
            clean = {key: value for key, value in message.items() if key != "extra"}
            clean["content"] = self._expand_local_images(clean.get("content"))
            prepared.append(clean)
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
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
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

        self.last_usage = usage
        self._record_usage(usage)
        assembled_calls = [tool_calls[index] for index in sorted(tool_calls)]
        actions = self._parse_actions(assembled_calls, finish_reason=finish_reason)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": assembled_calls,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "usage": usage,
                "reasoning": "".join(reasoning_parts),
                "timestamp": time.time(),
                "finish_reason": finish_reason,
            },
        }
        self.event_sink(AgentEvent(EventKind.RESPONSE_FINISHED, data={"usage": usage, "finish_reason": finish_reason}))
        return message

    def _record_usage(self, usage: dict[str, Any]) -> None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        write_tokens = int(details.get("cache_write_tokens") or 0) if isinstance(details, dict) else 0
        self.total_prompt_tokens += prompt_tokens
        self.total_cached_tokens += cached_tokens
        self.total_cache_write_tokens += write_tokens

    def reset_cache_metrics(self) -> None:
        self.last_usage = {}
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_cache_write_tokens = 0

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
                call["id"] = raw["id"]
            function = raw.get("function") or {}
            if function.get("name"):
                call["function"]["name"] += str(function["name"])
            if function.get("arguments"):
                call["function"]["arguments"] += str(function["arguments"])
        for index, call in target.items():
            if not call["id"]:
                call["id"] = f"gladiator_call_{index}"

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
