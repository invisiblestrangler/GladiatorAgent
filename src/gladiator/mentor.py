from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Iterable

import httpx

from gladiator.config import MentorConfig, ProviderConfig
from gladiator.models.openai_streaming import ProviderRequestError, ProviderTransportError

MENTOR_SYSTEM_PROMPT = """You are Gladiator's senior technical mentor. You are advisory only, not an execution agent.
Analyze only the question and evidence supplied in this request. Do not run tools, edit files, perform grunt work, or pretend to have executed tests. Your job is to provide high-value advice when the main agent is genuinely stuck on a hard problem or wants a final review of an unusually complex algorithm/design.
Focus on root cause, correctness, algorithmic complexity, performance, edge cases, architectural tradeoffs, and the smallest useful next investigation. If the supplied evidence is insufficient, say exactly which minimal additional file/log/measurement would resolve the uncertainty.
Keep the response compact and actionable for the main agent. Do not write user-facing status prose and do not restate large supplied inputs."""


@dataclass(frozen=True, slots=True)
class MentorRequest:
    question: str
    files: tuple[Path, ...] = ()
    logs: tuple[Path, ...] = ()


class MentorClient:
    """One-shot, tool-free mentor transport using only just-in-time evidence.

    This deliberately has no agent loop and no persistent mentor conversation. The
    main Gladiator agent remains the sole executor; the mentor returns advice only.
    """

    def __init__(
        self,
        *,
        provider: ProviderConfig,
        mentor: MentorConfig,
        workspace_root: Path,
        cancel_event: Event | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.mentor = mentor
        self.workspace_root = workspace_root.resolve()
        self.cancel_event = cancel_event
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))

    @property
    def model_name(self) -> str:
        return self.mentor.model or self.provider.model

    def consult(self, request: MentorRequest) -> str:
        if not self.mentor.enabled:
            raise RuntimeError("Mentor mode is disabled. Enable it with /mentor on.")
        question = request.question.strip()
        if not question:
            raise ValueError("Mentor question must not be empty")
        self._raise_if_cancelled()
        context = self._load_context(request)
        prompt = self._user_prompt(question, context)
        if self.provider.mode == "codex_oauth":
            advice = self._consult_codex(prompt)
        else:
            advice = self._consult_openai_compatible(prompt)
        advice = advice.strip()
        if not advice:
            raise RuntimeError("Mentor returned an empty response")
        if len(advice) > self.mentor.max_advice_chars:
            advice = advice[: self.mentor.max_advice_chars].rstrip() + "\n[mentor advice truncated by Gladiator]"
        return advice

    def _load_context(self, request: MentorRequest) -> str:
        entries: list[tuple[str, Path]] = [*(('FILE', path) for path in request.files), *(('LOG', path) for path in request.logs)]
        if len(entries) > self.mentor.max_files:
            entries = entries[: self.mentor.max_files]
        chunks: list[str] = []
        remaining = self.mentor.max_total_context_chars
        for kind, raw_path in entries:
            if remaining <= 0:
                break
            path = raw_path.expanduser().resolve()
            try:
                relative = path.relative_to(self.workspace_root)
            except ValueError as exc:
                raise ValueError(f"Mentor context path is outside workspace: {path}") from exc
            if not path.is_file():
                raise ValueError(f"Mentor context file does not exist: {path}")
            per_file = min(self.mentor.max_file_chars, remaining)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read(per_file + 1)
            except OSError as exc:
                raise ValueError(f"Could not read mentor context file {relative}: {exc}") from exc
            truncated = len(text) > per_file
            text = text[:per_file]
            if truncated:
                text += "\n[truncated]"
            block = f"--- {kind}: {relative} ---\n{text}"
            if len(block) > remaining:
                block = block[:remaining]
            chunks.append(block)
            remaining -= len(block)
        return "\n\n".join(chunks)

    @staticmethod
    def _user_prompt(question: str, context: str) -> str:
        if not context:
            return f"Main-agent question:\n{question}"
        return f"Main-agent question:\n{question}\n\nJust-in-time evidence:\n{context}"

    def _consult_openai_compatible(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": MENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        if self.mentor.reasoning_effort != "off":
            payload["reasoning_effort"] = self.mentor.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.provider.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        url = f"{self.provider.base_url.rstrip('/')}/chat/completions"

        def parse(response: httpx.Response) -> str:
            parts: list[str] = []
            for data in self._sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
            return "".join(parts)

        return self._stream_with_retries(url, headers, payload, parse)

    def _consult_codex(self, prompt: str) -> str:
        access_token = self.provider.codex_access_token.get_secret_value()
        account_id = self.provider.codex_account_id
        if not access_token or not account_id:
            raise RuntimeError("Codex OAuth mentor requires a connected access token and ChatGPT account id.")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": MENTOR_SYSTEM_PROMPT,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "store": False,
            "stream": True,
        }
        if self.mentor.reasoning_effort != "off":
            payload["reasoning"] = {"effort": self.mentor.reasoning_effort, "summary": "auto"}
        headers = {
            "Authorization": f"Bearer {access_token}",
            "ChatGPT-Account-ID": account_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        def parse(response: httpx.Response) -> str:
            parts: list[str] = []
            fallback_parts: list[str] = []
            for data in self._sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                kind = str(event.get("type") or "")
                if kind == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        parts.append(delta)
                elif kind == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "message":
                        for content in item.get("content") or []:
                            if not isinstance(content, dict):
                                continue
                            text = content.get("text")
                            if content.get("type") in {"output_text", "text"} and isinstance(text, str):
                                fallback_parts.append(text)
                elif kind in {"response.failed", "response.incomplete"}:
                    raise ProviderTransportError(
                        f"Mentor response failed after streaming began: {kind}",
                        stream_started=True,
                    )
            return "".join(parts).strip() or "".join(fallback_parts).strip()

        return self._stream_with_retries(self.provider.codex_responses_url, headers, payload, parse)

    def _stream_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        parser,
    ) -> str:
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(30.0, self.timeout_seconds))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._raise_if_cancelled()
            stream_started = False
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code >= 400:
                            raw = response.read()
                            detail = raw.decode("utf-8", errors="replace")[:1200] if isinstance(raw, bytes) else str(raw)[:1200]
                            retry_after = response.headers.get("retry-after")
                            retry_after_seconds: float | None = None
                            if retry_after is not None:
                                try:
                                    retry_after_seconds = max(0.0, float(retry_after))
                                except ValueError:
                                    retry_after_seconds = None
                            raise ProviderRequestError(
                                status_code=response.status_code,
                                detail=detail or "mentor request rejected",
                                request_id=str(response.headers.get("x-request-id") or ""),
                                retry_after_seconds=retry_after_seconds,
                            )
                        stream_started = True
                        return parser(response)
            except ProviderRequestError as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._retryable_status(exc.status_code):
                    raise
                delay = min(exc.retry_after_seconds if exc.retry_after_seconds is not None else 1.5 * (2**attempt), 8.0)
                self._sleep_with_cancel(delay)
            except ProviderTransportError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if stream_started or attempt >= self.max_retries:
                    raise ProviderTransportError(
                        f"Mentor transport error: {type(exc).__name__}: {exc}",
                        stream_started=stream_started,
                    ) from exc
                self._sleep_with_cancel(min(1.5 * (2**attempt), 8.0))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Mentor request failed without an error")

    def _sse_data(self, response: httpx.Response) -> Iterable[str]:
        for line in response.iter_lines():
            self._raise_if_cancelled()
            if not line or not line.startswith("data:"):
                continue
            yield line[5:].strip()

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code in {408, 409, 425, 429} or 500 <= status_code <= 599

    def _sleep_with_cancel(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            time.sleep(min(0.1, deadline - time.monotonic()))

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("Mentor consultation cancelled by user")
