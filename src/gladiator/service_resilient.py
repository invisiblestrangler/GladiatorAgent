from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

import httpx

from gladiator.service_goal import GoalAwareGladiatorService


class ResilientGoalAwareGladiatorService(GoalAwareGladiatorService):
    """Goal-aware runtime with process-safe Telegram transport boundaries.

    The model/provider path already contains request-level transport retry logic. This
    class protects the separate Telegram control plane so a transient Bot API/network
    failure cannot unwind the entire Gladiator process and take active work with it.
    """

    TELEGRAM_POLL_RETRY_BASE_SECONDS = 1.0
    TELEGRAM_POLL_RETRY_MAX_SECONDS = 30.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_telegram_runtime_guards()

    def _install_telegram_runtime_guards(self) -> None:
        original_get_updates = self.bot.client.get_updates
        original_handle_update = self.bot._handle_update

        async def resilient_get_updates(*, offset: int | None = None, timeout: int = 30):
            failures = 0
            while True:
                try:
                    updates = await original_get_updates(offset=offset, timeout=timeout)
                except asyncio.CancelledError:
                    raise
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    failures += 1
                    delay = self._telegram_poll_retry_delay(exc, failures)
                    if failures == 1 or failures % 10 == 0:
                        self._record_runtime_boundary_failure(
                            f"Telegram polling failed; keeping Gladiator alive and retrying in {delay:.1f}s",
                            exc,
                        )
                    await asyncio.sleep(delay)
                    continue
                if failures:
                    self._record_runtime_recovery(
                        f"Telegram polling recovered after {failures} failure(s); active task was kept alive"
                    )
                return updates

        async def guarded_handle_update(update: dict[str, Any]) -> None:
            try:
                await original_handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_runtime_boundary_failure(
                    "Telegram update handler failed; isolated the update instead of terminating Gladiator",
                    exc,
                )
                chat_id = self._telegram_update_chat_id(update)
                if chat_id is not None and self.bot._authorized(chat_id):
                    await self._try_send_html(
                        chat_id,
                        "<b>Telegram request failed, but Gladiator stayed running.</b> "
                        "The failure was isolated to this update; retry the command/message if needed.",
                    )

        self.bot.client.get_updates = resilient_get_updates  # type: ignore[method-assign]
        self.bot._handle_update = guarded_handle_update  # type: ignore[method-assign]

    def _telegram_poll_retry_delay(self, exc: Exception, failures: int) -> float:
        retry_after: float | None = None
        if isinstance(exc, httpx.HTTPStatusError):
            raw = exc.response.headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        if retry_after is not None and retry_after >= 0:
            return min(self.TELEGRAM_POLL_RETRY_MAX_SECONDS, retry_after)
        exponent = min(max(0, failures - 1), 8)
        return min(
            self.TELEGRAM_POLL_RETRY_MAX_SECONDS,
            self.TELEGRAM_POLL_RETRY_BASE_SECONDS * (2**exponent),
        )

    @staticmethod
    def _telegram_update_chat_id(update: dict[str, Any]) -> int | None:
        message = update.get("message")
        if not isinstance(message, dict):
            callback = update.get("callback_query")
            if isinstance(callback, dict):
                message = callback.get("message")
        if not isinstance(message, dict):
            stopped = update.get("stopped_message_generation")
            if isinstance(stopped, dict):
                chat = stopped.get("chat")
                if isinstance(chat, dict) and isinstance(chat.get("id"), int):
                    return int(chat["id"])
            return None
        chat = message.get("chat")
        if not isinstance(chat, dict) or not isinstance(chat.get("id"), int):
            return None
        return int(chat["id"])

    def _record_runtime_boundary_failure(self, label: str, exc: Exception) -> None:
        self._record_runtime_failure(label, exc)
        try:
            path = self.state_dir / "runtime-errors.log"
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(trace.rstrip() + "\n")
        except OSError:
            pass

    def _record_runtime_recovery(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        try:
            path = self.state_dir / "runtime-errors.log"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
