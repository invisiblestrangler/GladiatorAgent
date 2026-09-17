from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from gladiator.service_goal import GoalAwareGladiatorService
from gladiator.telegram.bot import IncomingTask


class ResilientGoalAwareGladiatorService(GoalAwareGladiatorService):
    """Goal-aware runtime with process-safe transport and ordinary-task recovery.

    Provider requests already have request-level retry logic. This class protects the
    Telegram control plane and persists enough state for an ordinary user turn to be
    resumed after an abrupt process death (OOM kill, host reboot, supervisor restart).
    Goal-start executions keep using their existing dedicated recovery path.
    """

    TELEGRAM_POLL_RETRY_BASE_SECONDS = 1.0
    TELEGRAM_POLL_RETRY_MAX_SECONDS = 30.0
    MAX_AUTOMATIC_MESSAGE_RECOVERIES = 2

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

    async def handle_task(self, chat_id: int, incoming: IncomingTask) -> None:
        """Persist normal Telegram turns so a SIGKILL/OOM cannot silently orphan them."""
        if self._inside_goal_supervisor():
            await super().handle_task(chat_id, incoming)
            return
        await self._run_tracked_message_task(chat_id, incoming, recovery_count=0, recovered=False)

    @staticmethod
    def _inside_goal_supervisor() -> bool:
        task = asyncio.current_task()
        if task is None:
            return False
        return task.get_name().startswith("gladiator-goal-")

    @property
    def _message_task_state_path(self) -> Path:
        return self.state_dir / "active-message-task.json"

    def _load_message_task_state(self) -> dict[str, Any] | None:
        path = self._message_task_state_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_message_task_state(
        self,
        *,
        status: str,
        chat_id: int,
        incoming: IncomingTask,
        recovery_count: int,
        detail: str = "",
        started_at: float | None = None,
    ) -> None:
        previous = self._load_message_task_state() or {}
        if started_at is None:
            started_at = time.time() if status == "running" else float(previous.get("started_at") or time.time())
        payload: dict[str, Any] = {
            "status": status,
            "chat_id": chat_id,
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.time(),
            "recovery_count": max(0, recovery_count),
            "task_text": incoming.text,
            "image_paths": [str(path) for path in incoming.image_paths],
            "file_paths": [str(path) for path in incoming.file_paths],
            "source_message_count": max(1, int(incoming.source_message_count)),
        }
        if detail:
            payload["detail"] = detail[:800]
        if status != "running":
            payload["finished_at"] = time.time()
        path = self._message_task_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _incoming_from_message_state(state: dict[str, Any]) -> IncomingTask | None:
        task_text = str(state.get("task_text") or "").strip()
        if not task_text:
            return None

        def existing_paths(key: str) -> list[Path]:
            values = state.get(key)
            if not isinstance(values, list):
                return []
            paths: list[Path] = []
            for value in values:
                path = Path(str(value)).expanduser()
                if path.exists():
                    paths.append(path)
            return paths

        try:
            source_message_count = max(1, int(state.get("source_message_count") or 1))
        except (TypeError, ValueError):
            source_message_count = 1
        return IncomingTask(
            text=task_text,
            image_paths=existing_paths("image_paths"),
            file_paths=existing_paths("file_paths"),
            source_message_count=source_message_count,
        )

    @staticmethod
    def _message_recovery_prompt(original_text: str) -> str:
        return (
            "Resume the user request below after an unexpected Gladiator process interruption. The interruption may "
            "have happened during a tool command, so first inspect the restored conversation trajectory, current "
            "workspace, TODO ledger, and relevant outputs to determine what actually completed. Do not blindly "
            "repeat commands or operations with side effects. Continue only the unfinished work, verify the current "
            "state, and finish the original request.\n\nOriginal user request:\n" + original_text
        )

    async def _run_tracked_message_task(
        self,
        chat_id: int,
        incoming: IncomingTask,
        *,
        recovery_count: int,
        recovered: bool,
    ) -> None:
        started_at = time.time()
        original = incoming
        self._write_message_task_state(
            status="running",
            chat_id=chat_id,
            incoming=original,
            recovery_count=recovery_count,
            detail="automatic process recovery" if recovered else "user message",
            started_at=started_at,
        )
        execution = incoming
        if recovered:
            execution = IncomingTask(
                text=self._message_recovery_prompt(original.text),
                image_paths=list(original.image_paths),
                file_paths=list(original.file_paths),
                source_message_count=original.source_message_count,
            )
        try:
            await super().handle_task(chat_id, execution)
        except asyncio.CancelledError:
            self._write_message_task_state(
                status="interrupted",
                chat_id=chat_id,
                incoming=original,
                recovery_count=recovery_count,
                detail="ordinary task coroutine was cancelled",
                started_at=started_at,
            )
            raise
        except Exception as exc:
            self._write_message_task_state(
                status="failed",
                chat_id=chat_id,
                incoming=original,
                recovery_count=recovery_count,
                detail=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
            )
            raise
        else:
            stopped = self.cancel_event.is_set()
            self._write_message_task_state(
                status="stopped" if stopped else "finished",
                chat_id=chat_id,
                incoming=original,
                recovery_count=recovery_count,
                detail="stopped by user" if stopped else "ordinary task turn returned normally",
                started_at=started_at,
            )

    def _spawn_supervised_message_task(
        self,
        chat_id: int,
        incoming: IncomingTask,
        *,
        recovery_count: int,
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(
            self._run_tracked_message_task(
                chat_id,
                incoming,
                recovery_count=recovery_count,
                recovered=True,
            ),
            name="gladiator-message-recovery",
        )
        self.bot._spawned.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self.bot._spawned.discard(completed)
            if completed.cancelled():
                return
            try:
                exc = completed.exception()
            except asyncio.CancelledError:
                return
            if exc is None:
                return
            self._record_runtime_failure("detached ordinary-task recovery escaped supervision", exc)
            asyncio.create_task(
                self._queue_or_send_failure(chat_id, f"Recovered task crashed unexpectedly: {exc}")
            )

        task.add_done_callback(done)
        return task

    async def _recover_interrupted_goal_execution(self) -> None:
        """Run the existing goal recovery, then recover an interrupted ordinary turn."""
        await super()._recover_interrupted_goal_execution()
        await self._recover_interrupted_message_execution()

    async def _recover_interrupted_message_execution(self) -> None:
        state = self._load_message_task_state()
        if state is None or state.get("status") not in {"running", "interrupted"}:
            return
        incoming = self._incoming_from_message_state(state)
        if incoming is None:
            return
        try:
            chat_id = int(state["chat_id"])
        except (KeyError, TypeError, ValueError):
            return
        try:
            previous_count = max(0, int(state.get("recovery_count") or 0))
        except (TypeError, ValueError):
            previous_count = 0
        try:
            started_at = float(state.get("started_at") or time.time())
        except (TypeError, ValueError):
            started_at = time.time()
        next_count = previous_count + 1
        self._write_message_task_state(
            status="interrupted",
            chat_id=chat_id,
            incoming=incoming,
            recovery_count=previous_count,
            detail="previous Gladiator process ended while this ordinary task was active",
            started_at=started_at,
        )

        if next_count > self.MAX_AUTOMATIC_MESSAGE_RECOVERIES:
            self._write_message_task_state(
                status="recovery_stopped",
                chat_id=chat_id,
                incoming=incoming,
                recovery_count=previous_count,
                detail="automatic recovery limit reached",
                started_at=started_at,
            )
            await self._try_send_html(
                chat_id,
                "<b>Automatic task recovery stopped.</b> This ordinary task was interrupted repeatedly. "
                "Its saved trajectory and workspace are still available; send a new message to continue manually.",
            )
            return

        await self._try_send_html(
            chat_id,
            "<b>Recovered an interrupted task.</b> The Gladiator process ended while your previous message was still "
            "running. I am checking existing work first, then continuing only what remains.",
        )
        self._spawn_supervised_message_task(chat_id, incoming, recovery_count=next_count)

    def _task_status_html(self) -> str:
        rendered = super()._task_status_html()
        state = self._load_message_task_state()
        if state is None:
            return rendered + "\nPersisted ordinary task: <i>none</i>"
        status = html.escape(str(state.get("status") or "unknown"))
        pid = html.escape(str(state.get("pid") or "unknown"))
        recovery_count = int(state.get("recovery_count") or 0)
        return (
            rendered
            + f"\nPersisted ordinary task: <b>{status}</b>"
            + f" · PID <code>{pid}</code> · recoveries {recovery_count}"
        )

    @staticmethod
    def _telegram_response_description(exc: Exception) -> str:
        if not isinstance(exc, httpx.HTTPStatusError):
            return ""
        try:
            payload = exc.response.json()
        except (ValueError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        for key in ("description", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return re.sub(r"\s+", " ", value).strip()[:800]
        return ""

    @staticmethod
    def _redact_telegram_credentials(text: str) -> str:
        return re.sub(
            r"(https://api\.telegram\.org/bot)[^/\s'\"<>]+",
            r"\1<redacted>",
            text,
            flags=re.IGNORECASE,
        )

    @classmethod
    def _is_cosmetic_telegram_failure(cls, label: str, exc: Exception) -> bool:
        if "telegram" not in label.lower():
            return False
        detail = cls._telegram_response_description(exc) or str(exc)
        return "message is not modified" in detail.lower()

    def _record_runtime_failure(self, label: str, exc: Exception) -> None:
        """Log actionable detail without leaking Telegram credentials or no-op edit noise."""
        if self._is_cosmetic_telegram_failure(label, exc):
            return
        if "telegram" in label.lower():
            if isinstance(exc, httpx.HTTPStatusError):
                status = int(exc.response.status_code)
                detail = self._telegram_response_description(exc) or "request failed"
                safe_exc = RuntimeError(f"Telegram HTTP {status}: {detail}")
            else:
                safe_exc = RuntimeError(self._redact_telegram_credentials(str(exc)))
            super()._record_runtime_failure(label, safe_exc)
            return
        super()._record_runtime_failure(label, exc)

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
        if self._is_cosmetic_telegram_failure(label, exc):
            return
        self._record_runtime_failure(label, exc)
        try:
            path = self.state_dir / "runtime-errors.log"
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            trace = self._redact_telegram_credentials(trace)
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
