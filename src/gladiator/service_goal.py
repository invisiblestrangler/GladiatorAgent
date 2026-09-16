from __future__ import annotations

import asyncio
import html
import json
import os
import time
from typing import Any

from gladiator.config import REASONING_EFFORTS, save_config
from gladiator.mentor import MentorClient, MentorRequest
from gladiator.model_context import (
    apply_model_context,
    clear_model_context,
    discover_model_context,
    format_model_context,
)
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.telegram.bot import IncomingTask, TELEGRAM_COMMANDS


class GoalAwareGladiatorService(ExtendedGladiatorService):
    """Extended service with immediate goals, mentor advice, and model metadata discovery."""

    MAX_AUTOMATIC_GOAL_RECOVERIES = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment.mentor_handler = self._consult_mentor_from_agent_thread

    @staticmethod
    def _install_native_runtime_commands() -> None:
        ExtendedGladiatorService._install_native_runtime_commands()
        if not any(name == "task" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(4, ("task", "Show live task execution state"))
        if not any(name == "mentor" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(5, ("mentor", "Configure rare advisory mentor"))

    async def run_forever(self) -> None:
        provider = self.config.provider
        if provider.context_window is None or provider.max_context_window is None:
            await self._refresh_model_context(clear_first=False)
        self._loop = asyncio.get_running_loop()
        await self._recover_interrupted_goal_execution()
        await super().run_forever()

    def _install_bot_extensions(self) -> None:
        super()._install_bot_extensions()
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            stripped = text.strip()
            lowered = stripped.lower()
            if lowered == "/task":
                await self.bot.client.send_message(chat_id, self._task_status_html())
                return True

            if lowered.startswith("/mentor"):
                _, _, argument = stripped.partition(" ")
                await self._handle_mentor_command(chat_id, argument.strip())
                return True

            if lowered.startswith("/model"):
                _, _, argument = stripped.partition(" ")
                await self._handle_model_command(chat_id, argument.strip())
                return True

            if lowered != "/goal start":
                before = self._provider_identity()
                handled = await original(chat_id, message, text)
                if handled and lowered in {"/start", "/help"}:
                    await self.bot.client.send_message(
                        chat_id,
                        "/task — show whether Gladiator is actually executing a task\n"
                        "/mentor [on|off|model ID|reasoning LEVEL] — configure the rare read-only advisory mentor\n"
                        "/model refresh — re-read model context limits from the active provider API",
                    )
                if handled and self._provider_identity() != before:
                    await self._refresh_model_context(clear_first=True)
                    await self.bot.client.send_message(
                        chat_id,
                        "<b>Model metadata</b>\n<code>" + html.escape(format_model_context(self.config.provider)) + "</code>",
                    )
                return handled

            goal = self.goal_manager.load()
            if goal is None or goal.status != "active":
                await self.bot.client.send_message(
                    chat_id,
                    "No active session goal. Set one first with <code>/goal set ...</code>.",
                )
                return True

            await self.bot.client.send_message(chat_id, "Starting the active session goal now.")
            self._spawn_supervised_goal_task(chat_id, recovery_count=0, recovered=False)
            return True

        self.bot._handle_command = command_handler  # type: ignore[method-assign]

    @property
    def _goal_task_state_path(self):
        return self.state_dir / "active-task.json"

    def _load_goal_task_state(self) -> dict[str, Any] | None:
        path = self._goal_task_state_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_goal_task_state(
        self,
        *,
        status: str,
        chat_id: int,
        recovery_count: int,
        detail: str = "",
        started_at: float | None = None,
    ) -> None:
        path = self._goal_task_state_path
        previous = self._load_goal_task_state() or {}
        if started_at is None:
            started_at = time.time() if status == "running" else float(previous.get("started_at") or time.time())
        payload: dict[str, Any] = {
            "status": status,
            "chat_id": chat_id,
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": time.time(),
            "recovery_count": max(0, recovery_count),
        }
        if detail:
            payload["detail"] = detail[:800]
        if status != "running":
            payload["finished_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _goal_execution_text(*, recovered: bool) -> str:
        if recovered:
            return (
                "Resume the active session goal after an interrupted Gladiator runtime. First inspect the current "
                "workspace, TODO ledger, and any relevant outputs to determine what actually completed before the "
                "interruption. Do not blindly repeat an operation with side effects. Then continue autonomously with "
                "the next concrete unfinished action, verify the work, and continue toward the active goal."
            )
        return (
            "Start executing the active session goal now. Work autonomously on the next concrete unfinished action "
            "or TODO. Do not merely restate the goal or previous report; perform the work, verify it, and continue "
            "toward completion."
        )

    async def _run_goal_execution(self, chat_id: int, *, recovery_count: int, recovered: bool) -> None:
        started_at = time.time()
        self._write_goal_task_state(
            status="running",
            chat_id=chat_id,
            recovery_count=recovery_count,
            detail="automatic recovery" if recovered else "goal start",
            started_at=started_at,
        )
        try:
            await self.handle_task(
                chat_id,
                IncomingTask(text=self._goal_execution_text(recovered=recovered)),
            )
        except asyncio.CancelledError:
            self._write_goal_task_state(
                status="interrupted",
                chat_id=chat_id,
                recovery_count=recovery_count,
                detail="goal execution coroutine was cancelled",
                started_at=started_at,
            )
            raise
        except Exception as exc:
            self._write_goal_task_state(
                status="failed",
                chat_id=chat_id,
                recovery_count=recovery_count,
                detail=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
            )
            self._record_runtime_failure("supervised goal execution failed", exc)
            await self._queue_or_send_failure(chat_id, f"Goal execution failed unexpectedly: {exc}")
        else:
            goal = self.goal_manager.load()
            if self.cancel_event.is_set():
                status = "stopped"
                detail = "stopped by user"
            elif goal is not None and goal.status == "active":
                status = "paused"
                detail = "agent turn ended but the session goal remains active"
            else:
                status = "completed"
                detail = "session goal execution completed"
            self._write_goal_task_state(
                status=status,
                chat_id=chat_id,
                recovery_count=recovery_count,
                detail=detail,
                started_at=started_at,
            )

    def _spawn_supervised_goal_task(self, chat_id: int, *, recovery_count: int, recovered: bool) -> asyncio.Task[None]:
        task = asyncio.create_task(
            self._run_goal_execution(chat_id, recovery_count=recovery_count, recovered=recovered),
            name="gladiator-goal-recovery" if recovered else "gladiator-goal-start",
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
            self._record_runtime_failure("detached goal task escaped supervision", exc)
            asyncio.create_task(
                self._queue_or_send_failure(chat_id, f"Detached goal task crashed unexpectedly: {exc}")
            )

        task.add_done_callback(done)
        return task

    async def _recover_interrupted_goal_execution(self) -> None:
        state = self._load_goal_task_state()
        if state is None or state.get("status") not in {"running", "interrupted"}:
            return
        try:
            chat_id = int(state["chat_id"])
        except (KeyError, TypeError, ValueError):
            return
        previous_count = int(state.get("recovery_count") or 0)
        started_at = float(state.get("started_at") or time.time())
        next_count = previous_count + 1
        self._write_goal_task_state(
            status="interrupted",
            chat_id=chat_id,
            recovery_count=previous_count,
            detail="previous Gladiator process ended while this goal task was active",
            started_at=started_at,
        )

        goal = self.goal_manager.load()
        has_active_goal = goal is not None and goal.status == "active"
        if not has_active_goal:
            await self._try_send_html(
                chat_id,
                "<b>Previous Gladiator task was interrupted.</b> No active session goal remains, so it was not resumed.",
            )
            return

        if next_count > self.MAX_AUTOMATIC_GOAL_RECOVERIES:
            await self._try_send_html(
                chat_id,
                "<b>Automatic goal recovery stopped.</b> The task was interrupted repeatedly. "
                "The goal is still active; inspect <code>/task</code> and restart it manually with <code>/goal start</code>.",
            )
            return

        await self._try_send_html(
            chat_id,
            "<b>Recovered an interrupted goal task.</b> The previous Gladiator process ended while execution was "
            "still active. I am resuming the active goal after checking existing work first.",
        )
        self._spawn_supervised_goal_task(chat_id, recovery_count=next_count, recovered=True)

    def _task_status_html(self) -> str:
        state = self._load_goal_task_state()
        live = bool(getattr(self, "_run_lock", None) and self._run_lock.locked())
        lines = ["<b>Task execution</b>", f"Runtime task lock: <b>{'RUNNING' if live else 'idle'}</b>"]
        if state is None:
            lines.append("Persisted goal task: <i>none</i>")
            return "\n".join(lines)

        status = str(state.get("status") or "unknown")
        started_at = state.get("started_at")
        if isinstance(started_at, (int, float)):
            age = self._format_elapsed(max(0.0, time.time() - float(started_at)))
            lines.append(f"Persisted goal task: <b>{html.escape(status)}</b> · started {age} ago")
        else:
            lines.append(f"Persisted goal task: <b>{html.escape(status)}</b>")
        lines.append(f"Recorded PID: <code>{html.escape(str(state.get('pid') or 'unknown'))}</code>")
        lines.append(f"Recovery attempts: {int(state.get('recovery_count') or 0)}")
        detail = str(state.get("detail") or "").strip()
        if detail:
            lines.append(f"Detail: {html.escape(detail[:300])}")
        if status == "running" and not live:
            lines.append("⚠ Persisted state says running but this process has no live task lock; recovery is required.")
        return "\n".join(lines)

    def _provider_identity(self) -> tuple[str, str, str, str | None]:
        provider = self.config.provider
        return provider.mode, provider.base_url, provider.model, provider.codex_account_id

    async def _handle_model_command(self, chat_id: int, argument: str) -> None:
        provider = self.config.provider
        refresh_only = argument.lower() == "refresh"
        if argument and not refresh_only:
            provider.model = argument
            await self._refresh_model_context(clear_first=True)
        elif refresh_only:
            await self._refresh_model_context(clear_first=True)
        elif provider.context_window is None or provider.max_context_window is None:
            await self._refresh_model_context(clear_first=False)

        self._refresh_model_settings()
        save_config(self.config, self.config_path)
        source = provider.context_window_source or "not reported"
        await self.bot.client.send_message(
            chat_id,
            f"Model: <code>{html.escape(provider.model)}</code>\n"
            f"{html.escape(format_model_context(provider))}\n"
            f"Metadata source: <code>{html.escape(source)}</code>",
        )

    async def _refresh_model_context(self, *, clear_first: bool) -> bool:
        provider = self.config.provider
        if clear_first:
            clear_model_context(provider)
        info = await asyncio.to_thread(discover_model_context, provider)
        if info is None:
            save_config(self.config, self.config_path)
            self.agent.config.model_context_window = provider.context_window
            return False
        apply_model_context(provider, info)
        save_config(self.config, self.config_path)
        self.agent.config.model_context_window = provider.context_window
        return True

    async def _handle_mentor_command(self, chat_id: int, argument: str) -> None:
        mentor = self.config.mentor
        parts = argument.split(maxsplit=1)
        operation = parts[0].lower() if parts else ""
        value = parts[1].strip() if len(parts) == 2 else ""

        if operation == "on":
            mentor.enabled = True
            save_config(self.config, self.config_path)
            body = "Mentor mode enabled."
        elif operation == "off":
            mentor.enabled = False
            save_config(self.config, self.config_path)
            body = "Mentor mode disabled."
        elif operation == "model":
            if value:
                mentor.model = None if value.lower() in {"default", "main", "same"} else value
                save_config(self.config, self.config_path)
            selected = mentor.model or self.config.provider.model
            source = "main model fallback" if mentor.model is None else "mentor override"
            body = f"Mentor model: {selected} ({source})"
        elif operation == "reasoning":
            if value:
                if value not in REASONING_EFFORTS:
                    await self.bot.client.send_message(
                        chat_id,
                        "Invalid mentor reasoning level. Choose: <code>" + " | ".join(REASONING_EFFORTS) + "</code>",
                    )
                    return
                mentor.reasoning_effort = value  # type: ignore[assignment]
                save_config(self.config, self.config_path)
            body = f"Mentor reasoning: {mentor.reasoning_effort}"
        elif argument:
            body = "Usage: /mentor [on|off|model ID|model default|reasoning LEVEL]"
        else:
            selected = mentor.model or self.config.provider.model
            body = (
                f"Mentor: {'ON' if mentor.enabled else 'OFF'}\n"
                f"Model: {selected}\n"
                f"Reasoning: {mentor.reasoning_effort}\n"
                "The mentor is a one-shot read-only advisor. It receives only the question and files/logs "
                "the main agent explicitly supplies at invocation time; it has no shell or autonomous agent loop."
            )
        await self.bot.client.send_message(chat_id, "<b>Mentor</b>\n<pre>" + html.escape(body) + "</pre>")

    def _consult_mentor_from_agent_thread(self, request: MentorRequest) -> str:
        client = MentorClient(
            provider=self.config.provider,
            mentor=self.config.mentor,
            workspace_root=self.workspace,
            cancel_event=self.cancel_event,
        )
        return client.consult(request)

    def _context_html(self) -> str:
        base = super()._context_html()
        provider = self.config.provider
        if provider.max_context_window is None:
            maximum = "not reported"
        else:
            maximum = f"{provider.max_context_window:,} tokens"
        source = provider.context_window_source or "not reported"
        return base + f"\nMaximum advertised context: <b>{maximum}</b>\nContext metadata source: <code>{html.escape(source)}</code>"

    def _extended_status_html(self) -> str:
        base = super()._extended_status_html()
        selected = self.config.mentor.model or self.config.provider.model
        context_text = format_model_context(self.config.provider)
        return base + (
            "\n"
            f"Model limits: <b>{html.escape(context_text)}</b>\n"
            f"Mentor: <b>{'on' if self.config.mentor.enabled else 'off'}</b> · "
            f"<code>{html.escape(selected)}</code> · reasoning <code>{self.config.mentor.reasoning_effort}</code>"
        )
