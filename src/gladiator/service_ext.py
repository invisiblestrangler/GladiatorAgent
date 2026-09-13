from __future__ import annotations

import html
import json
import re
import shutil
import time
from contextvars import ContextVar
from difflib import SequenceMatcher

from gladiator.cache_session import CacheStats
from gladiator.events import AgentEvent, EventKind
from gladiator.goal import GoalManager, is_continuation_request
from gladiator.runtime.session import load_or_create_session, rotate_session
from gladiator.service import GladiatorService
from gladiator.telegram.bot import IncomingTask, TELEGRAM_COMMANDS
from gladiator.telegram.renderer import markdown_to_telegram_html, telegram_message_parts
from gladiator.todo import TodoManager

_GOAL_MARKER_RE = re.compile(r"(?:^|\n)\s*\[\[GLADIATOR_GOAL:\s*(achieved|incomplete)\s*\]\]\s*$", re.IGNORECASE)


class ExtendedGladiatorService(GladiatorService):
    """Gladiator service with persistent sessions, goals, restore, TODOs, and cache telemetry."""

    def __init__(self, *args, **kwargs):
        self.cache_stats = CacheStats()
        self._task_started_at: ContextVar[float | None] = ContextVar("gladiator_task_started_at", default=None)
        self._previous_final: ContextVar[str] = ContextVar("gladiator_previous_final", default="")
        self._continuation_work_required: ContextVar[bool] = ContextVar(
            "gladiator_continuation_work_required", default=False
        )
        self._duplicate_suppressed: ContextVar[bool] = ContextVar("gladiator_duplicate_suppressed", default=False)
        self._goal_assessment: ContextVar[str | None] = ContextVar("gladiator_goal_assessment", default=None)
        self._last_task_elapsed_seconds: float | None = None
        self._session_task_elapsed_seconds = 0.0
        self._session_task_count = 0
        super().__init__(*args, **kwargs)
        self.session_path = self.state_dir / "session.json"
        self.session = load_or_create_session(self.session_path)
        self.todo_manager = TodoManager(self.state_dir / "todo.json")
        self.goal_manager = GoalManager(self.state_dir / "goal.json")
        self._restore_trajectory()
        self._install_bot_extensions()
        self._install_native_runtime_commands()

    def _emit_from_agent_thread(self, event: AgentEvent) -> None:
        if event.kind == EventKind.RESPONSE_FINISHED:
            usage = event.data.get("usage")
            if isinstance(usage, dict):
                self.cache_stats.add_usage(usage)
        super()._emit_from_agent_thread(event)

    @staticmethod
    def _install_native_runtime_commands() -> None:
        if not any(name == "context" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(2, ("context", "Show current context usage"))
        if not any(name == "goal" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(3, ("goal", "Show or set the session goal"))

    # Backward-compatible name used by older tests/callers.
    _install_native_context_command = _install_native_runtime_commands

    def _install_bot_extensions(self) -> None:
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            command, _, argument = text.partition(" ")
            argument = argument.strip()
            if command == "/new":
                await self.handle_new_session(chat_id)
                return True
            if command == "/todo":
                await self.bot.client.send_message(
                    chat_id,
                    "<b>Task ledger</b>\n<pre>" + html.escape(self.todo_manager.render()) + "</pre>",
                )
                return True
            if command == "/goal":
                await self._handle_goal_command(chat_id, argument)
                return True
            if command == "/context":
                await self.bot.client.send_message(chat_id, self._context_html())
                return True
            handled = await original(chat_id, message, text)
            if handled and command in {"/start", "/help"}:
                await self.bot.client.send_message(
                    chat_id,
                    "/context — show current context usage\n"
                    "/goal [set TEXT|clear|reopen] — show/manage the session goal\n"
                    "/new — start a clean agent session\n"
                    "/todo — show the current task ledger",
                )
            return handled

        self.bot._handle_command = command_handler  # type: ignore[method-assign]
        self.bot._status_html = self._extended_status_html  # type: ignore[method-assign]

    async def _handle_goal_command(self, chat_id: int, argument: str) -> None:
        lowered = argument.lower()
        if not argument:
            body = self.goal_manager.render()
        elif lowered == "clear":
            self.goal_manager.clear()
            body = "Session goal cleared."
        elif lowered == "reopen":
            try:
                state = self.goal_manager.reopen(reason="Reopened by user from Telegram.")
                body = f"Goal reopened: {state.text}"
            except ValueError as exc:
                body = str(exc)
        elif lowered == "achieved":
            try:
                state = self.goal_manager.assess("achieved", reason="Marked achieved by user from Telegram.")
                body = f"Goal marked achieved: {state.text}"
            except ValueError as exc:
                body = str(exc)
        else:
            goal_text = argument[4:].strip() if lowered.startswith("set ") else argument
            try:
                state = self.goal_manager.set(goal_text, reason="Set by user from Telegram.")
                body = f"Goal set: {state.text}"
            except ValueError as exc:
                body = str(exc)
        await self.bot.client.send_message(chat_id, "<b>Session goal</b>\n<pre>" + html.escape(body) + "</pre>")

    def _restore_trajectory(self) -> None:
        path = self.state_dir / "trajectory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                return
            self.agent.messages = messages
            stats = data.get("info", {}).get("model_stats", {})
            self.agent.n_calls = int(stats.get("api_calls") or 0)
            self.agent.cost = float(stats.get("instance_cost") or 0.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    async def handle_task(self, chat_id: int, incoming: IncomingTask) -> None:
        started = time.monotonic()
        timing_token = self._task_started_at.set(started)
        previous_token = self._previous_final.set(self._latest_final_submission())
        raw_continuation = is_continuation_request(incoming.text)
        work_remaining = self._work_remains()
        continuation_token = self._continuation_work_required.set(raw_continuation and work_remaining)
        duplicate_token = self._duplicate_suppressed.set(False)
        assessment_token = self._goal_assessment.set(None)
        try:
            current = self._with_goal_context(
                incoming,
                force_continuation=raw_continuation and work_remaining,
            )
            for duplicate_retry in range(2):
                self._duplicate_suppressed.set(False)
                self._goal_assessment.set(None)
                await super().handle_task(chat_id, current)
                self._sanitize_goal_marker_from_history()
                if not self._duplicate_suppressed.get():
                    self._apply_goal_assessment()
                    break

                self._discard_suppressed_duplicate()
                if duplicate_retry:
                    await self.bot.client.send_message(
                        chat_id,
                        "<b>Gladiator stopped a repetition loop.</b> The model repeated the previous final answer "
                        "twice while continuation work was still pending. Check <code>/goal</code> and "
                        "<code>/todo</code>, then retry.",
                    )
                    break

                current = self._with_goal_context(
                    IncomingTask(
                        text=(
                            "Continue execution now. Do not restate or summarize the previous answer. "
                            "Complete the next unfinished TODO or take the next concrete action toward the active "
                            "session goal, verify the work, and only then return a new final answer."
                        )
                    ),
                    force_continuation=True,
                )
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self._last_task_elapsed_seconds = elapsed
            self._session_task_elapsed_seconds += elapsed
            self._session_task_count += 1
            self._goal_assessment.reset(assessment_token)
            self._duplicate_suppressed.reset(duplicate_token)
            self._continuation_work_required.reset(continuation_token)
            self._previous_final.reset(previous_token)
            self._task_started_at.reset(timing_token)

    def _with_goal_context(self, incoming: IncomingTask, *, force_continuation: bool = False) -> IncomingTask:
        goal = self.goal_manager.load()
        open_todos = [item for item in self.todo_manager.list() if not item.done]
        lines = ["[Gladiator runtime state — control metadata, not user prose]"]
        if goal is None:
            lines.append("Session goal: none.")
            lines.append(
                "If this request clearly establishes a multi-step objective, set one concise session goal once with "
                "`gladiator goal set '...'`; do not create a goal for trivial one-step work. If you set a goal during "
                "this turn, append a final `[[GLADIATOR_GOAL: achieved]]` or `[[GLADIATOR_GOAL: incomplete]]` control line."
            )
        else:
            lines.append(f"Session goal ({goal.status}): {goal.text}")
            if goal.status == "active":
                lines.append("[GLADIATOR_GOAL_TRACKING_REQUIRED]")
                lines.append(
                    "Before the final user-facing answer, independently decide whether the SESSION GOAL is actually "
                    "achieved. Append exactly one final control line: `[[GLADIATOR_GOAL: achieved]]` or "
                    "`[[GLADIATOR_GOAL: incomplete]]`. The runtime removes this line before displaying or retaining "
                    "the answer."
                )
                lines.append(
                    "Mark achieved only when the objective is genuinely complete; an answer/report alone is not completion."
                )

        if open_todos:
            lines.append(f"Open TODOs ({len(open_todos)}):")
            for item in open_todos[:8]:
                lines.append(f"- #{item.id} {item.text[:180]}")
            if len(open_todos) > 8:
                lines.append(f"- … {len(open_todos) - 8} more; use `gladiator todo show` if needed")

        work_remaining = (goal is not None and goal.status == "active") or bool(open_todos)
        if force_continuation and work_remaining:
            lines.append("[GLADIATOR_CONTINUATION_WORK_REQUIRED]")
            lines.append(
                "The user is telling you to CONTINUE EXECUTING. Do not repeat, paraphrase, or re-present the previous "
                "final answer. Immediately work on the next open TODO or next concrete action toward the active goal."
            )
        lines.append("[/Gladiator runtime state]")
        text = incoming.text.rstrip() + "\n\n" + "\n".join(lines)
        return IncomingTask(
            text=text,
            image_paths=list(incoming.image_paths),
            file_paths=list(incoming.file_paths),
            source_message_count=incoming.source_message_count,
        )

    def _work_remains(self) -> bool:
        goal = self.goal_manager.load()
        return (goal is not None and goal.status == "active") or self.todo_manager.open_count > 0

    def _latest_final_submission(self) -> str:
        for message in reversed(self.agent.messages):
            if message.get("role") == "exit":
                extra = message.get("extra")
                if isinstance(extra, dict) and extra.get("exit_status") == "Submitted":
                    submission = str(extra.get("submission") or message.get("content") or "").strip()
                    if submission:
                        return submission
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _extract_goal_marker(text: str) -> tuple[str, str | None]:
        match = _GOAL_MARKER_RE.search(text)
        if match is None:
            return text.strip(), None
        clean = (text[: match.start()] + text[match.end() :]).strip()
        return clean, match.group(1).lower()

    def _sanitize_goal_marker_from_history(self) -> None:
        """Keep goal control metadata out of future provider-visible conversation history."""
        for message in self.agent.messages[-3:]:
            role = message.get("role")
            if role not in {"assistant", "exit"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                clean, status = self._extract_goal_marker(content)
                if status is not None:
                    message["content"] = clean
                    extra = message.get("extra")
                    if isinstance(extra, dict):
                        extra["goal_status"] = status
            extra = message.get("extra")
            if isinstance(extra, dict) and isinstance(extra.get("submission"), str):
                clean, status = self._extract_goal_marker(str(extra["submission"]))
                if status is not None:
                    extra["submission"] = clean
                    extra["goal_status"] = status

    @staticmethod
    def _normalize_for_repeat_check(text: str) -> str:
        text, _status = ExtendedGladiatorService._extract_goal_marker(text)
        return " ".join(text.lower().split())

    @classmethod
    def _substantially_repeats(cls, previous: str, current: str) -> bool:
        old = cls._normalize_for_repeat_check(previous)
        new = cls._normalize_for_repeat_check(current)
        if not old or not new:
            return False
        if old == new:
            return True
        if min(len(old), len(new)) < 80:
            return False
        shorter, longer = (old, new) if len(old) <= len(new) else (new, old)
        if shorter in longer and len(shorter) / max(1, len(longer)) >= 0.82:
            return True
        return SequenceMatcher(None, old, new, autojunk=False).ratio() >= 0.88

    def _discard_suppressed_duplicate(self) -> None:
        while self.agent.messages and self.agent.messages[-1].get("role") == "exit":
            self.agent.messages.pop()
        previous = self._previous_final.get()
        if self.agent.messages and self.agent.messages[-1].get("role") == "assistant":
            content = str(self.agent.messages[-1].get("content") or "")
            if self._substantially_repeats(previous, content):
                self.agent.messages.pop()

    def _apply_goal_assessment(self) -> None:
        goal = self.goal_manager.load()
        if goal is None or goal.status != "active":
            return
        assessment = self._goal_assessment.get()
        if assessment == "achieved":
            open_count = self.todo_manager.open_count
            if open_count:
                self.goal_manager.assess(
                    "active",
                    reason=f"Agent assessed achieved, but {open_count} TODO(s) are still open.",
                )
            else:
                self.goal_manager.assess("achieved", reason="Agent assessed the session goal as achieved after this turn.")
        elif assessment == "incomplete":
            self.goal_manager.assess("active", reason="Agent assessed the session goal as incomplete after this turn.")

    def _final_progress_body(self, status: str, summary: list[str]) -> str:
        if self._duplicate_suppressed.get():
            status = "↻ Repeated final rejected; continuing"
        started = self._task_started_at.get()
        if started is not None:
            status = f"{status} · {self._format_elapsed(time.monotonic() - started)}"
        return super()._final_progress_body(status, summary)

    async def _send_markdown(self, chat_id: int, text: str) -> None:
        clean, assessment = self._extract_goal_marker(text)
        if assessment is not None:
            self._goal_assessment.set(assessment)
        previous = self._previous_final.get()
        if self._continuation_work_required.get() and self._substantially_repeats(previous, clean):
            self._duplicate_suppressed.set(True)
            return
        for part in telegram_message_parts(clean):
            await self.bot.client.send_message(chat_id, markdown_to_telegram_html(part))

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        if total < 60:
            return f"{total}s"
        minutes, secs = divmod(total, 60)
        if minutes < 60:
            return f"{minutes}m {secs:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m {secs:02d}s"

    @staticmethod
    def _usage_token_value(usage: dict, *keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, float) and value >= 0:
                return int(value)
        return None

    def _cache_average_text(self) -> str:
        stats = getattr(self, "cache_stats", None)
        if stats is None:
            return "not reported"
        average = stats.average_request_hit_ratio
        if average is None:
            return "not reported"
        return f"{average * 100:.1f}% across {stats.cache_reporting_requests} request(s)"

    def _context_html(self) -> str:
        try:
            estimated = self.agent.estimate_context_tokens() if self.agent.messages else 0
        except Exception:
            estimated = 0

        usage = self.model.last_usage if isinstance(self.model.last_usage, dict) else {}
        provider_prompt = self._usage_token_value(usage, "prompt_tokens", "input_tokens")
        effective = provider_prompt if provider_prompt is not None else estimated
        window = self.config.provider.context_window
        compact_target = self.agent.effective_compact_threshold

        lines = ["<b>Context usage</b>"]
        if provider_prompt is None:
            lines.append("Latest provider prompt: <i>not reported</i>")
        else:
            lines.append(f"Latest provider prompt: <b>{provider_prompt:,}</b> tokens")
        lines.append(f"Current local history estimate: ~<b>{estimated:,}</b> tokens")

        if window:
            percent = (effective / window * 100.0) if window > 0 else 0.0
            remaining = max(0, window - effective)
            lines.append(f"Configured model window: {window:,} tokens")
            lines.append(f"Approx. window used: <b>{percent:.1f}%</b> · ~{remaining:,} tokens remaining")
        else:
            lines.append("Configured model window: <i>unknown</i>")

        threshold_percent = (estimated / compact_target * 100.0) if compact_target > 0 else 0.0
        until_compact = max(0, compact_target - estimated)
        lines.append(
            f"Compaction target: {compact_target:,} tokens · <b>{threshold_percent:.1f}%</b> used · "
            f"~{until_compact:,} remaining"
        )
        lines.append(f"Avg provider cache hit / request: <b>{self._cache_average_text()}</b>")
        lines.append("Provider prompt tokens are from the most recent completed model request; the local estimate reflects current stored history.")
        return "\n".join(lines)

    async def handle_new_session(self, chat_id: int) -> None:
        if self._run_lock.locked():
            self.cancel_event.set()
        async with self._run_lock:
            old_id = self.session.session_id
            self._archive_session(old_id)
            self.agent.messages = []
            self.agent.n_calls = 0
            self.agent.cost = 0.0
            self.agent.n_consecutive_format_errors = 0
            self.agent.extra_template_vars.clear()
            self.cache_stats = CacheStats()
            self.model.last_usage = {}
            self._last_task_elapsed_seconds = None
            self._session_task_elapsed_seconds = 0.0
            self._session_task_count = 0
            for path in (self.state_dir / "trajectory.json", self.state_dir / "contextAfterCompact.md"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self.todo_manager.clear()
            self.goal_manager.clear()
            self.session = rotate_session(self.session_path)
            await self.bot.client.send_message(
                chat_id,
                "<b>New Gladiator session.</b> Conversation context, goal, and TODO state were cleared; "
                "workspace files, skills, and settings were kept."
                f"\nSession: <code>{html.escape(self.session.session_id[:20])}…</code>",
            )

    def _archive_session(self, session_id: str) -> None:
        archive = self.state_dir / "sessions"
        archive.mkdir(parents=True, exist_ok=True)
        for source, suffix in (
            (self.state_dir / "trajectory.json", "trajectory.json"),
            (self.state_dir / "contextAfterCompact.md", "contextAfterCompact.md"),
            (self.state_dir / "todo.json", "todo.json"),
            (self.state_dir / "goal.json", "goal.json"),
        ):
            if source.exists():
                shutil.copy2(source, archive / f"{session_id}.{suffix}")

    def _extended_status_html(self) -> str:
        ratio = self.cache_stats.hit_ratio
        cache_text = "not reported" if ratio is None else f"{ratio * 100:.1f}%"
        try:
            context_tokens = self.agent.estimate_context_tokens() if self.agent.messages else 0
        except Exception:
            context_tokens = 0
        last_task = "—" if self._last_task_elapsed_seconds is None else self._format_elapsed(self._last_task_elapsed_seconds)
        session_time = self._format_elapsed(self._session_task_elapsed_seconds)
        goal = self.goal_manager.load()
        goal_text = "none" if goal is None else f"{goal.status} · {goal.text[:90]}"
        return (
            "<b>Gladiator</b>\n"
            f"Provider: <code>{html.escape(self.config.provider.base_url)}</code>\n"
            f"Model: <code>{html.escape(self.config.provider.model)}</code>\n"
            f"Reasoning: <code>{self.config.provider.reasoning_effort}</code>\n"
            f"Trace: <code>{self.config.runtime.trace_mode}</code>\n"
            "YOLO: <b>on</b>\n"
            f"Session: <code>{html.escape(self.session.session_id[:20])}…</code>\n"
            f"Goal: <b>{html.escape(goal_text)}</b> · use /goal for details\n"
            f"Context: ~{context_tokens:,} tokens · use /context for details\n"
            f"Last task: {last_task}\n"
            f"Session task time: {session_time} across {self._session_task_count} task(s)\n"
            f"Open TODOs: {self.todo_manager.open_count}\n"
            f"Provider-reported prompt cache hit ratio: <b>{cache_text}</b>\n"
            f"Avg cache hit / request: <b>{self._cache_average_text()}</b>\n"
            f"Cached / prompt tokens: {self.cache_stats.cached_tokens:,} / {self.cache_stats.prompt_tokens:,}\n"
            f"Cache-write tokens: {self.cache_stats.cache_write_tokens:,}\n"
            f"Ask timeout: {self.config.runtime.escalation_timeout_seconds // 60} min\n"
            f"Compact target: {self.config.runtime.compact_threshold_tokens:,} tokens\n"
            f"Search: <code>{self.config.search.mode}</code>\n"
            f"Browser: {'enabled' if self.config.browser.enabled else 'not installed'}"
        )
