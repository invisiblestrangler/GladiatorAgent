from __future__ import annotations

import html
import json
import shutil
import time
from contextvars import ContextVar

from gladiator.cache_session import CacheStats
from gladiator.events import AgentEvent, EventKind
from gladiator.runtime.session import load_or_create_session, rotate_session
from gladiator.service import GladiatorService
from gladiator.telegram.bot import TELEGRAM_COMMANDS
from gladiator.telegram.renderer import markdown_to_telegram_html, telegram_message_parts
from gladiator.todo import TodoManager


class ExtendedGladiatorService(GladiatorService):
    """Gladiator service with persistent sessions, restore, /new, /todo, and cache telemetry."""

    def __init__(self, *args, **kwargs):
        self.cache_stats = CacheStats()
        self._task_started_at: ContextVar[float | None] = ContextVar("gladiator_task_started_at", default=None)
        self._last_task_elapsed_seconds: float | None = None
        self._session_task_elapsed_seconds = 0.0
        self._session_task_count = 0
        super().__init__(*args, **kwargs)
        self.session_path = self.state_dir / "session.json"
        self.session = load_or_create_session(self.session_path)
        self.todo_manager = TodoManager(self.state_dir / "todo.json")
        self._restore_trajectory()
        self._install_bot_extensions()
        self._install_native_context_command()

    def _emit_from_agent_thread(self, event: AgentEvent) -> None:
        if event.kind == EventKind.RESPONSE_FINISHED:
            usage = event.data.get("usage")
            if isinstance(usage, dict):
                self.cache_stats.add_usage(usage)
        super()._emit_from_agent_thread(event)

    @staticmethod
    def _install_native_context_command() -> None:
        if not any(name == "context" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(2, ("context", "Show current context usage"))

    def _install_bot_extensions(self) -> None:
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            command = text.partition(" ")[0]
            if command == "/new":
                await self.handle_new_session(chat_id)
                return True
            if command == "/todo":
                await self.bot.client.send_message(
                    chat_id,
                    "<b>Task ledger</b>\n<pre>" + html.escape(self.todo_manager.render()) + "</pre>",
                )
                return True
            if command == "/context":
                await self.bot.client.send_message(chat_id, self._context_html())
                return True
            handled = await original(chat_id, message, text)
            if handled and command in {"/start", "/help"}:
                await self.bot.client.send_message(
                    chat_id,
                    "/context — show current context usage\n"
                    "/new — start a clean agent session\n"
                    "/todo — show the current task ledger",
                )
            return handled

        self.bot._handle_command = command_handler  # type: ignore[method-assign]
        self.bot._status_html = self._extended_status_html  # type: ignore[method-assign]

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

    async def handle_task(self, chat_id: int, incoming) -> None:
        started = time.monotonic()
        token = self._task_started_at.set(started)
        try:
            await super().handle_task(chat_id, incoming)
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self._last_task_elapsed_seconds = elapsed
            self._session_task_elapsed_seconds += elapsed
            self._session_task_count += 1
            self._task_started_at.reset(token)

    def _final_progress_body(self, status: str, summary: list[str]) -> str:
        started = self._task_started_at.get()
        if started is not None:
            status = f"{status} · {self._format_elapsed(time.monotonic() - started)}"
        return super()._final_progress_body(status, summary)

    async def _send_markdown(self, chat_id: int, text: str) -> None:
        for part in telegram_message_parts(text):
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
            self.session = rotate_session(self.session_path)
            await self.bot.client.send_message(
                chat_id,
                "<b>New Gladiator session.</b> Conversation context and TODO state were cleared; "
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
        return (
            "<b>Gladiator</b>\n"
            f"Provider: <code>{html.escape(self.config.provider.base_url)}</code>\n"
            f"Model: <code>{html.escape(self.config.provider.model)}</code>\n"
            f"Reasoning: <code>{self.config.provider.reasoning_effort}</code>\n"
            f"Trace: <code>{self.config.runtime.trace_mode}</code>\n"
            "YOLO: <b>on</b>\n"
            f"Session: <code>{html.escape(self.session.session_id[:20])}…</code>\n"
            f"Context: ~{context_tokens:,} tokens · use /context for details\n"
            f"Last task: {last_task}\n"
            f"Session task time: {session_time} across {self._session_task_count} task(s)\n"
            f"Open TODOs: {self.todo_manager.open_count}\n"
            f"Provider-reported prompt cache hit ratio: <b>{cache_text}</b>\n"
            f"Cached / prompt tokens: {self.cache_stats.cached_tokens:,} / {self.cache_stats.prompt_tokens:,}\n"
            f"Cache-write tokens: {self.cache_stats.cache_write_tokens:,}\n"
            f"Ask timeout: {self.config.runtime.escalation_timeout_seconds // 60} min\n"
            f"Compact target: {self.config.runtime.compact_threshold_tokens:,} tokens\n"
            f"Search: <code>{self.config.search.mode}</code>\n"
            f"Browser: {'enabled' if self.config.browser.enabled else 'not installed'}"
        )
