from __future__ import annotations

import html
import json
import shutil

from gladiator.cache_session import CacheStats
from gladiator.events import AgentEvent, EventKind
from gladiator.runtime.session import load_or_create_session, rotate_session
from gladiator.service import GladiatorService


class ExtendedGladiatorService(GladiatorService):
    """Gladiator service with persistent sessions, restore, /new, and cache telemetry."""

    def __init__(self, *args, **kwargs):
        self.cache_stats = CacheStats()
        super().__init__(*args, **kwargs)
        self.session_path = self.state_dir / "session.json"
        self.session = load_or_create_session(self.session_path)
        self.model.session_id = self.session.session_id
        self._restore_trajectory()
        self._install_bot_extensions()

    def _emit_from_agent_thread(self, event: AgentEvent) -> None:
        if event.kind == EventKind.RESPONSE_FINISHED:
            usage = event.data.get("usage")
            if isinstance(usage, dict):
                self.cache_stats.add_usage(usage)
        super()._emit_from_agent_thread(event)

    def _install_bot_extensions(self) -> None:
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            command = text.partition(" ")[0]
            if command == "/new":
                await self.handle_new_session(chat_id)
                return True
            handled = await original(chat_id, message, text)
            if handled and command in {"/start", "/help"}:
                await self.bot.client.send_message(chat_id, "/new — start a clean agent session")
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
            self.model.reset_cache_metrics()
            for path in (self.state_dir / "trajectory.json", self.state_dir / "contextAfterCompact.md"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self.session = rotate_session(self.session_path)
            self.model.session_id = self.session.session_id
            await self.bot.client.send_message(
                chat_id,
                "<b>New Gladiator session.</b> Conversation context was cleared; workspace files, skills, and settings were kept."
                f"\nSession: <code>{html.escape(self.session.session_id[:20])}…</code>",
            )

    def _archive_session(self, session_id: str) -> None:
        archive = self.state_dir / "sessions"
        archive.mkdir(parents=True, exist_ok=True)
        for source, suffix in (
            (self.state_dir / "trajectory.json", "trajectory.json"),
            (self.state_dir / "contextAfterCompact.md", "contextAfterCompact.md"),
        ):
            if source.exists():
                shutil.copy2(source, archive / f"{session_id}.{suffix}")

    def _extended_status_html(self) -> str:
        ratio = self.cache_stats.hit_ratio
        cache_text = "not reported yet" if ratio is None else f"{ratio * 100:.1f}%"
        try:
            context_tokens = self.agent.estimate_context_tokens() if self.agent.messages else 0
        except Exception:
            context_tokens = 0
        return (
            "<b>Gladiator</b>\n"
            f"Provider: <code>{html.escape(self.config.provider.base_url)}</code>\n"
            f"Model: <code>{html.escape(self.config.provider.model)}</code>\n"
            f"Reasoning: <code>{self.config.provider.reasoning_effort}</code>\n"
            f"Trace: <code>{self.config.runtime.trace_mode}</code>\n"
            "YOLO: <b>on</b>\n"
            f"Session: <code>{html.escape(self.session.session_id[:20])}…</code>\n"
            f"Context: ~{context_tokens:,} tokens\n"
            f"Prompt cache hit ratio: <b>{cache_text}</b>\n"
            f"Cached / prompt tokens: {self.cache_stats.cached_tokens:,} / {self.cache_stats.prompt_tokens:,}\n"
            f"Cache-write tokens: {self.cache_stats.cache_write_tokens:,}\n"
            f"Response cache: <code>{html.escape(self.model.last_response_cache_status or 'not reported')}</code>\n"
            f"Ask timeout: {self.config.runtime.escalation_timeout_seconds // 60} min\n"
            f"Compact target: {self.config.runtime.compact_threshold_tokens:,} tokens\n"
            f"Search: <code>{self.config.search.mode}</code>\n"
            f"Browser: {'enabled' if self.config.browser.enabled else 'not installed'}"
        )
