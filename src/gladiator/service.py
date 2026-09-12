from __future__ import annotations

import asyncio
import html
import time
from collections import deque
from pathlib import Path
from threading import Event

from rich.console import Console

from gladiator.agent import GladiatorAgent
from gladiator.config import GladiatorConfig, data_root
from gladiator.environment import GladiatorLocalEnvironment
from gladiator.events import AgentEvent, EventKind
from gladiator.models import OpenAICompatibleStreamingModel
from gladiator.runtime.decision import DecisionBroker, DecisionRequest, DecisionResult
from gladiator.skills import SkillManager, user_explicitly_requested_skill_write
from gladiator.telegram.bot import IncomingTask, TelegramBotRuntime
from gladiator.telegram.renderer import markdown_to_telegram_html, split_markdown
from gladiator.telegram.traces import TraceHighlighter
from gladiator.webtools import WebTools

console = Console()


class GladiatorService:
    def __init__(self, *, config: GladiatorConfig, config_path: Path, workspace: Path):
        self.config = config
        self.config_path = config_path
        self.workspace = workspace.expanduser().resolve()
        self.state_dir = self.workspace / ".gladiator"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_event = Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._run_lock = asyncio.Lock()
        self._current_chat_id: int | None = None

        provider = config.provider
        self.model = OpenAICompatibleStreamingModel(
            base_url=provider.base_url,
            api_key=provider.api_key.get_secret_value(),
            model_name=provider.model,
            reasoning_effort=provider.reasoning_effort,
            event_sink=self._emit_from_agent_thread,
            cancel_event=self.cancel_event,
        )
        web_tools = WebTools(
            searxng_url=config.search.searxng_url,
            storage_dir=self.state_dir / "web",
            char_limit=config.runtime.web_observation_char_limit,
            search_result_limit=config.runtime.search_result_limit,
        ) if config.search.mode != "none" else None
        self.skill_manager = SkillManager(data_root() / "skills")
        self.environment = GladiatorLocalEnvironment(
            output_dir=self.state_dir / "tool-output",
            workspace_root=self.workspace,
            observation_char_limit=config.runtime.shell_observation_char_limit,
            event_sink=self._emit_from_agent_thread,
            decision_handler=self._resolve_decision_from_agent_thread,
            web_tools=web_tools,
            skill_manager=self.skill_manager,
            cwd=str(self.workspace),
            env={"PAGER": "cat", "MANPAGER": "cat", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"},
            timeout=120,
        )
        self.agent = GladiatorAgent(
            self.model,
            self.environment,
            event_sink=self._emit_from_agent_thread,
            context_after_compact_path=self.state_dir / "contextAfterCompact.md",
            compact_threshold_tokens=config.runtime.compact_threshold_tokens,
            compact_fraction_of_model_window=config.runtime.compact_fraction_of_model_window,
            model_context_window=provider.context_window,
            output_path=self.state_dir / "trajectory.json",
            step_limit=0,
            cost_limit=0.0,
        )
        self.bot = TelegramBotRuntime(
            config=config,
            config_path=config_path,
            workspace=self.workspace,
            on_task=self.handle_task,
            on_stop=self.handle_stop,
            on_compact=self.handle_compact,
        )

    async def run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        console.print(f"[green]Gladiator workspace:[/green] {self.workspace}")
        await self.bot.run_forever()

    def _emit_from_agent_thread(self, event: AgentEvent) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _resolve_decision_from_agent_thread(self, request: DecisionRequest) -> DecisionResult:
        loop = self._loop
        chat_id = self._current_chat_id
        if loop is None or chat_id is None:
            return DecisionResult(choice=request.conservative_choice, timed_out=True)
        future = asyncio.run_coroutine_threadsafe(self._resolve_decision(chat_id, request), loop)
        try:
            return future.result(timeout=self.config.runtime.escalation_timeout_seconds + 30)
        except Exception:
            future.cancel()
            return DecisionResult(choice=request.conservative_choice, timed_out=True)

    async def _resolve_decision(self, chat_id: int, request: DecisionRequest) -> DecisionResult:
        broker = DecisionBroker(
            lambda req: self.bot.ask_decision(chat_id, req),
            timeout_seconds=self.config.runtime.escalation_timeout_seconds,
        )
        result = await broker.resolve(request)
        if result.timed_out:
            await self.bot.client.send_message(
                chat_id,
                f"No reply before the decision timeout. Continuing conservatively with "
                f"<code>{html.escape(result.choice)}</code>.",
            )
        return result

    async def handle_stop(self, chat_id: int) -> None:
        if self._current_chat_id == chat_id:
            self.cancel_event.set()

    async def handle_compact(self, _chat_id: int) -> None:
        self.agent.request_compaction()

    async def handle_task(self, chat_id: int, incoming: IncomingTask) -> None:
        if self._run_lock.locked():
            await self.bot.client.send_message(chat_id, "Queued behind the current Gladiator task.")
        async with self._run_lock:
            self._current_chat_id = chat_id
            self.cancel_event.clear()
            self._refresh_model_settings()
            self.environment.skill_write_authorized = user_explicitly_requested_skill_write(incoming.text)
            self._drain_event_queue()
            draft_id = int(time.time_ns() % 2_000_000_000) or 1
            finished = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_heartbeat(chat_id, finished))
            event_task = asyncio.create_task(self._consume_events(chat_id, draft_id, finished))
            try:
                await self.bot.client.send_message_draft(chat_id, draft_id, "", can_stop=True)
                result = await asyncio.to_thread(
                    self.agent.run_task,
                    incoming.text,
                    image_paths=incoming.image_paths,
                    file_paths=incoming.file_paths,
                )
                submission = str(result.get("submission") or "").strip()
                if not submission:
                    status = str(result.get("exit_status") or "Finished")
                    submission = "Cancelled by user." if status == "Cancelled" else f"Gladiator stopped: {status}"
                await self._send_markdown(chat_id, submission)
            except Exception as exc:
                await self.bot.client.send_message(
                    chat_id,
                    f"<b>Gladiator error</b>\n<code>{html.escape(str(exc))}</code>",
                )
                raise
            finally:
                finished.set()
                for task in (typing_task, event_task):
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        console.print(f"[yellow]Telegram progress task failed: {exc}[/yellow]")
                self.environment.skill_write_authorized = False
                self._current_chat_id = None

    def _refresh_model_settings(self) -> None:
        provider = self.config.provider
        self.model.base_url = provider.base_url.rstrip("/")
        self.model.api_key = provider.api_key.get_secret_value()
        self.model.model_name = provider.model
        self.model.reasoning_effort = provider.reasoning_effort
        self.agent.config.model_context_window = provider.context_window

    def _drain_event_queue(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _typing_heartbeat(self, chat_id: int, finished: asyncio.Event) -> None:
        while not finished.is_set():
            try:
                await self.bot.client.send_chat_action(chat_id, "typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(finished.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue

    async def _consume_events(self, chat_id: int, draft_id: int, finished: asyncio.Event) -> None:
        highlighter = TraceHighlighter()
        milestones: deque[str] = deque(maxlen=8)
        visible_text = ""
        verbose_reasoning = ""
        last_update = 0.0

        while not finished.is_set() or not self._events.empty():
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if event.kind == EventKind.REASONING_DELTA:
                if self.config.runtime.trace_mode == "milestones":
                    milestones.extend(highlighter.feed(event.text))
                elif self.config.runtime.trace_mode == "verbose":
                    verbose_reasoning = (verbose_reasoning + event.text)[-1800:]
            elif event.kind == EventKind.TEXT_DELTA:
                visible_text = (visible_text + event.text)[-1800:]
            elif event.kind == EventKind.TOOL_STARTED:
                command = str(event.data.get("command", "")).replace("\n", " ")
                milestones.append(f"▶ {command[:220]}")
            elif event.kind == EventKind.TOOL_FINISHED:
                rc = event.data.get("returncode")
                milestones.append("✓ command finished" if rc == 0 else f"✗ command exited {rc}")
            elif event.kind == EventKind.COMPACTION_STARTED:
                milestones.append("🧠 Compacting working context…")
            elif event.kind == EventKind.COMPACTION_FINISHED:
                milestones.append("✓ Context compacted; continuing")
            elif event.kind == EventKind.WARNING:
                milestones.append(f"⚠ {event.text[:300]}")
            elif event.kind == EventKind.ARTIFACT_READY:
                await self._send_artifact(chat_id, Path(str(event.data["path"])), bool(event.data.get("is_image")))

            now = time.monotonic()
            if now - last_update < 0.55:
                continue
            body = self._draft_body(milestones, visible_text, verbose_reasoning)
            try:
                await self.bot.client.send_message_draft(
                    chat_id,
                    draft_id,
                    markdown_to_telegram_html(body) if body else "",
                    can_stop=True,
                )
                last_update = now
            except Exception:
                pass

    def _draft_body(self, milestones: deque[str], visible_text: str, verbose_reasoning: str) -> str:
        parts: list[str] = []
        trace_mode = self.config.runtime.trace_mode
        if trace_mode == "verbose" and verbose_reasoning.strip():
            parts.append("Thinking…\n" + verbose_reasoning.strip())
        elif milestones:
            parts.append("Working…\n" + "\n".join(milestones))
        if visible_text.strip():
            parts.append(visible_text.strip())
        return "\n\n".join(parts)[-3400:]

    async def _send_markdown(self, chat_id: int, text: str) -> None:
        for chunk in split_markdown(text, limit=3400):
            await self.bot.client.send_message(chat_id, markdown_to_telegram_html(chunk))

    async def _send_artifact(self, chat_id: int, path: Path, is_image: bool) -> None:
        if is_image:
            await self.bot.client.send_photo(chat_id, path)
        else:
            await self.bot.client.send_document(chat_id, path)
