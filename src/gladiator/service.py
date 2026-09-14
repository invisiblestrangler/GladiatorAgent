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
from gladiator.models import CodexOAuthStreamingModel, OpenAICompatibleStreamingModel, extract_chatgpt_account_id
from gladiator.runtime.decision import DecisionBroker, DecisionRequest, DecisionResult
from gladiator.skills import SkillManager, user_explicitly_requested_skill_write
from gladiator.telegram.bot import IncomingTask, TelegramBotRuntime
from gladiator.telegram.renderer import markdown_to_telegram_html, split_markdown
from gladiator.telegram.traces import TraceHighlighter, reasoning_preview, summarize_shell_command
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
        self.model = self._build_model()
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

    def _build_model(self) -> OpenAICompatibleStreamingModel:
        provider = self.config.provider
        if provider.mode == "codex_oauth":
            access_token = provider.codex_access_token.get_secret_value()
            account_id = provider.codex_account_id or extract_chatgpt_account_id(access_token)
            if not access_token:
                raise ValueError("Codex OAuth mode is enabled but no access token is stored. Use /codex connect <token>.")
            if not account_id:
                raise ValueError(
                    "Codex OAuth mode is enabled but the ChatGPT account id is unavailable. "
                    "Reconnect with /codex connect <token> <account-id>."
                )
            return CodexOAuthStreamingModel(
                access_token=access_token,
                account_id=account_id,
                responses_url=provider.codex_responses_url,
                model_name=provider.model,
                reasoning_effort=provider.reasoning_effort,
                event_sink=self._emit_from_agent_thread,
                cancel_event=self.cancel_event,
            )
        return OpenAICompatibleStreamingModel(
            base_url=provider.base_url,
            api_key=provider.api_key.get_secret_value(),
            model_name=provider.model,
            reasoning_effort=provider.reasoning_effort,
            event_sink=self._emit_from_agent_thread,
            cancel_event=self.cancel_event,
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
            finished = asyncio.Event()
            progress_final = "✓ Done"
            progress_summary: list[str] = []
            progress_message = await self.bot.client.send_message(
                chat_id,
                "<i>Working…</i>",
                reply_markup=self._stop_reply_markup(),
            )
            progress_message_id = int(progress_message["message_id"])
            typing_task = asyncio.create_task(self._typing_heartbeat(chat_id, finished))
            event_task = asyncio.create_task(self._consume_events(chat_id, progress_message_id, finished))
            try:
                result = await asyncio.to_thread(
                    self.agent.run_task,
                    incoming.text,
                    image_paths=incoming.image_paths,
                    file_paths=incoming.file_paths,
                )
                status = str(result.get("exit_status") or "Finished")
                submission = str(result.get("submission") or "").strip()
                if not submission:
                    submission = "Cancelled by user." if status == "Cancelled" else f"Gladiator stopped: {status}"
                if status == "Cancelled":
                    progress_final = "■ Stopped"
                elif status not in {"Submitted", "Finished", "Success", "Completed"}:
                    progress_final = f"⚠ {status}"
                await self._send_markdown(chat_id, submission)
            except Exception as exc:
                progress_final = "⚠ Error"
                await self.bot.client.send_message(
                    chat_id,
                    f"<b>Gladiator error</b>\n<code>{html.escape(str(exc))}</code>",
                )
                raise
            finally:
                finished.set()
                for task in (typing_task, event_task):
                    try:
                        task_result = await task
                        if task is event_task and isinstance(task_result, list):
                            progress_summary = task_result
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        console.print(f"[yellow]Telegram progress task failed: {exc}[/yellow]")
                final_progress = self._final_progress_body(progress_final, progress_summary)
                try:
                    await self.bot.client.edit_message_text(
                        chat_id,
                        progress_message_id,
                        markdown_to_telegram_html(final_progress),
                    )
                except Exception:
                    pass
                try:
                    await self.bot.client.edit_message_reply_markup(chat_id, progress_message_id, None)
                except Exception:
                    pass
                self.environment.skill_write_authorized = False
                self._current_chat_id = None

    @staticmethod
    def _stop_reply_markup() -> dict:
        return {"inline_keyboard": [[{"text": "Stop", "callback_data": "gladiator:stop"}]]}

    @staticmethod
    def _final_progress_body(status: str, summary: list[str]) -> str:
        if not summary:
            return status
        return status + "\n" + "\n".join(summary[:4])

    def _refresh_model_settings(self) -> None:
        provider = self.config.provider
        wants_codex = provider.mode == "codex_oauth"
        has_codex = isinstance(self.model, CodexOAuthStreamingModel)
        if wants_codex != has_codex:
            self.model = self._build_model()
            self.agent.model = self.model
        elif has_codex:
            access_token = provider.codex_access_token.get_secret_value()
            account_id = provider.codex_account_id or extract_chatgpt_account_id(access_token)
            if not account_id:
                raise ValueError("ChatGPT account id is unavailable. Reconnect with /codex connect <token> <account-id>.")
            self.model.access_token = access_token  # type: ignore[attr-defined]
            self.model.account_id = account_id  # type: ignore[attr-defined]
            self.model.responses_url = provider.codex_responses_url  # type: ignore[attr-defined]
            self.model.model_name = provider.model
            self.model.reasoning_effort = provider.reasoning_effort
        else:
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

    async def _consume_events(self, chat_id: int, message_id: int, finished: asyncio.Event) -> list[str]:
        highlighter = TraceHighlighter()
        milestones: deque[str] = deque(maxlen=5)
        verbose_reasoning = ""
        turn_preview_buffer = ""
        turn_preview_emitted = False
        first_activity_preview: str | None = None
        last_update = 0.0
        last_rendered = "<i>Working…</i>"

        while not finished.is_set() or not self._events.empty():
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if event.kind == EventKind.REASONING_DELTA:
                if self.config.runtime.trace_mode == "milestones":
                    if not turn_preview_emitted:
                        turn_preview_buffer += event.text
                        preview = reasoning_preview(turn_preview_buffer)
                        if preview:
                            item = f"💭 {preview}"
                            milestones.append(item)
                            if first_activity_preview is None:
                                first_activity_preview = item
                            turn_preview_emitted = True
                    for highlight in highlighter.feed(event.text):
                        item = f"💡 {highlight[:210]}"
                        if not milestones or milestones[-1] != item:
                            milestones.append(item)
                elif self.config.runtime.trace_mode == "verbose":
                    verbose_reasoning = (verbose_reasoning + event.text)[-1800:]
            elif event.kind == EventKind.RESPONSE_FINISHED:
                turn_preview_buffer = ""
                turn_preview_emitted = False
            elif event.kind == EventKind.TOOL_STARTED:
                summary = summarize_shell_command(str(event.data.get("command", "")))
                milestones.append(f"▶ {summary}")
            elif event.kind == EventKind.TOOL_FINISHED:
                summary = summarize_shell_command(str(event.data.get("command", "")))
                rc = event.data.get("returncode")
                completed = f"✓ {summary}" if rc == 0 else f"✗ {summary}"
                started = f"▶ {summary}"
                if milestones and milestones[-1] == started:
                    milestones[-1] = completed
                else:
                    milestones.append(completed)
            elif event.kind == EventKind.COMPACTION_STARTED:
                milestones.append("🧠 Compacting working context…")
            elif event.kind == EventKind.COMPACTION_FINISHED:
                milestones.append("✓ Context compacted; continuing")
            elif event.kind == EventKind.WARNING:
                milestones.append(f"⚠ {event.text[:220]}")
            elif event.kind == EventKind.STATUS:
                if event.data.get("replace_slow_stream"):
                    milestones = deque(
                        (item for item in milestones if not item.startswith("⏳ Still waiting for the model")),
                        maxlen=5,
                    )
                item = event.text[:220]
                if not milestones or milestones[-1] != item:
                    milestones.append(item)
            elif event.kind == EventKind.ARTIFACT_READY:
                path = Path(str(event.data["path"]))
                await self._send_artifact(chat_id, path, bool(event.data.get("is_image")))
                milestones.append(f"✓ Sent {path.name}")

            now = time.monotonic()
            if now - last_update < 0.55:
                continue
            body = self._draft_body(milestones, verbose_reasoning)
            rendered = markdown_to_telegram_html(body) if body else "<i>Working…</i>"
            if rendered == last_rendered:
                continue
            try:
                await self.bot.client.edit_message_text(
                    chat_id,
                    message_id,
                    rendered,
                    reply_markup=self._stop_reply_markup(),
                )
                last_rendered = rendered
                last_update = now
            except Exception:
                pass

        return self._summary_items(first_activity_preview, milestones)

    @staticmethod
    def _summary_items(first_activity_preview: str | None, milestones: deque[str]) -> list[str]:
        summary: list[str] = []
        if first_activity_preview:
            summary.append(first_activity_preview)
        for item in milestones:
            if item.startswith("▶ ") or item.startswith("⏳ Still waiting for the model") or item in summary:
                continue
            summary.append(item)
        if len(summary) <= 4:
            return summary
        if first_activity_preview and summary[0] == first_activity_preview:
            return [summary[0], *summary[-3:]]
        return summary[-4:]

    def _draft_body(self, milestones: deque[str], verbose_reasoning: str) -> str:
        trace_mode = self.config.runtime.trace_mode
        if trace_mode == "verbose" and verbose_reasoning.strip():
            return "Thinking…\n" + verbose_reasoning.strip()
        if milestones:
            return "Working…\n" + "\n".join(milestones)
        return ""

    async def _send_markdown(self, chat_id: int, text: str) -> None:
        for chunk in split_markdown(text, limit=3400):
            await self.bot.client.send_message(chat_id, markdown_to_telegram_html(chunk))

    async def _send_artifact(self, chat_id: int, path: Path, is_image: bool) -> None:
        if is_image:
            await self.bot.client.send_photo(chat_id, path)
        else:
            await self.bot.client.send_document(chat_id, path)
