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
    TYPING_HEARTBEAT_SECONDS = 4.0
    TYPING_ACTION_TIMEOUT_SECONDS = 2.5
    PROGRESS_LIVENESS_SECONDS = 20.0
    FINAL_UI_RETRY_DELAYS = (0.0, 0.6, 1.8)

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
            self.environment.skill_write_authorized = user_explicitly_requested_skill_write(incoming.text)
            self._drain_event_queue()
            try:
                self._refresh_model_settings()
            except Exception as exc:
                self._record_runtime_failure("model setup failed", exc)
                await self._queue_or_send_failure(chat_id, f"Gladiator could not start the task: {exc}")
                self.environment.skill_write_authorized = False
                self._current_chat_id = None
                return

            await self._flush_pending_deliveries(chat_id)
            try:
                progress_message = await self.bot.client.send_message(
                    chat_id,
                    "<i>Working…</i>",
                    reply_markup=self._stop_reply_markup(),
                )
            except Exception as exc:
                self._record_runtime_failure("could not create Telegram progress message", exc)
                self.environment.skill_write_authorized = False
                self._current_chat_id = None
                return

            progress_message_id = int(progress_message["message_id"])
            events_finished = asyncio.Event()
            ui_finished = asyncio.Event()
            progress_final = "✓ Done"
            progress_summary: list[str] = []
            delivery_summary: list[str] = []
            typing_task = asyncio.create_task(self._typing_heartbeat(chat_id, ui_finished))
            event_task = asyncio.create_task(self._consume_events(chat_id, progress_message_id, events_finished))
            try:
                try:
                    result = await asyncio.to_thread(
                        self.agent.run_task,
                        incoming.text,
                        image_paths=incoming.image_paths,
                        file_paths=incoming.file_paths,
                    )
                except Exception as exc:
                    progress_final = "⚠ Error"
                    self._record_runtime_failure("agent task failed", exc)
                    pending = self._queue_pending_delivery(f"Gladiator error\n\n{exc}")
                    delivered = await self._try_send_html(
                        chat_id,
                        f"<b>Gladiator error</b>\n<code>{html.escape(str(exc))}</code>",
                    )
                    if delivered:
                        self._remove_pending_delivery(pending)
                    else:
                        delivery_summary.append("⚠ Error report saved for recovery")
                else:
                    status = str(result.get("exit_status") or "Finished")
                    submission = str(result.get("submission") or "").strip()
                    if not submission:
                        submission = "Cancelled by user." if status == "Cancelled" else f"Gladiator stopped: {status}"
                    if status == "Cancelled":
                        progress_final = "■ Stopped"
                    elif status not in {"Submitted", "Finished", "Success", "Completed"}:
                        progress_final = f"⚠ {status}"
                    try:
                        await self._send_markdown(chat_id, submission)
                    except Exception as exc:
                        progress_final = "⚠ Delivery failed"
                        self._record_runtime_failure("final Telegram response delivery failed", exc)
                        self._queue_pending_delivery(submission)
                        delivery_summary.append("⚠ Final response saved locally for automatic recovery")
            finally:
                events_finished.set()
                try:
                    task_result = await event_task
                    if isinstance(task_result, list):
                        progress_summary = task_result
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._record_runtime_failure("Telegram event/progress task failed", exc)
                    delivery_summary.append("⚠ Progress renderer failed")

                final_progress = self._final_progress_body(progress_final, [*progress_summary, *delivery_summary])
                await self._finalize_progress_message(chat_id, progress_message_id, final_progress)
                ui_finished.set()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._record_runtime_failure("typing heartbeat task failed", exc)
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

    def _record_runtime_failure(self, label: str, exc: Exception) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {label}: {type(exc).__name__}: {exc}"
        console.print(f"[yellow]{line}[/yellow]")
        try:
            path = self.state_dir / "runtime-errors.log"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    @property
    def _pending_delivery_dir(self) -> Path:
        return self.state_dir / "pending-delivery"

    def _queue_pending_delivery(self, text: str) -> Path:
        directory = self._pending_delivery_dir
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{time.time_ns()}.md"
        target.write_text(text.rstrip() + "\n", encoding="utf-8")
        return target

    @staticmethod
    def _remove_pending_delivery(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    async def _flush_pending_deliveries(self, chat_id: int) -> None:
        directory = self._pending_delivery_dir
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                self._record_runtime_failure("could not read pending Telegram delivery", exc)
                continue
            if not text:
                self._remove_pending_delivery(path)
                continue
            try:
                await self.bot.client.send_message(
                    chat_id,
                    "<b>Recovered undelivered output from an earlier Gladiator task:</b>",
                )
                await self._send_markdown(chat_id, text)
            except Exception as exc:
                self._record_runtime_failure("pending Telegram delivery retry failed", exc)
                return
            self._remove_pending_delivery(path)

    async def _queue_or_send_failure(self, chat_id: int, text: str) -> None:
        pending = self._queue_pending_delivery(text)
        delivered = await self._try_send_html(
            chat_id,
            "<b>Gladiator error</b>\n<code>" + html.escape(text) + "</code>",
        )
        if delivered:
            self._remove_pending_delivery(pending)

    async def _try_send_html(self, chat_id: int, text: str) -> bool:
        try:
            await self.bot.client.send_message(chat_id, text)
            return True
        except Exception as exc:
            self._record_runtime_failure("Telegram message delivery failed", exc)
            return False

    async def _finalize_progress_message(self, chat_id: int, message_id: int, final_progress: str) -> None:
        rendered = markdown_to_telegram_html(final_progress)
        edited = False
        for delay in self.FINAL_UI_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                await self.bot.client.edit_message_text(chat_id, message_id, rendered)
                edited = True
                break
            except Exception as exc:
                last_edit_error = exc
        if not edited:
            self._record_runtime_failure("final Telegram progress edit failed", last_edit_error)

        cleared = False
        for delay in self.FINAL_UI_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                await self.bot.client.edit_message_reply_markup(chat_id, message_id, None)
                cleared = True
                break
            except Exception as exc:
                last_markup_error = exc
        if not cleared:
            self._record_runtime_failure("final Telegram Stop-button removal failed", last_markup_error)

    async def _typing_heartbeat(self, chat_id: int, finished: asyncio.Event) -> None:
        consecutive_failures = 0
        while not finished.is_set():
            try:
                await asyncio.wait_for(
                    self.bot.client.send_chat_action(chat_id, "typing"),
                    timeout=self.TYPING_ACTION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    self._record_runtime_failure("Telegram typing heartbeat timed out", exc)
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    self._record_runtime_failure("Telegram typing heartbeat failed", exc)
            else:
                if consecutive_failures:
                    console.print(
                        f"[green]Telegram typing heartbeat recovered after {consecutive_failures} failure(s).[/green]"
                    )
                consecutive_failures = 0
            try:
                await asyncio.wait_for(finished.wait(), timeout=self.TYPING_HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                continue

    async def _consume_events(self, chat_id: int, message_id: int, finished: asyncio.Event) -> list[str]:
        highlighter = TraceHighlighter()
        milestones: deque[str] = deque(maxlen=5)
        verbose_reasoning = ""
        turn_preview_buffer = ""
        turn_preview_emitted = False
        first_activity_preview: str | None = None
        started_at = time.monotonic()
        last_update = 0.0
        last_liveness = started_at
        last_rendered = "<i>Working…</i>"
        progress_edit_failures = 0

        while not finished.is_set() or not self._events.empty():
            event: AgentEvent | None = None
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

            if event is not None:
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
                    try:
                        await self._send_artifact(chat_id, path, bool(event.data.get("is_image")))
                    except Exception as exc:
                        self._record_runtime_failure(f"artifact delivery failed for {path.name}", exc)
                        milestones.append(f"⚠ Could not send {path.name}")
                    else:
                        milestones.append(f"✓ Sent {path.name}")

            now = time.monotonic()
            should_show_liveness = now - last_liveness >= self.PROGRESS_LIVENESS_SECONDS
            if event is not None and now - last_update < 0.55 and not should_show_liveness:
                continue
            if event is None and not should_show_liveness:
                continue

            body = self._draft_body(milestones, verbose_reasoning)
            if should_show_liveness:
                liveness = f"⏱ Still working · {self._format_elapsed_short(now - started_at)}"
                body = (body + "\n" + liveness) if body else ("Working…\n" + liveness)
                last_liveness = now
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
                progress_edit_failures = 0
            except Exception as exc:
                progress_edit_failures += 1
                if progress_edit_failures == 1 or progress_edit_failures % 10 == 0:
                    self._record_runtime_failure("Telegram progress edit failed", exc)

        return self._summary_items(first_activity_preview, milestones)

    @staticmethod
    def _format_elapsed_short(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        if minutes < 60:
            return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

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
