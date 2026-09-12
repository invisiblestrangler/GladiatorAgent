from __future__ import annotations

import asyncio
import html
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from rich.console import Console

from gladiator.config import GladiatorConfig, save_config
from gladiator.runtime.decision import DecisionRequest
from gladiator.telegram.client import TelegramClient

console = Console()

TELEGRAM_COMMANDS: list[tuple[str, str]] = [
    ("start", "Show Gladiator help"),
    ("status", "Show runtime, model, context and cache status"),
    ("model", "Show or change the model"),
    ("reasoning", "Show or change reasoning level"),
    ("trace", "Show or change progress trace mode"),
    ("provider", "Show or change OpenAI-compatible endpoint"),
    ("compact", "Compact context at the next safe boundary"),
    ("new", "Start a clean agent session"),
    ("todo", "Show the current task ledger"),
    ("stop", "Stop the current agent run"),
    ("help", "Show Gladiator help"),
]


@dataclass(slots=True)
class IncomingTask:
    text: str
    image_paths: list[Path] = field(default_factory=list)
    file_paths: list[Path] = field(default_factory=list)


TaskHandler = Callable[[int, IncomingTask], Awaitable[None]]
SimpleHandler = Callable[[int], Awaitable[None]]


class TelegramBotRuntime:
    def __init__(
        self,
        *,
        config: GladiatorConfig,
        config_path: Path,
        workspace: Path,
        on_task: TaskHandler,
        on_stop: SimpleHandler,
        on_compact: SimpleHandler,
    ):
        self.config = config
        self.config_path = config_path
        self.workspace = workspace.resolve()
        self.inbox = self.workspace / ".gladiator" / "inbox"
        self.on_task = on_task
        self.on_stop = on_stop
        self.on_compact = on_compact
        self.client = TelegramClient(config.telegram.bot_token.get_secret_value())
        self.offset: int | None = None
        self.pairing_code = f"{secrets.randbelow(900000) + 100000}"
        self._spawned: set[asyncio.Task] = set()
        self._decision_waiters: dict[int, tuple[DecisionRequest, asyncio.Future[str | None]]] = {}

    async def run_forever(self) -> None:
        try:
            await self.client.set_my_commands(TELEGRAM_COMMANDS)
        except Exception as exc:
            console.print(f"[yellow]Could not register Telegram command menu: {exc}[/yellow]")

        if not self.config.telegram.allowed_user_ids:
            console.print(
                "[yellow]Telegram is not paired yet.[/yellow] "
                f"Send [bold]/pair {self.pairing_code}[/bold] to the bot from your private Telegram account."
            )
        try:
            while True:
                updates = await self.client.get_updates(offset=self.offset, timeout=30)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    await self._handle_update(update)
        finally:
            for task in self._spawned:
                task.cancel()
            await self.client.close()

    async def _handle_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            callback_id = str(callback.get("id") or "")
            data = str(callback.get("data") or "")
            message = callback.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if isinstance(chat_id, int) and self._authorized(chat_id) and data == "gladiator:stop":
                await self.on_stop(chat_id)
                if callback_id:
                    await self.client.answer_callback_query(callback_id, "Stop requested.")
            elif callback_id:
                await self.client.answer_callback_query(callback_id)
            return

        stopped = update.get("stopped_message_generation")
        if isinstance(stopped, dict):
            chat = stopped.get("chat") or {}
            chat_id = chat.get("id")
            if isinstance(chat_id, int) and self._authorized(chat_id):
                await self.on_stop(chat_id)
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not isinstance(chat_id, int) or chat.get("type") != "private":
            return
        text = str(message.get("text") or message.get("caption") or "").strip()

        if not self._authorized(chat_id):
            await self._handle_pairing(chat_id, text)
            return

        if chat_id in self._decision_waiters:
            await self._handle_decision_reply(chat_id, text)
            return

        if text.startswith("/") and await self._handle_command(chat_id, message, text):
            return

        incoming = await self._build_task(message, text)
        task = asyncio.create_task(self.on_task(chat_id, incoming))
        self._spawned.add(task)
        task.add_done_callback(self._spawned.discard)

    def _authorized(self, chat_id: int) -> bool:
        return chat_id in self.config.telegram.allowed_user_ids

    async def _handle_pairing(self, chat_id: int, text: str) -> None:
        if text == f"/pair {self.pairing_code}" and not self.config.telegram.allowed_user_ids:
            self.config.telegram.allowed_user_ids.append(chat_id)
            save_config(self.config, self.config_path)
            await self.client.send_message(chat_id, "<b>Paired.</b> Gladiator is ready.")
            console.print(f"[green]Paired Telegram user {chat_id}.[/green]")
            return
        if text.startswith("/pair"):
            await self.client.send_message(chat_id, "Pairing code rejected.")

    async def _handle_command(self, chat_id: int, message: dict, text: str) -> bool:
        command, _, argument = text.partition(" ")
        argument = argument.strip()

        if command in {"/start", "/help"}:
            await self.client.send_message(
                chat_id,
                "<b>Gladiator</b>\n"
                "/status — runtime state\n"
                "/model [id] — show/change model\n"
                "/reasoning [off|minimal|low|medium|high|xhigh]\n"
                "/trace [off|milestones|verbose]\n"
                "/provider [endpoint] [api-key] — show/change OpenAI-compatible provider\n"
                "/compact — compact at the next safe boundary\n"
                "/new — start a clean agent session\n"
                "/todo — show the current task ledger\n"
                "/stop — stop the current run",
            )
            return True
        if command == "/status":
            await self.client.send_message(chat_id, self._status_html())
            return True
        if command == "/model":
            if argument:
                self.config.provider.model = argument
                save_config(self.config, self.config_path)
            await self.client.send_message(chat_id, f"Model: <code>{html.escape(self.config.provider.model)}</code>")
            return True
        if command == "/reasoning":
            allowed = {"off", "minimal", "low", "medium", "high", "xhigh"}
            if argument:
                if argument not in allowed:
                    await self.client.send_message(chat_id, "Invalid reasoning level.")
                    return True
                self.config.provider.reasoning_effort = argument  # type: ignore[assignment]
                save_config(self.config, self.config_path)
            await self.client.send_message(chat_id, f"Reasoning: <code>{self.config.provider.reasoning_effort}</code>")
            return True
        if command == "/trace":
            allowed = {"off", "milestones", "verbose"}
            if argument:
                if argument not in allowed:
                    await self.client.send_message(chat_id, "Invalid trace mode.")
                    return True
                self.config.runtime.trace_mode = argument  # type: ignore[assignment]
                save_config(self.config, self.config_path)
            await self.client.send_message(chat_id, f"Trace: <code>{self.config.runtime.trace_mode}</code>")
            return True
        if command == "/provider":
            if argument:
                parts = argument.split(maxsplit=1)
                self.config.provider.base_url = parts[0].rstrip("/")
                if len(parts) == 2:
                    self.config.provider.api_key = parts[1]
                    try:
                        await self.client.delete_message(chat_id, int(message["message_id"]))
                    except Exception:
                        pass
                save_config(self.config, self.config_path)
            await self.client.send_message(
                chat_id,
                f"Provider: <code>{html.escape(self.config.provider.base_url)}</code>\n"
                "API key is stored locally and never shown.",
            )
            return True
        if command == "/compact":
            await self.on_compact(chat_id)
            await self.client.send_message(chat_id, "Compaction requested; it will run at a safe agent boundary.")
            return True
        if command == "/stop":
            await self.on_stop(chat_id)
            await self.client.send_message(chat_id, "Stop requested.")
            return True
        return False

    async def ask_decision(self, chat_id: int, request: DecisionRequest) -> str | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        self._decision_waiters[chat_id] = (request, future)
        options = "\n".join(f"{idx}. <code>{html.escape(option)}</code>" for idx, option in enumerate(request.options, 1))
        await self.client.send_message(
            chat_id,
            "<b>Gladiator needs your choice</b>\n"
            f"{html.escape(request.question)}\n\n"
            f"Reason: {html.escape(request.reason)}\n\n"
            f"{options}\n\n"
            f"If you don't reply in time, I'll continue conservatively with: "
            f"<code>{html.escape(request.conservative_choice)}</code>",
        )
        try:
            return await future
        finally:
            current = self._decision_waiters.get(chat_id)
            if current and current[1] is future:
                self._decision_waiters.pop(chat_id, None)

    async def _handle_decision_reply(self, chat_id: int, text: str) -> None:
        request, future = self._decision_waiters[chat_id]
        choice: str | None = None
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(request.options):
                choice = request.options[idx]
        elif text in request.options:
            choice = text
        if choice is None:
            await self.client.send_message(chat_id, "Reply with an option number or the exact option text.")
            return
        if not future.done():
            future.set_result(choice)

    def _status_html(self) -> str:
        return (
            "<b>Gladiator</b>\n"
            f"Provider: <code>{html.escape(self.config.provider.base_url)}</code>\n"
            f"Model: <code>{html.escape(self.config.provider.model)}</code>\n"
            f"Reasoning: <code>{self.config.provider.reasoning_effort}</code>\n"
            f"Trace: <code>{self.config.runtime.trace_mode}</code>\n"
            "YOLO: <b>on</b>\n"
            f"Ask timeout: {self.config.runtime.escalation_timeout_seconds // 60} min\n"
            f"Compact target: {self.config.runtime.compact_threshold_tokens:,} tokens\n"
            f"Search: <code>{self.config.search.mode}</code>\n"
            f"Browser: {'enabled' if self.config.browser.enabled else 'not installed'}"
        )

    async def _build_task(self, message: dict, text: str) -> IncomingTask:
        message_id = int(message.get("message_id", 0))
        image_paths: list[Path] = []
        file_paths: list[Path] = []

        photos = message.get("photo") or []
        if photos:
            largest = photos[-1]
            file_id = str(largest["file_id"])
            path = self.inbox / f"telegram-{message_id}.jpg"
            image_paths.append(await self.client.download_file(file_id, path))

        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            name = Path(str(document.get("file_name") or f"telegram-{message_id}.bin")).name
            path = self.inbox / f"{message_id}-{name}"
            downloaded = await self.client.download_file(str(document["file_id"]), path)
            if str(document.get("mime_type", "")).startswith("image/"):
                image_paths.append(downloaded)
            else:
                file_paths.append(downloaded)

        if not text:
            text = "Please inspect the uploaded attachment(s) and help me with them."
        return IncomingTask(text=text, image_paths=image_paths, file_paths=file_paths)
