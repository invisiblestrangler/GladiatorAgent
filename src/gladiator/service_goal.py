from __future__ import annotations

import asyncio
import html

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment.mentor_handler = self._consult_mentor_from_agent_thread

    @staticmethod
    def _install_native_runtime_commands() -> None:
        ExtendedGladiatorService._install_native_runtime_commands()
        if not any(name == "mentor" for name, _description in TELEGRAM_COMMANDS):
            TELEGRAM_COMMANDS.insert(4, ("mentor", "Configure rare advisory mentor"))

    async def run_forever(self) -> None:
        provider = self.config.provider
        if provider.context_window is None or provider.max_context_window is None:
            await self._refresh_model_context(clear_first=False)
        await super().run_forever()

    def _install_bot_extensions(self) -> None:
        super()._install_bot_extensions()
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            stripped = text.strip()
            lowered = stripped.lower()
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
            task = asyncio.create_task(
                self.handle_task(
                    chat_id,
                    IncomingTask(
                        text=(
                            "Start executing the active session goal now. Work autonomously on the next concrete "
                            "unfinished action or TODO. Do not merely restate the goal or previous report; perform "
                            "the work, verify it, and continue toward completion."
                        )
                    ),
                )
            )
            self.bot._spawned.add(task)
            task.add_done_callback(self.bot._spawned.discard)
            return True

        self.bot._handle_command = command_handler  # type: ignore[method-assign]

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
