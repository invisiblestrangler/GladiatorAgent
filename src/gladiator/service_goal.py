from __future__ import annotations

import asyncio

from gladiator.service_ext import ExtendedGladiatorService
from gladiator.telegram.bot import IncomingTask


class GoalAwareGladiatorService(ExtendedGladiatorService):
    """Extended service with an immediate Telegram `/goal start` action."""

    def _install_bot_extensions(self) -> None:
        super()._install_bot_extensions()
        original = self.bot._handle_command

        async def command_handler(chat_id: int, message: dict, text: str) -> bool:
            if text.strip().lower() != "/goal start":
                return await original(chat_id, message, text)

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
