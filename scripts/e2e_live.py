from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx

from gladiator.config import (
    GladiatorConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
    TelegramConfig,
    save_config,
)
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.telegram.bot import IncomingTask

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SENTINEL = "GLADIATOR_E2E_FILE_OK"
FINAL_SENTINEL = "GLADIATOR_E2E_AGENT_OK"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required E2E environment variable is missing: {name}")
    return value


def retryable_provider_error(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return exc.response.status_code in {429, 500, 502, 503, 504}


async def run_once(*, api_key: str, bot_token: str, user_id: int, attempt: int) -> None:
    workspace = Path(tempfile.mkdtemp(prefix=f"gladiator-e2e-{attempt}-"))
    config_path = workspace / ".gladiator" / "e2e-config.json"
    config = GladiatorConfig(
        provider=ProviderConfig(
            name="e2e",
            base_url=os.environ.get("E2E_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            model=os.environ.get("E2E_MODEL", DEFAULT_MODEL),
            reasoning_effort="low",
            context_window=1_000_000,
        ),
        telegram=TelegramConfig(bot_token=bot_token, allowed_user_ids=[user_id]),
        search=SearchConfig(mode="none"),
        runtime=RuntimeConfig(trace_mode="milestones", compact_threshold_tokens=300_000),
    )
    save_config(config, config_path)

    service = ExtendedGladiatorService(config=config, config_path=config_path, workspace=workspace)
    service._loop = asyncio.get_running_loop()
    final_outputs: list[str] = []
    original_send_markdown = service._send_markdown

    async def capture_and_send_markdown(chat_id: int, text: str) -> None:
        final_outputs.append(text)
        await original_send_markdown(chat_id, text)

    service._send_markdown = capture_and_send_markdown  # type: ignore[method-assign]

    try:
        identity = await service.bot.client._call("getMe")
        if not isinstance(identity, dict) or not identity.get("id"):
            raise RuntimeError("Telegram getMe did not return a bot identity")

        await service.bot.client.send_message(
            user_id,
            f"<b>Gladiator E2E starting</b>\nAttempt {attempt}. Running real model + bash + TODO + session-reset checks.",
        )

        task = IncomingTask(
            text=(
                "Automated Gladiator E2E test. Work autonomously and use bash. "
                "First add one TODO named 'E2E sentinel workflow' using `gladiator todo add`. "
                f"Then create a file named e2e_probe.txt containing exactly {SENTINEL} followed by a newline. "
                "Read the file back and verify its exact contents. Then mark TODO #1 done with `gladiator todo done 1`. "
                f"Finally complete the task with final output exactly `{FINAL_SENTINEL}`."
            )
        )
        await service.handle_task(user_id, task)

        if not final_outputs or final_outputs[-1].strip() != FINAL_SENTINEL:
            raise RuntimeError("Agent final submission did not match the expected E2E sentinel")
        probe = workspace / "e2e_probe.txt"
        if not probe.exists() or probe.read_text(encoding="utf-8") != SENTINEL + "\n":
            raise RuntimeError("Agent tool loop did not create the expected sentinel file")
        if service.todo_manager.open_count != 0:
            raise RuntimeError("TODO ledger still has open items after the agent completed the E2E task")
        trajectory = service.state_dir / "trajectory.json"
        if not trajectory.exists():
            raise RuntimeError("Agent trajectory was not persisted")

        old_session_id = service.session.session_id
        await service.handle_new_session(user_id)
        if service.session.session_id == old_session_id:
            raise RuntimeError("/new did not rotate the local session ID")
        if service.agent.messages:
            raise RuntimeError("/new did not clear conversational model history")
        archived = service.state_dir / "sessions" / f"{old_session_id}.trajectory.json"
        if not archived.exists():
            raise RuntimeError("/new did not archive the previous trajectory")

        await service.bot.client.send_message(
            user_id,
            "<b>Gladiator E2E PASS</b>\nTelegram ✓\nModel stream ✓\nFinal submission ✓\nBash tool loop ✓\nTODO ledger ✓\n/new archive + reset ✓",
        )
    finally:
        await service.bot.client.close()


async def main() -> None:
    api_key = required_env("E2E_API_KEY")
    bot_token = required_env("GLADIATOR_TG_BOT")
    user_id = int(required_env("TG_USER_ID"))

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            await run_once(api_key=api_key, bot_token=bot_token, user_id=user_id, attempt=attempt)
            print("GLADIATOR_LIVE_E2E_PASS")
            return
        except BaseException as exc:
            last_error = exc
            if not retryable_provider_error(exc) or attempt == 3:
                raise
            print(f"Retryable provider response on attempt {attempt}; retrying live E2E.")
            await asyncio.sleep(5 * attempt)

    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    asyncio.run(main())
