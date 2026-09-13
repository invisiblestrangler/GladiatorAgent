from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gladiator.config import GladiatorConfig, ProviderConfig, RuntimeConfig, TelegramConfig
from gladiator.telegram.bot import IncomingTask, TelegramBotRuntime


def _config(*, debounce: float = 0.03, max_burst: float = 0.12) -> GladiatorConfig:
    return GladiatorConfig(
        provider=ProviderConfig(base_url="https://example.invalid/v1", api_key="test-key", model="test-model"),
        telegram=TelegramConfig(bot_token="test-token", allowed_user_ids=[7]),
        runtime=RuntimeConfig(
            telegram_input_debounce_seconds=debounce,
            telegram_input_max_burst_seconds=max_burst,
        ),
    )


async def _noop(_chat_id: int) -> None:
    return None


async def _close(runtime: TelegramBotRuntime) -> None:
    for burst in runtime._pending_bursts.values():
        if burst.flush_task is not None:
            burst.flush_task.cancel()
    for task in list(runtime._spawned):
        task.cancel()
    await asyncio.sleep(0)
    await runtime.client.close()


@pytest.mark.asyncio
async def test_rapid_text_messages_become_one_agent_turn(tmp_path: Path):
    received: list[IncomingTask] = []

    async def on_task(_chat_id: int, incoming: IncomingTask) -> None:
        received.append(incoming)

    runtime = TelegramBotRuntime(
        config=_config(),
        config_path=tmp_path / "config.json",
        workspace=tmp_path,
        on_task=on_task,
        on_stop=_noop,
        on_compact=_noop,
    )
    try:
        runtime._queue_incoming_task(7, IncomingTask(text="first chunk"))
        await asyncio.sleep(0.005)
        runtime._queue_incoming_task(7, IncomingTask(text="second chunk"))
        await asyncio.sleep(0.07)

        assert len(received) == 1
        assert received[0].text == "first chunk\n\nsecond chunk"
        assert received[0].source_message_count == 2
    finally:
        await _close(runtime)


@pytest.mark.asyncio
async def test_messages_outside_debounce_window_stay_separate(tmp_path: Path):
    received: list[IncomingTask] = []

    async def on_task(_chat_id: int, incoming: IncomingTask) -> None:
        received.append(incoming)

    runtime = TelegramBotRuntime(
        config=_config(debounce=0.02, max_burst=0.08),
        config_path=tmp_path / "config.json",
        workspace=tmp_path,
        on_task=on_task,
        on_stop=_noop,
        on_compact=_noop,
    )
    try:
        runtime._queue_incoming_task(7, IncomingTask(text="first request"))
        await asyncio.sleep(0.05)
        runtime._queue_incoming_task(7, IncomingTask(text="second request"))
        await asyncio.sleep(0.05)

        assert [item.text for item in received] == ["first request", "second request"]
        assert [item.source_message_count for item in received] == [1, 1]
    finally:
        await _close(runtime)


@pytest.mark.asyncio
async def test_text_image_text_burst_keeps_text_order_and_attachment(tmp_path: Path):
    received: list[IncomingTask] = []

    async def on_task(_chat_id: int, incoming: IncomingTask) -> None:
        received.append(incoming)

    image = tmp_path / "comparison.png"
    runtime = TelegramBotRuntime(
        config=_config(),
        config_path=tmp_path / "config.json",
        workspace=tmp_path,
        on_task=on_task,
        on_stop=_noop,
        on_compact=_noop,
    )
    try:
        runtime._queue_incoming_task(7, IncomingTask(text="describe this"))
        await asyncio.sleep(0.004)
        runtime._queue_incoming_task(7, IncomingTask(text="", image_paths=[image]))
        await asyncio.sleep(0.004)
        runtime._queue_incoming_task(7, IncomingTask(text="and crop it square"))
        await asyncio.sleep(0.07)

        assert len(received) == 1
        assert received[0].text == "describe this\n\nand crop it square"
        assert received[0].image_paths == [image]
        assert received[0].source_message_count == 3
    finally:
        await _close(runtime)


@pytest.mark.asyncio
async def test_rapid_followup_during_busy_task_becomes_one_queued_turn(tmp_path: Path):
    received: list[IncomingTask] = []
    run_lock = asyncio.Lock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def on_task(_chat_id: int, incoming: IncomingTask) -> None:
        async with run_lock:
            received.append(incoming)
            if len(received) == 1:
                first_started.set()
                await release_first.wait()

    runtime = TelegramBotRuntime(
        config=_config(),
        config_path=tmp_path / "config.json",
        workspace=tmp_path,
        on_task=on_task,
        on_stop=_noop,
        on_compact=_noop,
    )
    try:
        runtime._queue_incoming_task(7, IncomingTask(text="long running request"))
        await asyncio.wait_for(first_started.wait(), timeout=0.2)

        runtime._queue_incoming_task(7, IncomingTask(text="also check edge cases"))
        await asyncio.sleep(0.005)
        runtime._queue_incoming_task(7, IncomingTask(text="especially retries"))
        await asyncio.sleep(0.06)
        assert len(received) == 1

        release_first.set()
        await asyncio.sleep(0.06)
        assert len(received) == 2
        assert received[1].text == "also check edge cases\n\nespecially retries"
        assert received[1].source_message_count == 2
    finally:
        release_first.set()
        await _close(runtime)


@pytest.mark.asyncio
async def test_stop_command_discards_unflushed_burst_and_bypasses_coalescing(tmp_path: Path, monkeypatch):
    received: list[IncomingTask] = []
    stopped: list[int] = []

    async def on_task(_chat_id: int, incoming: IncomingTask) -> None:
        received.append(incoming)

    async def on_stop(chat_id: int) -> None:
        stopped.append(chat_id)

    runtime = TelegramBotRuntime(
        config=_config(debounce=0.05, max_burst=0.2),
        config_path=tmp_path / "config.json",
        workspace=tmp_path,
        on_task=on_task,
        on_stop=on_stop,
        on_compact=_noop,
    )

    async def fake_send_message(*_args, **_kwargs):
        return {"message_id": 1}

    monkeypatch.setattr(runtime.client, "send_message", fake_send_message)
    try:
        runtime._queue_incoming_task(7, IncomingTask(text="this should not run"))
        await runtime._handle_update(
            {
                "update_id": 2,
                "message": {
                    "message_id": 11,
                    "chat": {"id": 7, "type": "private"},
                    "text": "/stop",
                },
            }
        )
        await asyncio.sleep(0.08)

        assert stopped == [7]
        assert received == []
    finally:
        await _close(runtime)
