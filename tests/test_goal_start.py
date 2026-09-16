from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from gladiator.goal import GoalManager
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.service_goal import GoalAwareGladiatorService


class _FakeClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **_kwargs):
        self.messages.append((chat_id, text))
        return {"message_id": len(self.messages)}


@pytest.mark.asyncio
async def test_goal_start_launches_active_goal_immediately(tmp_path, monkeypatch):
    service = object.__new__(GoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.goal_manager.set("Finish the remaining refund work")
    client = _FakeClient()

    async def original_handler(_chat_id: int, _message: dict, _text: str) -> bool:
        return False

    service.bot = SimpleNamespace(_handle_command=original_handler, client=client, _spawned=set())
    started = asyncio.Event()
    received = []

    async def fake_handle_task(chat_id, incoming):
        received.append((chat_id, incoming))
        started.set()

    service.handle_task = fake_handle_task
    monkeypatch.setattr(ExtendedGladiatorService, "_install_bot_extensions", lambda _self: None)
    service._install_bot_extensions()

    handled = await service.bot._handle_command(7, {}, "/goal start")
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await asyncio.gather(*tuple(service.bot._spawned))

    assert handled is True
    assert received[0][0] == 7
    assert "Start executing the active session goal now" in received[0][1].text
    assert "Do not merely restate" in received[0][1].text
    assert any("Starting the active session goal now" in text for _chat, text in client.messages)
    state = service._load_goal_task_state()
    assert state is not None
    assert state["status"] == "paused"


@pytest.mark.asyncio
async def test_goal_start_without_active_goal_does_not_launch_task(tmp_path, monkeypatch):
    service = object.__new__(GoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    client = _FakeClient()

    async def original_handler(_chat_id: int, _message: dict, _text: str) -> bool:
        return False

    service.bot = SimpleNamespace(_handle_command=original_handler, client=client, _spawned=set())
    received = []

    async def fake_handle_task(chat_id, incoming):
        received.append((chat_id, incoming))

    service.handle_task = fake_handle_task
    monkeypatch.setattr(ExtendedGladiatorService, "_install_bot_extensions", lambda _self: None)
    service._install_bot_extensions()

    handled = await service.bot._handle_command(7, {}, "/goal start")

    assert handled is True
    assert received == []
    assert any("No active session goal" in text for _chat, text in client.messages)
