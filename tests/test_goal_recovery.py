from __future__ import annotations

import asyncio
import json
import os
import time
from threading import Event

import pytest

from gladiator.goal import GoalManager
from gladiator.service_goal import GoalAwareGladiatorService


@pytest.mark.asyncio
async def test_interrupted_running_goal_is_recovered_on_startup(tmp_path):
    service = object.__new__(GoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.goal_manager.set("Finish the current integration")

    (tmp_path / "active-task.json").write_text(
        json.dumps(
            {
                "status": "running",
                "chat_id": 123,
                "pid": 111,
                "started_at": time.time() - 90,
                "updated_at": time.time() - 60,
                "recovery_count": 0,
            }
        ),
        encoding="utf-8",
    )

    notices: list[tuple[int, str]] = []
    spawned: list[tuple[int, int, bool]] = []

    async def fake_try_send(chat_id: int, text: str) -> bool:
        notices.append((chat_id, text))
        return True

    def fake_spawn(chat_id: int, *, recovery_count: int, recovered: bool):
        spawned.append((chat_id, recovery_count, recovered))
        return None

    service._try_send_html = fake_try_send
    service._spawn_supervised_goal_task = fake_spawn

    await service._recover_interrupted_goal_execution()

    assert spawned == [(123, 1, True)]
    assert any("resuming the active goal" in text for _chat, text in notices)
    state = service._load_goal_task_state()
    assert state is not None
    assert state["status"] == "interrupted"


@pytest.mark.asyncio
async def test_repeated_interruptions_stop_automatic_recovery(tmp_path):
    service = object.__new__(GoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.goal_manager.set("Finish the current integration")
    service.MAX_AUTOMATIC_GOAL_RECOVERIES = 2

    (tmp_path / "active-task.json").write_text(
        json.dumps(
            {
                "status": "running",
                "chat_id": 123,
                "pid": 111,
                "started_at": time.time() - 90,
                "updated_at": time.time() - 60,
                "recovery_count": 2,
            }
        ),
        encoding="utf-8",
    )

    notices: list[str] = []
    spawned = []

    async def fake_try_send(_chat_id: int, text: str) -> bool:
        notices.append(text)
        return True

    def fake_spawn(*_args, **_kwargs):
        spawned.append(True)

    service._try_send_html = fake_try_send
    service._spawn_supervised_goal_task = fake_spawn

    await service._recover_interrupted_goal_execution()

    assert spawned == []
    assert any("Automatic goal recovery stopped" in text for text in notices)


def test_task_status_detects_stale_running_record(tmp_path):
    service = object.__new__(GoalAwareGladiatorService)
    service.state_dir = tmp_path
    service._run_lock = asyncio.Lock()
    (tmp_path / "active-task.json").write_text(
        json.dumps(
            {
                "status": "running",
                "chat_id": 123,
                "pid": os.getpid() - 1,
                "started_at": time.time() - 30,
                "updated_at": time.time() - 20,
                "recovery_count": 0,
                "detail": "goal start",
            }
        ),
        encoding="utf-8",
    )

    rendered = service._task_status_html()

    assert "Runtime task lock: <b>idle</b>" in rendered
    assert "Persisted goal task: <b>running</b>" in rendered
    assert "recovery is required" in rendered
