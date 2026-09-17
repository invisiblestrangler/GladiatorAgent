from __future__ import annotations

import asyncio
import json
import time
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

from gladiator.service_goal import GoalAwareGladiatorService
from gladiator.service_resilient import ResilientGoalAwareGladiatorService
from gladiator.telegram.bot import IncomingTask


@pytest.mark.asyncio
async def test_interrupted_ordinary_task_is_recovered_on_startup(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.MAX_AUTOMATIC_MESSAGE_RECOVERIES = 2

    (tmp_path / "active-message-task.json").write_text(
        json.dumps(
            {
                "status": "running",
                "chat_id": 123,
                "pid": 111,
                "started_at": time.time() - 90,
                "updated_at": time.time() - 60,
                "recovery_count": 0,
                "task_text": "Fix the failing deployment and verify it",
                "image_paths": [],
                "file_paths": [],
                "source_message_count": 1,
            }
        ),
        encoding="utf-8",
    )

    notices: list[tuple[int, str]] = []
    spawned: list[tuple[int, IncomingTask, int]] = []

    async def fake_try_send(chat_id: int, text: str) -> bool:
        notices.append((chat_id, text))
        return True

    def fake_spawn(chat_id: int, incoming: IncomingTask, *, recovery_count: int):
        spawned.append((chat_id, incoming, recovery_count))
        return None

    service._try_send_html = fake_try_send
    service._spawn_supervised_message_task = fake_spawn

    await service._recover_interrupted_message_execution()

    assert len(spawned) == 1
    assert spawned[0][0] == 123
    assert spawned[0][1].text == "Fix the failing deployment and verify it"
    assert spawned[0][2] == 1
    assert any("checking existing work first" in text for _chat, text in notices)
    state = service._load_message_task_state()
    assert state is not None
    assert state["status"] == "interrupted"
    assert state["task_text"] == "Fix the failing deployment and verify it"


def test_message_recovery_prompt_avoids_blind_side_effect_replay():
    prompt = ResilientGoalAwareGladiatorService._message_recovery_prompt("Deploy the service")

    assert "Original user request:\nDeploy the service" in prompt
    assert "inspect" in prompt
    assert "Do not blindly repeat" in prompt
    assert "side effects" in prompt


@pytest.mark.asyncio
async def test_repeated_ordinary_task_interruptions_stop_automatic_recovery(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service.MAX_AUTOMATIC_MESSAGE_RECOVERIES = 2

    (tmp_path / "active-message-task.json").write_text(
        json.dumps(
            {
                "status": "running",
                "chat_id": 123,
                "pid": 111,
                "started_at": time.time() - 90,
                "updated_at": time.time() - 60,
                "recovery_count": 2,
                "task_text": "Continue the migration",
                "image_paths": [],
                "file_paths": [],
                "source_message_count": 1,
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
    service._spawn_supervised_message_task = fake_spawn

    await service._recover_interrupted_message_execution()

    assert spawned == []
    assert any("Automatic task recovery stopped" in text for text in notices)
    state = service._load_message_task_state()
    assert state is not None
    assert state["status"] == "recovery_stopped"


class _RecordingClient:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **_kwargs):
        self.messages.append((chat_id, text))
        return {"message_id": len(self.messages)}


def _bare_runtime_service(tmp_path) -> ResilientGoalAwareGladiatorService:
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.cancel_event = Event()
    service._run_lock = asyncio.Lock()
    service._message_dispatch_lock = asyncio.Lock()
    service.bot = SimpleNamespace(client=_RecordingClient())
    return service


@pytest.mark.asyncio
async def test_queued_ordinary_task_does_not_replace_active_recovery_marker(tmp_path, monkeypatch):
    service = _bare_runtime_service(tmp_path)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_parent_handle_task(self, _chat_id: int, incoming: IncomingTask) -> None:
        async with self._run_lock:
            if incoming.text == "first task":
                first_started.set()
                await release_first.wait()

    monkeypatch.setattr(GoalAwareGladiatorService, "handle_task", fake_parent_handle_task)

    first = asyncio.create_task(service.handle_task(123, IncomingTask(text="first task")))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    state = service._load_message_task_state()
    assert state is not None
    assert state["status"] == "running"
    assert state["task_text"] == "first task"

    second = asyncio.create_task(service.handle_task(123, IncomingTask(text="second task")))
    await asyncio.sleep(0.02)
    queued_state = service._load_message_task_state()
    assert queued_state is not None
    assert queued_state["task_text"] == "first task"
    assert any("Queued behind" in text for _chat, text in service.bot.client.messages)

    release_first.set()
    await asyncio.gather(first, second)
    final_state = service._load_message_task_state()
    assert final_state is not None
    assert final_state["status"] == "finished"
    assert final_state["task_text"] == "second task"


@pytest.mark.asyncio
async def test_ordinary_marker_waits_until_shared_agent_lock_is_free(tmp_path, monkeypatch):
    service = _bare_runtime_service(tmp_path)

    async def fake_parent_handle_task(self, _chat_id: int, _incoming: IncomingTask) -> None:
        async with self._run_lock:
            return

    monkeypatch.setattr(GoalAwareGladiatorService, "handle_task", fake_parent_handle_task)
    await service._run_lock.acquire()
    pending = asyncio.create_task(service.handle_task(123, IncomingTask(text="queued task")))
    await asyncio.sleep(0.02)

    assert not service._message_task_state_path.exists()
    assert any("Queued behind" in text for _chat, text in service.bot.client.messages)

    service._run_lock.release()
    await asyncio.wait_for(pending, timeout=1)
    state = service._load_message_task_state()
    assert state is not None
    assert state["status"] == "finished"
    assert state["task_text"] == "queued task"


def _telegram_http_error(description: str, *, credential: str = "SUPERSECRET") -> httpx.HTTPStatusError:
    request = httpx.Request(
        "POST",
        f"https://api.telegram.org/bot{credential}/editMessageReplyMarkup",
    )
    response = httpx.Response(
        400,
        request=request,
        json={"ok": False, "error_code": 400, "description": description},
    )
    return httpx.HTTPStatusError("Telegram request failed", request=request, response=response)


def test_runtime_log_suppresses_cosmetic_telegram_noop_from_response_body(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path

    service._record_runtime_failure(
        "final Telegram Stop-button removal failed",
        _telegram_http_error("Bad Request: message is not modified"),
    )

    assert not (tmp_path / "runtime-errors.log").exists()


def test_runtime_log_redacts_telegram_credential_but_keeps_actionable_error(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    credential = "SUPERSECRET"
    failure = _telegram_http_error("Bad Request: chat not found", credential=credential)

    service._record_runtime_boundary_failure("Telegram polling failed", failure)

    logged = (tmp_path / "runtime-errors.log").read_text(encoding="utf-8")
    assert credential not in logged
    assert "Bad Request: chat not found" in logged
    assert "Telegram HTTP 400" in logged


def test_non_telegram_provider_failure_is_logged_verbatim(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path

    service._record_runtime_failure(
        "agent task failed",
        RuntimeError("Provider HTTP 400: upstream error"),
    )

    logged = (tmp_path / "runtime-errors.log").read_text(encoding="utf-8")
    assert "agent task failed" in logged
    assert "Provider HTTP 400: upstream error" in logged
