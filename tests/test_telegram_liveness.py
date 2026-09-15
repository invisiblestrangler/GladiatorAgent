import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from gladiator.events import AgentEvent, EventKind
from gladiator.service import GladiatorService
from gladiator.telegram.bot import IncomingTask


class _DeliveryFailureClient:
    def __init__(self):
        self.send_calls = 0
        self.typing_calls = 0
        self.edits: list[str] = []
        self.markup_calls = 0

    async def send_message(self, _chat_id, text, **_kwargs):
        self.send_calls += 1
        if self.send_calls == 1:
            return {"message_id": 99}
        raise RuntimeError("simulated Telegram send failure")

    async def send_chat_action(self, _chat_id, _action):
        self.typing_calls += 1
        return True

    async def edit_message_text(self, _chat_id, _message_id, text, **_kwargs):
        await asyncio.sleep(0.03)
        self.edits.append(text)
        return True

    async def edit_message_reply_markup(self, _chat_id, _message_id, _reply_markup=None):
        self.markup_calls += 1
        return True


@pytest.mark.asyncio
async def test_final_delivery_failure_is_persisted_and_progress_is_finalized(tmp_path):
    service = object.__new__(GladiatorService)
    service.state_dir = tmp_path / ".gladiator"
    service.state_dir.mkdir()
    service._run_lock = asyncio.Lock()
    service._current_chat_id = None
    service.cancel_event = Event()
    service._events = asyncio.Queue()
    service.environment = SimpleNamespace(skill_write_authorized=False)
    service.config = SimpleNamespace(runtime=SimpleNamespace(trace_mode="milestones"))
    service._refresh_model_settings = lambda: None
    service._drain_event_queue = lambda: None
    service.TYPING_HEARTBEAT_SECONDS = 0.005
    service.TYPING_ACTION_TIMEOUT_SECONDS = 0.05
    service.FINAL_UI_RETRY_DELAYS = (0.0,)

    def run_task(*_args, **_kwargs):
        return {"exit_status": "Submitted", "submission": "important final answer"}

    service.agent = SimpleNamespace(run_task=run_task)
    client = _DeliveryFailureClient()
    service.bot = SimpleNamespace(client=client)

    async def consume_events(_chat_id, _message_id, finished):
        await finished.wait()
        return ["✓ Work itself completed"]

    service._consume_events = consume_events

    await service.handle_task(123, IncomingTask(text="do it"))

    pending = list((service.state_dir / "pending-delivery").glob("*.md"))
    assert len(pending) == 1
    assert pending[0].read_text(encoding="utf-8").strip() == "important final answer"
    assert any("Delivery failed" in edit for edit in client.edits)
    assert client.markup_calls == 1
    assert client.typing_calls >= 2
    assert service._current_chat_id is None


@pytest.mark.asyncio
async def test_pending_delivery_is_replayed_and_removed(tmp_path):
    service = object.__new__(GladiatorService)
    service.state_dir = tmp_path / ".gladiator"
    pending_dir = service.state_dir / "pending-delivery"
    pending_dir.mkdir(parents=True)
    pending = pending_dir / "1.md"
    pending.write_text("previous final answer\n", encoding="utf-8")
    sent: list[str] = []

    class Client:
        async def send_message(self, _chat_id, text, **_kwargs):
            sent.append(text)
            return {"message_id": len(sent)}

    service.bot = SimpleNamespace(client=Client())

    await service._flush_pending_deliveries(123)

    assert not pending.exists()
    assert any("Recovered undelivered output" in item for item in sent)
    assert any("previous final answer" in item for item in sent)


@pytest.mark.asyncio
async def test_typing_heartbeat_recovers_after_transient_failure(tmp_path):
    service = object.__new__(GladiatorService)
    service.state_dir = tmp_path / ".gladiator"
    service.state_dir.mkdir()
    service.TYPING_HEARTBEAT_SECONDS = 0.005
    service.TYPING_ACTION_TIMEOUT_SECONDS = 0.05
    finished = asyncio.Event()

    class Client:
        calls = 0

        async def send_chat_action(self, _chat_id, _action):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            finished.set()
            return True

    client = Client()
    service.bot = SimpleNamespace(client=client)

    await service._typing_heartbeat(123, finished)

    assert client.calls == 2
    errors = (service.state_dir / "runtime-errors.log").read_text(encoding="utf-8")
    assert "typing heartbeat failed" in errors


@pytest.mark.asyncio
async def test_artifact_delivery_failure_does_not_kill_progress_consumer(tmp_path):
    service = object.__new__(GladiatorService)
    service.state_dir = tmp_path / ".gladiator"
    service.state_dir.mkdir()
    service._events = asyncio.Queue()
    service.config = SimpleNamespace(runtime=SimpleNamespace(trace_mode="milestones"))
    service.PROGRESS_LIVENESS_SECONDS = 999
    finished = asyncio.Event()

    class Client:
        async def edit_message_text(self, *_args, **_kwargs):
            return True

    service.bot = SimpleNamespace(client=Client())

    async def fail_artifact(*_args, **_kwargs):
        raise RuntimeError("upload failed")

    service._send_artifact = fail_artifact
    await service._events.put(
        AgentEvent(EventKind.ARTIFACT_READY, data={"path": str(tmp_path / "result.png"), "is_image": True})
    )
    finished.set()

    summary = await service._consume_events(123, 99, finished)

    assert any("Could not send result.png" in item for item in summary)
    errors = (service.state_dir / "runtime-errors.log").read_text(encoding="utf-8")
    assert "artifact delivery failed" in errors
