from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from gladiator.service_resilient import ResilientGoalAwareGladiatorService


class _FlakyPollingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get_updates(self, *, offset=None, timeout=30):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError(
                "temporary Telegram connection failure",
                request=httpx.Request("POST", "https://api.telegram.org/botTEST/getUpdates"),
            )
        return [{"update_id": 11}]


@pytest.mark.asyncio
async def test_poll_transport_failure_does_not_escape_or_restart_runtime(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.TELEGRAM_POLL_RETRY_BASE_SECONDS = 0.0
    service.TELEGRAM_POLL_RETRY_MAX_SECONDS = 0.0
    client = _FlakyPollingClient()

    async def original_update(_update):
        return None

    service.bot = SimpleNamespace(
        client=client,
        _handle_update=original_update,
        _authorized=lambda _chat_id: True,
    )
    failures = []
    recoveries = []
    service._record_runtime_boundary_failure = lambda label, exc: failures.append((label, exc))
    service._record_runtime_recovery = recoveries.append

    async def fake_try_send(_chat_id: int, _text: str) -> bool:
        return True

    service._try_send_html = fake_try_send
    service._install_telegram_runtime_guards()

    updates = await service.bot.client.get_updates(offset=3, timeout=30)

    assert updates == [{"update_id": 11}]
    assert client.calls == 2
    assert len(failures) == 1
    assert "keeping Gladiator alive" in failures[0][0]
    assert recoveries and "recovered after 1 failure" in recoveries[0]


@pytest.mark.asyncio
async def test_update_handler_failure_is_isolated_from_poll_loop(tmp_path):
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.state_dir = tmp_path
    service.TELEGRAM_POLL_RETRY_BASE_SECONDS = 0.0
    service.TELEGRAM_POLL_RETRY_MAX_SECONDS = 0.0

    async def get_updates(*, offset=None, timeout=30):
        return []

    async def failing_update(_update):
        raise httpx.ConnectError(
            "sendMessage connection failed",
            request=httpx.Request("POST", "https://api.telegram.org/botTEST/sendMessage"),
        )

    service.bot = SimpleNamespace(
        client=SimpleNamespace(get_updates=get_updates),
        _handle_update=failing_update,
        _authorized=lambda chat_id: chat_id == 7,
    )
    failures = []
    notices = []
    service._record_runtime_boundary_failure = lambda label, exc: failures.append((label, exc))
    service._record_runtime_recovery = lambda _message: None

    async def fake_try_send(chat_id: int, text: str) -> bool:
        notices.append((chat_id, text))
        return True

    service._try_send_html = fake_try_send
    service._install_telegram_runtime_guards()

    await service.bot._handle_update({"update_id": 20, "message": {"chat": {"id": 7, "type": "private"}}})

    assert len(failures) == 1
    assert "isolated the update" in failures[0][0]
    assert notices and notices[0][0] == 7
    assert "stayed running" in notices[0][1]


def test_poll_retry_delay_honors_retry_after_and_caps_backoff():
    service = object.__new__(ResilientGoalAwareGladiatorService)
    service.TELEGRAM_POLL_RETRY_BASE_SECONDS = 1.0
    service.TELEGRAM_POLL_RETRY_MAX_SECONDS = 30.0
    request = httpx.Request("POST", "https://api.telegram.org/botTEST/getUpdates")
    response = httpx.Response(429, headers={"retry-after": "7"}, request=request)
    error = httpx.HTTPStatusError("rate limited", request=request, response=response)

    assert service._telegram_poll_retry_delay(error, 1) == 7.0
    assert service._telegram_poll_retry_delay(httpx.ConnectError("x", request=request), 20) == 30.0
