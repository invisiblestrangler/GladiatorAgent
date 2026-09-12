import asyncio

import pytest

from gladiator.runtime.decision import DecisionBroker, DecisionRequest


@pytest.mark.asyncio
async def test_uses_valid_user_choice():
    async def ask(_request):
        return "b"

    broker = DecisionBroker(ask, timeout_seconds=1)
    result = await broker.resolve(DecisionRequest("Choose", ("a", "b"), "a", "material ambiguity"))
    assert result.choice == "b"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_timeout_resumes_conservatively():
    async def ask(_request):
        await asyncio.sleep(0.1)
        return "b"

    broker = DecisionBroker(ask, timeout_seconds=0.01)
    result = await broker.resolve(DecisionRequest("Choose", ("a", "b"), "a", "material ambiguity"))
    assert result.choice == "a"
    assert result.timed_out is True
