from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    question: str
    options: tuple[str, ...]
    conservative_choice: str
    reason: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.conservative_choice not in self.options:
            raise ValueError("conservative_choice must be one of options")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    choice: str
    timed_out: bool


AskUser = Callable[[DecisionRequest], Awaitable[str | None]]


class DecisionBroker:
    """Exceptional user-decision gate for an otherwise YOLO agent.

    Gladiator should only call this when the agent is genuinely unable to choose a
    safe/reasonable path. If the user is unavailable, execution resumes after the
    timeout using the caller-supplied conservative choice.
    """

    def __init__(self, ask_user: AskUser, timeout_seconds: float = 3600):
        self.ask_user = ask_user
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    async def resolve(self, request: DecisionRequest) -> DecisionResult:
        try:
            answer = await asyncio.wait_for(self.ask_user(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return DecisionResult(choice=request.conservative_choice, timed_out=True)

        if answer in request.options:
            return DecisionResult(choice=answer, timed_out=False)
        return DecisionResult(choice=request.conservative_choice, timed_out=False)
