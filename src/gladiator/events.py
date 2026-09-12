from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


class EventKind(StrEnum):
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    RESPONSE_FINISHED = "response_finished"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_FINISHED = "compaction_finished"
    STATUS = "status"
    WARNING = "warning"
    ARTIFACT_READY = "artifact_ready"


@dataclass(slots=True)
class AgentEvent:
    kind: EventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventSink = Callable[[AgentEvent], None]


def null_event_sink(_event: AgentEvent) -> None:
    return None
