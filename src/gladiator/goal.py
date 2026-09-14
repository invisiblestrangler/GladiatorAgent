from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GoalStatus = Literal["active", "achieved"]


@dataclass(slots=True)
class GoalState:
    text: str
    status: GoalStatus = "active"
    reason: str = ""
    updated_at: float = 0.0
    assessments: int = 0


class GoalManager:
    """Small session-scoped goal record kept outside ordinary model context."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> GoalState | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        text = str(raw.get("text") or "").strip()
        if not text:
            return None
        status = str(raw.get("status") or "active")
        if status not in {"active", "achieved"}:
            status = "active"
        try:
            updated_at = float(raw.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            updated_at = 0.0
        try:
            assessments = max(0, int(raw.get("assessments") or 0))
        except (TypeError, ValueError):
            assessments = 0
        return GoalState(
            text=text,
            status=status,  # type: ignore[arg-type]
            reason=str(raw.get("reason") or "").strip(),
            updated_at=updated_at,
            assessments=assessments,
        )

    def _save(self, state: GoalState) -> GoalState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = time.time()
        payload = {
            "text": state.text,
            "status": state.status,
            "reason": state.reason,
            "updated_at": state.updated_at,
            "assessments": state.assessments,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return state

    def set(self, text: str, *, reason: str = "") -> GoalState:
        text = text.strip()
        if not text:
            raise ValueError("Goal text must not be empty")
        return self._save(GoalState(text=text, status="active", reason=reason.strip()))

    def assess(self, status: GoalStatus, *, reason: str = "") -> GoalState:
        state = self.load()
        if state is None:
            raise ValueError("No active goal exists")
        state.status = status
        state.reason = reason.strip()
        state.assessments += 1
        if status == "achieved":
            # A completed goal should stop occupying the session's active-goal slot.
            # Return the completed state to the caller for reporting, then remove the
            # persistent goal record so the next turn starts with no active goal.
            state.updated_at = time.time()
            self.clear()
            return state
        return self._save(state)

    def reopen(self, *, reason: str = "") -> GoalState:
        return self.assess("active", reason=reason)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    @property
    def active(self) -> bool:
        state = self.load()
        return state is not None and state.status == "active"

    def render(self) -> str:
        state = self.load()
        if state is None:
            return "No session goal is set."
        label = "ACTIVE" if state.status == "active" else "ACHIEVED"
        lines = [f"[{label}] {state.text}"]
        if state.reason:
            lines.append(f"Reason: {state.reason}")
        if state.assessments:
            lines.append(f"Assessments: {state.assessments}")
        return "\n".join(lines)


def is_continuation_request(text: str) -> bool:
    """Conservative detector for short user directives that mean 'keep executing'."""
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return False
    exact = {
        "continue",
        "continue the work",
        "continue working",
        "keep going",
        "keep working",
        "go on",
        "proceed",
        "resume",
        "resume the work",
        "carry on",
        "next",
        "do the next one",
        "complete the next todo",
        "complete the next remaining todo",
    }
    if normalized in exact:
        return True
    prefixes = (
        "continue ",
        "keep going ",
        "keep working ",
        "proceed ",
        "resume ",
        "carry on ",
        "go on ",
    )
    return any(normalized.startswith(prefix) and len(normalized) <= 120 for prefix in prefixes)
