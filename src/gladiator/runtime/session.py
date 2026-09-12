from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SessionState:
    session_id: str
    started_at: float

    @classmethod
    def new(cls) -> "SessionState":
        return cls(session_id=f"gladiator-{uuid.uuid4().hex}", started_at=time.time())


def load_or_create_session(path: Path) -> SessionState:
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            session_id = str(raw.get("session_id") or "").strip()
            started_at = float(raw.get("started_at") or time.time())
            if session_id and len(session_id) <= 256:
                return SessionState(session_id=session_id, started_at=started_at)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    state = SessionState.new()
    save_session(path, state)
    return state


def save_session(path: Path, state: SessionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"session_id": state.session_id, "started_at": state.started_at}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def rotate_session(path: Path) -> SessionState:
    state = SessionState.new()
    save_session(path, state)
    return state
