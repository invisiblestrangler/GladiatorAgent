from __future__ import annotations

from pathlib import Path

from gladiator.environment import GladiatorLocalEnvironment
from gladiator.events import AgentEvent, EventKind


def _environment(workspace: Path, events: list[AgentEvent]) -> GladiatorLocalEnvironment:
    env = object.__new__(GladiatorLocalEnvironment)
    env.workspace_root = workspace.resolve()
    env.event_sink = events.append
    env.mentor_handler = None
    return env


def test_send_supports_multiple_files_without_real_shell_binary(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"png")
    second.write_bytes(b"png")
    events: list[AgentEvent] = []
    env = _environment(tmp_path, events)

    result = env._artifact_command("gladiator send first.png second.png", cwd=str(tmp_path))

    assert result is not None
    assert result["returncode"] == 0
    artifacts = [event for event in events if event.kind == EventKind.ARTIFACT_READY]
    assert [Path(event.data["path"]).name for event in artifacts] == ["first.png", "second.png"]


def test_malformed_send_is_intercepted_instead_of_falling_through_to_shell(tmp_path: Path):
    events: list[AgentEvent] = []
    env = _environment(tmp_path, events)

    missing_path = env._artifact_command("gladiator send", cwd=str(tmp_path))
    compound = env._artifact_command("cd /tmp && gladiator send screenshot.png", cwd=str(tmp_path))

    assert missing_path is not None
    assert missing_path["returncode"] == 2
    assert "Usage: gladiator send" in missing_path["output"]
    assert compound is not None
    assert compound["returncode"] == 2
    assert "sole command" in compound["output"]
