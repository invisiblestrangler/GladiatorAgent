from __future__ import annotations

import re
import shlex
from collections import deque
from pathlib import Path


class TraceHighlighter:
    """Select useful progress milestones from streamed model reasoning without another LLM call."""

    KEYWORDS = (
        "aha",
        "now i see",
        "root cause",
        "the problem is",
        "problem is",
        "however",
        "smoking gun",
        "this explains",
        "found it",
        "the failure is",
        "tests pass",
        "test passes",
        "tests fail",
        "test fails",
        "solved",
        "fixed",
        "instead",
    )

    def __init__(self, *, max_seen: int = 100):
        self.buffer = ""
        self.seen = deque(maxlen=max_seen)

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta
        parts = re.split(r"(?<=[.!?])\s+|\n+", self.buffer)
        if len(parts) <= 1:
            return []
        self.buffer = parts.pop()
        highlights: list[str] = []
        for sentence in parts:
            sentence = sentence.strip()
            if len(sentence) < 8 or len(sentence) > 500:
                continue
            lowered = sentence.lower()
            if not any(keyword in lowered for keyword in self.KEYWORDS):
                continue
            normalized = re.sub(r"\s+", " ", lowered)
            if normalized in self.seen:
                continue
            self.seen.append(normalized)
            highlights.append(sentence)
        return highlights


def summarize_shell_command(command: str, *, max_chars: int = 120) -> str:
    """Turn a potentially huge shell command into a useful Telegram progress label.

    This is presentation-only. The real command remains in the agent trajectory and tool
    observation; Telegram should not become a scrolling shell transcript.
    """
    compact = re.sub(r"\s+", " ", command).strip()
    if not compact:
        return "Running command"

    try:
        args = shlex.split(compact)
    except ValueError:
        args = compact.split()
    if not args:
        return "Running command"

    executable = Path(args[0]).name.lower()
    lowered = compact.lower()

    if len(args) >= 2 and args[:2] == ["gladiator", "send"]:
        target = Path(args[2]).name if len(args) >= 3 else "file"
        return f"Sending {target}"
    if len(args) >= 2 and args[:2] == ["gladiator", "todo"]:
        return "Updating task list"
    if len(args) >= 2 and args[:2] == ["gladiator", "skill"]:
        return "Using skill memory"
    if len(args) >= 2 and args[:2] == ["gladiator", "web"]:
        return "Searching the web" if len(args) >= 3 and args[2] == "search" else "Reading web page"
    if len(args) >= 2 and args[:2] == ["gladiator", "ask"]:
        return "Preparing a decision question"

    if executable in {"python", "python3", "python3.11", "uv"}:
        if executable == "uv" and len(args) >= 3 and args[1:3] == ["run", "pytest"]:
            return "Running tests"
        if executable == "uv" and "ruff" in args:
            return "Running Ruff checks"
        return "Running Python"
    if executable.startswith("pytest") or " pytest" in f" {lowered}":
        return "Running tests"
    if executable == "ruff" or " ruff " in f" {lowered} ":
        return "Running Ruff checks"
    if executable == "git":
        action = args[1] if len(args) > 1 else "command"
        labels = {
            "status": "Checking Git status",
            "diff": "Inspecting Git changes",
            "log": "Inspecting Git history",
            "show": "Inspecting Git revision",
            "add": "Staging changes",
            "commit": "Committing changes",
            "push": "Pushing changes",
            "pull": "Pulling changes",
        }
        return labels.get(action, f"Running git {action}")
    if executable in {"rg", "grep", "sed", "awk", "cat", "head", "tail", "find", "ls"}:
        return "Inspecting files"
    if executable == "curl":
        return "Calling HTTP endpoint"

    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"
