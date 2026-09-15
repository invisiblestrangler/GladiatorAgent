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


_FILE_RE = re.compile(
    r"(?<![\w.-])((?:(?:[A-Za-z]:)?[/~.]?)?(?:[\w@+,.=-]+/)*[\w@+,.=-]+\."
    r"(?:py|md|json|ya?ml|toml|txt|log|png|jpe?g|webp|gif|svg|pdf|html?|css|js|mjs|cjs|ts|tsx|jsx|"
    r"swift|rs|go|sh|bash|zsh|csv|xml|sql|ini|cfg|conf))(?![\w.-])",
    re.IGNORECASE,
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}


def reasoning_preview(text: str, *, max_chars: int = 170, min_chars: int = 60) -> str | None:
    """Return the first useful piece of one model reasoning turn for Telegram observation.

    We wait for either a complete sentence/newline or a modest amount of text so tiny
    streaming deltas do not create useless progress entries.
    """
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return None
    sentence_end = re.search(r"[.!?](?:\s|$)", compact)
    if sentence_end:
        compact = compact[: sentence_end.end()].strip()
    elif len(compact) < min_chars:
        return None
    if len(compact) > max_chars:
        compact = compact[: max_chars - 1].rstrip() + "…"
    return compact


def _short_path(value: str, *, max_chars: int = 58) -> str:
    cleaned = value.strip("'\"`()[]{}:;, ")
    if not cleaned:
        return "file"
    path = Path(cleaned)
    if path.is_absolute():
        parts = [part for part in path.parts if part not in {path.anchor, "/"}]
        cleaned = "/".join(parts[-2:]) if len(parts) >= 2 else path.name
    cleaned = cleaned.removeprefix("./")
    if len(cleaned) <= max_chars:
        return cleaned
    name = Path(cleaned).name
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1].rstrip() + "…"


def _file_names(command: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _FILE_RE.finditer(command):
        display = _short_path(match.group(1))
        key = display.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(display)
    return names


def _with_extra_files(prefix: str, files: list[str]) -> str:
    if not files:
        return prefix
    if len(files) == 1:
        return f"{prefix} {files[0]}"
    return f"{prefix} {files[0]} +{len(files) - 1} file{'s' if len(files) > 2 else ''}"


def summarize_shell_command(command: str, *, max_chars: int = 120) -> str:
    """Turn a potentially huge shell command into an informative Telegram progress label.

    This is presentation-only. The real command remains in the agent trajectory and tool
    observation; Telegram should expose enough context to observe the run without becoming
    a scrolling shell transcript.
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
    files = _file_names(compact)

    if len(args) >= 2 and args[:2] == ["gladiator", "send"]:
        targets = [Path(value).name for value in args[2:] if value and not value.startswith("-")]
        if not targets:
            return "Sending file"
        if len(targets) == 1:
            return f"Sending {targets[0]}"
        return f"Sending {targets[0]} +{len(targets) - 1} files"
    if len(args) >= 2 and args[:2] == ["gladiator", "mentor"]:
        return _with_extra_files("Consulting mentor with", files) if files else "Consulting mentor"
    if len(args) >= 2 and args[:2] == ["gladiator", "todo"]:
        return "Updating task list"
    if len(args) >= 2 and args[:2] == ["gladiator", "skill"]:
        name = args[3] if len(args) >= 4 else "skill"
        return f"Using skill {name}"
    if len(args) >= 2 and args[:2] == ["gladiator", "web"]:
        if len(args) >= 3 and args[2] == "search":
            query = " ".join(args[3:]).strip()
            return f"Searching web: {query[:70]}" if query else "Searching the web"
        return "Reading web page"
    if len(args) >= 2 and args[:2] == ["gladiator", "ask"]:
        return "Preparing a decision question"

    # Shell redirection/heredoc commands often start with mkdir/printf/cat but the useful
    # observation is the file being created, not the full payload (which may contain URLs).
    if (">" in compact or ">>" in compact) and files:
        return f"Writing {files[-1]}"

    if executable in {"python", "python3", "python3.11"}:
        image_files = [name for name in files if Path(name).suffix.lower() in _IMAGE_SUFFIXES]
        if len(image_files) >= 2:
            return f"Processing {image_files[0]} → {image_files[-1]}"
        if len(args) >= 2 and not args[1].startswith("-") and Path(args[1]).suffix.lower() == ".py":
            return f"Running {_short_path(args[1])}"
        if files:
            return _with_extra_files("Running Python with", files)
        return "Running Python"

    if executable == "uv":
        if len(args) >= 3 and args[1:3] == ["run", "pytest"]:
            targets = [name for name in files if name.endswith(".py")]
            return _with_extra_files("Testing", targets) if targets else "Running tests"
        if "ruff" in args:
            return _with_extra_files("Linting", files) if files else "Running Ruff checks"
        if len(args) >= 3 and args[1] == "run" and Path(args[2]).suffix.lower() == ".py":
            return f"Running {_short_path(args[2])}"
        return _with_extra_files("Running uv with", files) if files else "Running uv"

    if executable.startswith("pytest") or " pytest" in f" {lowered}":
        targets = [name for name in files if name.endswith(".py")]
        return _with_extra_files("Testing", targets) if targets else "Running tests"
    if executable == "ruff" or " ruff " in f" {lowered} ":
        return _with_extra_files("Linting", files) if files else "Running Ruff checks"

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
        base = labels.get(action, f"Running git {action}")
        return _with_extra_files(base + " in", files) if files else base

    if executable in {"rg", "grep"}:
        return _with_extra_files("Searching", files) if files else "Searching files"
    if executable in {"sed", "awk", "cat", "head", "tail"}:
        return _with_extra_files("Reading", files) if files else "Inspecting text"
    if executable == "find":
        return "Finding files"
    if executable == "ls":
        return "Listing files"
    if executable in {"cp", "mv"} and len(files) >= 2:
        verb = "Copying" if executable == "cp" else "Moving"
        return f"{verb} {files[0]} → {files[-1]}"
    if executable == "rm":
        return _with_extra_files("Removing", files) if files else "Removing files"
    if executable == "curl":
        return _with_extra_files("Calling HTTP endpoint for", files) if files else "Calling HTTP endpoint"

    if files:
        return _with_extra_files(f"Running {Path(args[0]).name} on", files)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"
