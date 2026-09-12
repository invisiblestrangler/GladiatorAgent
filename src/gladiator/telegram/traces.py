from __future__ import annotations

import re
from collections import deque


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
