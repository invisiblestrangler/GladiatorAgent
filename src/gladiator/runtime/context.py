from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ContextBudget:
    configured_compact_threshold: int = 300_000
    model_context_window: int | None = None
    model_window_fraction: float = 0.82

    @property
    def effective_compact_threshold(self) -> int:
        if not self.model_context_window:
            return self.configured_compact_threshold
        safe_model_limit = int(self.model_context_window * self.model_window_fraction)
        return min(self.configured_compact_threshold, safe_model_limit)

    def should_compact(self, current_context_tokens: int) -> bool:
        return current_context_tokens >= self.effective_compact_threshold


@dataclass(slots=True)
class ObservationLimitResult:
    text: str
    truncated: bool
    original_chars: int
    saved_path: Path | None = None


class ObservationLimiter:
    """Keep large tool output on disk and inject only a bounded useful excerpt."""

    def __init__(self, char_limit: int = 12_000, output_dir: Path | None = None):
        self.char_limit = max(1_000, char_limit)
        self.output_dir = output_dir

    def limit(self, text: str, *, label: str = "tool-output") -> ObservationLimitResult:
        original_chars = len(text)
        if original_chars <= self.char_limit:
            return ObservationLimitResult(text=text, truncated=False, original_chars=original_chars)

        saved_path = self._save_full_output(text, label=label)
        head_budget = int(self.char_limit * 0.55)
        tail_budget = int(self.char_limit * 0.25)
        signal_budget = self.char_limit - head_budget - tail_budget

        head = text[:head_budget]
        tail = text[-tail_budget:]
        signal = self._extract_signal_lines(text, signal_budget)
        storage_note = f"Full output saved to: {saved_path}" if saved_path else "Full output was truncated."
        excerpt = (
            f"[output truncated: {original_chars:,} chars]\n"
            f"{storage_note}\n\n"
            f"--- beginning ---\n{head}\n"
            f"--- likely-relevant lines ---\n{signal}\n"
            f"--- end ---\n{tail}"
        )
        return ObservationLimitResult(
            text=excerpt,
            truncated=True,
            original_chars=original_chars,
            saved_path=saved_path,
        )

    def _save_full_output(self, text: str, *, label: str) -> Path | None:
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-") or "tool-output"
        index = len(list(self.output_dir.glob(f"{safe_label}-*.txt"))) + 1
        target = self.output_dir / f"{safe_label}-{index:04d}.txt"
        target.write_text(text, encoding="utf-8", errors="replace")
        return target

    @staticmethod
    def _extract_signal_lines(text: str, budget: int) -> str:
        needles = (
            "error",
            "exception",
            "failed",
            "failure",
            "traceback",
            "warning",
            "assert",
            "fatal",
            "passed",
        )
        selected: list[str] = []
        used = 0
        seen: set[str] = set()
        for line in text.splitlines():
            lowered = line.lower()
            if not any(needle in lowered for needle in needles):
                continue
            normalized = line.strip()
            if not normalized or normalized in seen:
                continue
            if used + len(line) + 1 > budget:
                break
            selected.append(line)
            seen.add(normalized)
            used += len(line) + 1
        return "\n".join(selected) if selected else "(no obvious error/test signal lines found)"
