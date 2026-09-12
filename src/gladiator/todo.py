from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TodoItem:
    id: int
    text: str
    done: bool = False


class TodoManager:
    """Small filesystem-backed task ledger kept outside model context."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> list[TodoItem]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return []
        result: list[TodoItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                result.append(
                    TodoItem(
                        id=int(item["id"]),
                        text=str(item["text"]),
                        done=bool(item.get("done", False)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _save(self, items: list[TodoItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [{"id": item.id, "text": item.text, "done": item.done} for item in items]}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def list(self) -> list[TodoItem]:
        return self._load()

    def add(self, text: str) -> TodoItem:
        text = text.strip()
        if not text:
            raise ValueError("TODO text must not be empty")
        items = self._load()
        next_id = max((item.id for item in items), default=0) + 1
        item = TodoItem(id=next_id, text=text)
        items.append(item)
        self._save(items)
        return item

    def mark_done(self, item_id: int) -> TodoItem:
        items = self._load()
        for item in items:
            if item.id == item_id:
                item.done = True
                self._save(items)
                return item
        raise ValueError(f"TODO #{item_id} does not exist")

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    @property
    def open_count(self) -> int:
        return sum(not item.done for item in self._load())

    def render(self) -> str:
        items = self._load()
        if not items:
            return "No tracked TODOs."
        lines = []
        for item in items:
            marker = "x" if item.done else " "
            lines.append(f"- [{marker}] #{item.id} {item.text}")
        return "\n".join(lines)
