from __future__ import annotations

import re
import shutil
from pathlib import Path

from gladiator.runtime.context import ObservationLimiter

_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EXPLICIT_SKILL_WRITE_RE = re.compile(
    r"\b(create|make|write|add|save|update|edit|modify)\b.{0,50}\b(?:a\s+|the\s+)?skill\b",
    re.IGNORECASE | re.DOTALL,
)
_SKILL_WRITE_NEGATION_RE = re.compile(
    r"\b(don't|dont|do\s+not|never|without)\b.{0,35}\b(create|make|write|add|save|update|edit|modify)\b.{0,50}\bskill\b",
    re.IGNORECASE | re.DOTALL,
)


def user_explicitly_requested_skill_write(text: str) -> bool:
    """Conservative one-turn capability detector; false negatives are safer than auto-created skills."""
    if _SKILL_WRITE_NEGATION_RE.search(text):
        return False
    return bool(_EXPLICIT_SKILL_WRITE_RE.search(text))


class SkillManager:
    def __init__(self, root: Path, *, read_char_limit: int = 12_000):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limiter = ObservationLimiter(char_limit=read_char_limit, output_dir=None)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not _SKILL_NAME_RE.fullmatch(name):
            raise ValueError("Skill name must be 1-64 characters using letters, numbers, '_' or '-'.")
        return name

    def path_for(self, name: str) -> Path:
        return self.root / self._validate_name(name) / "SKILL.md"

    def list(self) -> list[str]:
        names: list[str] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file() and _SKILL_NAME_RE.fullmatch(child.name):
                names.append(child.name)
        return names

    def read(self, name: str) -> str:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"Skill not found: {name}")
        result = self.limiter.limit(path.read_text(encoding="utf-8", errors="replace"), label=f"skill-{name}")
        return result.text

    def write_from_file(self, name: str, source: Path) -> Path:
        target = self.path_for(name)
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Skill source file not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def delete(self, name: str) -> None:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Skill not found: {name}")
        shutil.rmtree(path.parent)
