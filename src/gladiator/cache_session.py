from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CacheStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def hit_ratio(self) -> float | None:
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens

    def add_usage(self, usage: dict) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            self.cached_tokens += int(details.get("cached_tokens") or 0)
            self.cache_write_tokens += int(details.get("cache_write_tokens") or 0)
