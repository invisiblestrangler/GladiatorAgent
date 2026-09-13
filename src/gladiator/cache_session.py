from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CacheStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cache_reporting_requests: int = 0
    request_hit_ratio_sum: float = 0.0
    last_request_hit_ratio: float | None = None

    @property
    def hit_ratio(self) -> float | None:
        """Token-weighted cache hit ratio across all reported usage."""
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens

    @property
    def average_request_hit_ratio(self) -> float | None:
        """Arithmetic mean of per-request cache hit ratios.

        Only requests where the provider explicitly reports both prompt tokens and
        cached tokens are included. This makes intermittent cold routes visible
        instead of letting large warm requests dominate the aggregate ratio.
        """
        if self.cache_reporting_requests <= 0:
            return None
        return self.request_hit_ratio_sum / self.cache_reporting_requests

    def add_usage(self, usage: dict) -> None:
        prompt_value = usage.get("prompt_tokens")
        if prompt_value is None:
            prompt_value = usage.get("input_tokens")
        prompt_tokens = int(prompt_value or 0)
        self.prompt_tokens += prompt_tokens

        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = usage.get("input_tokens_details")
        if not isinstance(details, dict):
            return

        cached_reported = "cached_tokens" in details
        cached_tokens = int(details.get("cached_tokens") or 0)
        self.cached_tokens += cached_tokens
        self.cache_write_tokens += int(details.get("cache_write_tokens") or 0)

        if cached_reported and prompt_tokens > 0:
            ratio = max(0.0, min(1.0, cached_tokens / prompt_tokens))
            self.cache_reporting_requests += 1
            self.request_hit_ratio_sum += ratio
            self.last_request_hit_ratio = ratio
