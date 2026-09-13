from __future__ import annotations

import pytest

from gladiator.cache_session import CacheStats
from gladiator.service_ext import ExtendedGladiatorService


def test_average_cache_hit_ratio_is_request_weighted_not_token_weighted():
    stats = CacheStats()
    stats.add_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}})
    stats.add_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}})
    stats.add_usage({"prompt_tokens": 400, "prompt_tokens_details": {"cached_tokens": 360}})

    assert stats.hit_ratio == pytest.approx(440 / 600)
    assert stats.average_request_hit_ratio == pytest.approx((0.8 + 0.0 + 0.9) / 3)
    assert stats.cache_reporting_requests == 3
    assert stats.last_request_hit_ratio == pytest.approx(0.9)


def test_requests_without_explicit_cached_token_telemetry_are_not_counted():
    stats = CacheStats()
    stats.add_usage({"prompt_tokens": 100})
    stats.add_usage({"prompt_tokens": 200, "prompt_tokens_details": {}})

    assert stats.prompt_tokens == 300
    assert stats.average_request_hit_ratio is None
    assert stats.cache_reporting_requests == 0


def test_input_token_usage_shape_is_supported():
    stats = CacheStats()
    stats.add_usage({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 750}})

    assert stats.prompt_tokens == 1000
    assert stats.cached_tokens == 750
    assert stats.average_request_hit_ratio == pytest.approx(0.75)


def test_status_cache_average_text_includes_sample_count():
    service = ExtendedGladiatorService.__new__(ExtendedGladiatorService)
    service.cache_stats = CacheStats()
    assert service._cache_average_text() == "not reported"

    service.cache_stats.add_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}})
    service.cache_stats.add_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 10}})
    assert service._cache_average_text() == "50.0% across 2 request(s)"
