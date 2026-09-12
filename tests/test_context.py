from pathlib import Path

from gladiator.runtime.context import ContextBudget, ObservationLimiter


def test_context_threshold_uses_smaller_model_safe_limit():
    budget = ContextBudget(
        configured_compact_threshold=300_000,
        model_context_window=200_000,
        model_window_fraction=0.82,
    )
    assert budget.effective_compact_threshold == 164_000
    assert budget.should_compact(164_000)
    assert not budget.should_compact(163_999)


def test_context_threshold_uses_configured_target_for_large_models():
    budget = ContextBudget(configured_compact_threshold=300_000, model_context_window=1_000_000)
    assert budget.effective_compact_threshold == 300_000


def test_observation_limiter_saves_full_output_and_keeps_signal(tmp_path: Path):
    text = "start\n" + ("noise\n" * 3000) + "ERROR: smoking gun\n" + ("tail\n" * 3000)
    limiter = ObservationLimiter(char_limit=2_000, output_dir=tmp_path)
    result = limiter.limit(text, label="pytest")
    assert result.truncated
    assert result.saved_path is not None
    assert result.saved_path.read_text() == text
    assert "ERROR: smoking gun" in result.text
    assert len(result.text) < 3_000
