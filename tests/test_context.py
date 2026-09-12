import json
from pathlib import Path

import pytest
from minisweagent.exceptions import Submitted

from gladiator.agent import GladiatorAgent
from gladiator.models.openai_streaming import OpenAICompatibleStreamingModel
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


class _SubmittingEnvironment:
    def execute(self, _action):
        raise Submitted(
            {
                "role": "exit",
                "content": "Finished cleanly.",
                "extra": {"exit_status": "Submitted", "submission": "Finished cleanly."},
            }
        )

    def get_template_vars(self, **_kwargs):
        return {}

    def serialize(self):
        return {}


def _completion_agent(tmp_path: Path):
    model = OpenAICompatibleStreamingModel(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model_name="test/model",
    )
    agent = GladiatorAgent(
        model,
        _SubmittingEnvironment(),
        context_after_compact_path=tmp_path / "contextAfterCompact.md",
        output_path=tmp_path / "trajectory.json",
        step_limit=0,
        cost_limit=0.0,
    )
    return agent, model


def _completion_message(call_id: str = "call_terminal") -> dict:
    command = "printf completion-marker"
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        "extra": {"actions": [{"command": command, "tool_call_id": call_id}]},
    }


def test_submitted_completion_persists_matching_tool_result_before_exit(tmp_path: Path):
    agent, model = _completion_agent(tmp_path)
    completion = _completion_message()
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "do the work"},
        completion,
    ]

    with pytest.raises(Submitted):
        agent.execute_actions(completion)

    observation = agent.messages[-1]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call_terminal"
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in observation["content"]
    model._prepare_messages_for_api(agent.messages)


def test_next_user_turn_repairs_legacy_submitted_gap_without_new_session(tmp_path: Path):
    agent, model = _completion_agent(tmp_path)
    completion = _completion_message("call_legacy")
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "create the file"},
        completion,
        {
            "role": "exit",
            "content": "Finished cleanly.",
            "extra": {"exit_status": "Submitted", "submission": "Finished cleanly."},
        },
    ]

    agent._start_or_continue_task("What happened during the previous task?")

    assert [message["role"] for message in agent.messages[-3:]] == ["assistant", "tool", "user"]
    repaired = agent.messages[-2]
    assert repaired["tool_call_id"] == "call_legacy"
    assert repaired["extra"]["legacy_terminal_repair"] is True
    assert "older Gladiator version" in repaired["content"]
    assert agent.messages[-1]["content"].startswith("New user request:")
    model._prepare_messages_for_api(agent.messages)


def test_non_submitted_exit_does_not_invent_tool_result(tmp_path: Path):
    agent, _model = _completion_agent(tmp_path)
    completion = _completion_message("call_cancelled")
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "do the work"},
        completion,
        {
            "role": "exit",
            "content": "Cancelled by user",
            "extra": {"exit_status": "Cancelled", "submission": "Cancelled by user."},
        },
    ]

    agent._start_or_continue_task("continue")

    assert agent.messages[-2] is completion
    assert agent.messages[-1]["role"] == "user"
