import json

from gladiator.agent import GladiatorAgent, SYSTEM_TEMPLATE


def _terminal_exit(submission: str) -> dict:
    return {
        "role": "exit",
        "content": submission,
        "extra": {"exit_status": "Submitted", "submission": submission},
    }


def test_system_prompt_prefers_direct_assistant_final_after_tools():
    assert "return the final user-facing answer directly as ordinary assistant text" in SYSTEM_TEMPLATE
    assert "Do NOT call bash only to submit or print the final answer" in SYSTEM_TEMPLATE
    assert "preferred explicit completion path" not in SYSTEM_TEMPLATE


def test_legacy_terminal_completion_is_collapsed_to_one_assistant_turn():
    final = "Implemented Step 2 and verified the refund invariants."
    command = "printf '%s\\n%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' '" + final + "'"
    call_id = "call_abc123"
    agent = object.__new__(GladiatorAgent)
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "implement step 2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "<returncode>0</returncode>\n<output>\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"
            + final
            + "</output>",
        },
    ]

    assert agent._normalize_terminal_submission_history(_terminal_exit(final)) is True
    assert [message["role"] for message in agent.messages] == ["system", "user", "assistant"]
    assert agent.messages[-1]["content"] == final
    assert final not in json.dumps(agent.messages[:-1])
    assert "tool_calls" not in agent.messages[-1]


def test_normal_direct_final_is_not_rewritten():
    agent = object.__new__(GladiatorAgent)
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello!"},
    ]

    assert agent._normalize_terminal_submission_history(_terminal_exit("Hello!")) is False
    assert agent.messages[-1] == {"role": "assistant", "content": "Hello!"}


def test_non_completion_tool_call_is_not_collapsed():
    agent = object.__new__(GladiatorAgent)
    call_id = "call_abc123"
    agent.messages = [
        {"role": "system", "content": "stable-prefix"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": "pytest -q"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "3 passed"},
    ]

    assert agent._normalize_terminal_submission_history(_terminal_exit("Done")) is False
    assert agent.messages[-1]["role"] == "tool"
