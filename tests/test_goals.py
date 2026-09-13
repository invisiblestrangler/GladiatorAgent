from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path

from gladiator.goal import GoalManager, is_continuation_request
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.telegram.bot import IncomingTask
from gladiator.todo import TodoManager


def test_goal_manager_lifecycle(tmp_path: Path):
    manager = GoalManager(tmp_path / "goal.json")
    assert manager.load() is None
    assert manager.render() == "No session goal is set."

    state = manager.set("Finish the refund hardening work")
    assert state.status == "active"
    assert manager.active is True

    state = manager.assess("achieved", reason="All acceptance criteria passed.")
    assert state.status == "achieved"
    assert state.assessments == 1
    assert "ACHIEVED" in manager.render()

    state = manager.reopen(reason="A regression was found.")
    assert state.status == "active"
    assert state.assessments == 2

    manager.clear()
    assert manager.load() is None


def test_continuation_detector_is_conservative():
    assert is_continuation_request("continue the work")
    assert is_continuation_request("Keep going")
    assert is_continuation_request("proceed with the next TODO")
    assert not is_continuation_request("Can you explain what continue means in Python?")
    assert not is_continuation_request("Please write a new implementation plan for the next release")


def test_goal_context_turns_continue_into_execution_instruction(tmp_path: Path):
    service = object.__new__(ExtendedGladiatorService)
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.todo_manager = TodoManager(tmp_path / "todo.json")
    service.goal_manager.set("Finish the refund system hardening")
    service.todo_manager.add("Implement the stuck reservation release path")
    service.todo_manager.add("Add regression tests")

    incoming = service._with_goal_context(IncomingTask(text="continue the work"), force_continuation=True)

    assert "Session goal (active): Finish the refund system hardening" in incoming.text
    assert "#1 Implement the stuck reservation release path" in incoming.text
    assert "[GLADIATOR_GOAL_TRACKING_REQUIRED]" in incoming.text
    assert "[GLADIATOR_CONTINUATION_WORK_REQUIRED]" in incoming.text
    assert "Do not repeat" in incoming.text
    assert "[[GLADIATOR_GOAL: achieved]]" in incoming.text


def test_goal_marker_is_removed_from_visible_answer():
    text = "Implemented the remaining refund fix and tests pass.\n[[GLADIATOR_GOAL: achieved]]"
    clean, status = ExtendedGladiatorService._extract_goal_marker(text)
    assert clean == "Implemented the remaining refund fix and tests pass."
    assert status == "achieved"


def test_repeat_guard_catches_same_report_but_not_new_progress():
    previous = (
        "Refund investigation report. The Stripe account restriction is the primary issue. "
        "There is also a stuck credits reservation that needs a supported release path. "
        "No repository files were modified in this investigation."
    )
    repeated = previous + "\n[[GLADIATOR_GOAL: incomplete]]"
    progressed = (
        "Implemented releaseRefundInTransaction, added the admin error surface, and added three regression tests. "
        "The refund suite now passes. The remaining TODO is Stripe dashboard reconciliation."
    )
    assert ExtendedGladiatorService._substantially_repeats(previous, repeated)
    assert not ExtendedGladiatorService._substantially_repeats(previous, progressed)


def test_goal_cannot_be_marked_achieved_while_todos_are_open(tmp_path: Path):
    service = object.__new__(ExtendedGladiatorService)
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.todo_manager = TodoManager(tmp_path / "todo.json")
    service._goal_assessment = ContextVar("test_goal_assessment", default=None)
    service.goal_manager.set("Finish all refund work")
    service.todo_manager.add("One remaining regression test")
    service._goal_assessment.set("achieved")

    service._apply_goal_assessment()

    state = service.goal_manager.load()
    assert state is not None
    assert state.status == "active"
    assert "1 TODO" in state.reason


def test_goal_is_marked_achieved_when_model_says_so_and_todos_are_done(tmp_path: Path):
    service = object.__new__(ExtendedGladiatorService)
    service.goal_manager = GoalManager(tmp_path / "goal.json")
    service.todo_manager = TodoManager(tmp_path / "todo.json")
    service._goal_assessment = ContextVar("test_goal_assessment_done", default=None)
    service.goal_manager.set("Finish all refund work")
    service._goal_assessment.set("achieved")

    service._apply_goal_assessment()

    state = service.goal_manager.load()
    assert state is not None
    assert state.status == "achieved"
    assert state.assessments == 1
