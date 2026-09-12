from pathlib import Path

from gladiator.skills import SkillManager, user_explicitly_requested_skill_write


def test_skill_write_requires_explicit_positive_request():
    assert user_explicitly_requested_skill_write("Please create a skill for deploying this app")
    assert user_explicitly_requested_skill_write("save this workflow as a skill")
    assert not user_explicitly_requested_skill_write("Do not create a skill for this")
    assert not user_explicitly_requested_skill_write("Use the existing deploy skill")


def test_skill_manager_is_lazy_and_persistent(tmp_path: Path):
    manager = SkillManager(tmp_path / "skills", read_char_limit=2000)
    source = tmp_path / "candidate.md"
    source.write_text("# Deploy\nRun tests first.\n")
    target = manager.write_from_file("deploy", source)
    assert target.is_file()
    assert manager.list() == ["deploy"]
    assert "Run tests first" in manager.read("deploy")
    manager.delete("deploy")
    assert manager.list() == []
