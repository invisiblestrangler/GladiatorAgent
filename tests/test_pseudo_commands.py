from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gladiator.environment import GladiatorLocalEnvironment


class _CaptureWebTools:
    def __init__(self):
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str) -> str:
        self.searches.append(query)
        return f"search:{query}"

    def fetch(self, url: str) -> str:
        self.fetches.append(url)
        return f"fetch:{url}"


def _environment(workspace: Path) -> GladiatorLocalEnvironment:
    env = object.__new__(GladiatorLocalEnvironment)
    env.workspace_root = workspace.resolve()
    env.event_sink = lambda _event: None
    env.decision_handler = None
    env.mentor_handler = lambda _request: "unused"
    env.web_tools = _CaptureWebTools()
    env.skill_manager = object()
    env.skill_write_authorized = False
    env.config = SimpleNamespace(cwd=str(workspace))
    return env


def test_reserved_pseudo_commands_reject_compound_shell_syntax(tmp_path: Path):
    env = _environment(tmp_path)

    results = [
        env._artifact_command("gladiator send result.txt > /tmp/out", cwd=str(tmp_path)),
        env._decision_command(
            "cd /tmp && gladiator ask --question q --option a --option b --conservative a"
        ),
        env._mentor_command("gladiator mentor --question help | head", cwd=str(tmp_path)),
        env._web_command("echo ready; gladiator web search query"),
        env._skill_command("gladiator skill list && echo done", cwd=str(tmp_path)),
        env._web_command("(gladiator web search grouped)"),
    ]

    for result in results:
        assert result is not None
        assert result["returncode"] == 2
        assert "sole command" in result["output"]


def test_web_suffix_pipe_is_rejected_instead_of_becoming_search_text(tmp_path: Path):
    env = _environment(tmp_path)

    result = env._web_command("gladiator web search 'vercel ai gateway' | head -50")

    assert result is not None
    assert result["returncode"] == 2
    assert env.web_tools.searches == []


def test_quoted_or_escaped_shell_punctuation_remains_valid_web_input(tmp_path: Path):
    env = _environment(tmp_path)

    search = env._web_command("gladiator web search 'C++ && Rust | shell syntax; redirection > test'")
    escaped = env._web_command(r"gladiator web search foo\|bar")
    fetch = env._web_command("gladiator web fetch 'https://example.test/path?a=1&b=2'")

    assert search is not None and search["returncode"] == 0
    assert escaped is not None and escaped["returncode"] == 0
    assert fetch is not None and fetch["returncode"] == 0
    assert env.web_tools.searches == [
        "C++ && Rust | shell syntax; redirection > test",
        "foo|bar",
    ]
    assert env.web_tools.fetches == ["https://example.test/path?a=1&b=2"]


def test_reserved_words_used_as_normal_arguments_are_not_intercepted(tmp_path: Path):
    env = _environment(tmp_path)

    assert env._artifact_command("echo gladiator send", cwd=str(tmp_path)) is None
    assert env._decision_command("echo gladiator ask") is None
    assert env._mentor_command("echo gladiator mentor", cwd=str(tmp_path)) is None
    assert env._web_command("echo gladiator web") is None
    assert env._skill_command("echo gladiator skill", cwd=str(tmp_path)) is None
