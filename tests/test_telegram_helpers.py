from gladiator.telegram.renderer import markdown_to_telegram_html, split_markdown
from gladiator.telegram.traces import TraceHighlighter, summarize_shell_command


def test_renderer_preserves_fenced_code():
    rendered = markdown_to_telegram_html("Before\n```python\nprint('<x>')\n```\nAfter")
    assert '<pre><code class="language-python">' in rendered
    assert "print('&lt;x&gt;')" in rendered
    assert "<pre>" in rendered


def test_split_markdown_keeps_fences_balanced():
    text = "```python\n" + ("print('x')\n" * 1000) + "```"
    chunks = split_markdown(text, limit=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0


def test_trace_highlighter_only_emits_milestones():
    traces = TraceHighlighter()
    assert traces.feed("I will inspect the code. ") == []
    found = traces.feed("Aha, the root cause is the stale cache. Next I will fix it. ")
    assert found == ["Aha, the root cause is the stale cache."]


def test_command_summary_hides_long_python_payload():
    command = "python3 -c \"" + ("print('lots of implementation detail');" * 30) + "\""
    assert summarize_shell_command(command) == "Running Python"


def test_command_summary_describes_common_gladiator_actions():
    assert summarize_shell_command("gladiator send /tmp/result.png") == "Sending result.png"
    assert summarize_shell_command("gladiator todo done 2") == "Updating task list"
    assert summarize_shell_command("gladiator web search 'latest docs'") == "Searching the web"


def test_command_summary_truncates_unknown_commands():
    summary = summarize_shell_command("custom-tool " + "x" * 300, max_chars=80)
    assert len(summary) <= 80
    assert summary.endswith("…")
