from gladiator.telegram.renderer import markdown_to_telegram_html, split_markdown
from gladiator.telegram.traces import TraceHighlighter


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
