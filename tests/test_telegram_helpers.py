from collections import deque

import pytest

from gladiator.service import GladiatorService
from gladiator.telegram.bot import TELEGRAM_COMMANDS
from gladiator.telegram.client import TelegramClient
from gladiator.telegram.renderer import markdown_to_telegram_html, split_markdown
from gladiator.telegram.traces import TraceHighlighter, reasoning_preview, summarize_shell_command


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


def test_reasoning_preview_waits_for_useful_text_and_keeps_first_sentence():
    assert reasoning_preview("I will inspect") is None
    assert reasoning_preview("I will inspect src/gladiator/service.py first. Then I will run tests.") == (
        "I will inspect src/gladiator/service.py first."
    )


def test_command_summary_hides_long_python_payload():
    command = "python3 -c \"" + ("print('lots of implementation detail');" * 30) + "\""
    assert summarize_shell_command(command) == "Running Python"


def test_command_summary_describes_common_gladiator_actions():
    assert summarize_shell_command("gladiator send /tmp/result.png") == "Sending result.png"
    assert summarize_shell_command("gladiator todo done 2") == "Updating task list"
    assert summarize_shell_command("gladiator web search 'latest docs'") == "Searching web: latest docs"


def test_command_summary_keeps_relevant_file_names():
    crop = (
        "python3 -c \"from PIL import Image; "
        "Image.open('/tmp/run/.gladiator/inbox/comparison.png').save('/tmp/run/gladiator_square.png')\""
    )
    assert summarize_shell_command(crop) == "Processing inbox/comparison.png → run/gladiator_square.png"
    assert summarize_shell_command("cat src/gladiator/service.py") == "Reading src/gladiator/service.py"
    assert summarize_shell_command("uv run pytest tests/test_telegram_helpers.py") == (
        "Testing tests/test_telegram_helpers.py"
    )


def test_compound_svg_creation_is_summarized_without_embedded_url():
    command = (
        "mkdir -p /root/cute-dog-project && cat > /root/cute-dog-project/cute-dog.svg <<'SVG' "
        "<svg xmlns='http://www.w3.org/2000/svg'><circle cx='20' cy='20' r='10'/></svg> SVG"
    )
    summary = summarize_shell_command(command)
    assert summary == "Writing cute-dog-project/cute-dog.svg"
    assert "http" not in summary
    assert "w3.org" not in summary


def test_command_summary_truncates_unknown_commands():
    summary = summarize_shell_command("custom-tool " + "x" * 300, max_chars=80)
    assert len(summary) <= 80
    assert summary.endswith("…")


def test_final_progress_summary_preserves_observation_and_recent_results():
    milestones = deque(
        [
            "💭 I will inspect the Telegram renderer first.",
            "✓ Reading src/gladiator/service.py",
            "✓ Testing tests/test_telegram_helpers.py",
            "✓ Sent result.png",
        ],
        maxlen=5,
    )
    summary = GladiatorService._summary_items(milestones[0], milestones)
    body = GladiatorService._final_progress_body("✓ Done", summary)
    assert body.startswith("✓ Done\n💭 I will inspect the Telegram renderer first.")
    assert "src/gladiator/service.py" in body
    assert "tests/test_telegram_helpers.py" in body
    assert "result.png" in body


def test_native_telegram_command_menu_contains_runtime_controls():
    names = {name for name, _description in TELEGRAM_COMMANDS}
    assert {"status", "model", "reasoning", "trace", "provider", "compact", "new", "todo", "stop"} <= names


@pytest.mark.asyncio
async def test_set_my_commands_uses_telegram_command_payload(monkeypatch):
    client = TelegramClient("test-token")
    captured = {}

    async def fake_call(method, payload=None):
        captured["method"] = method
        captured["payload"] = payload
        return True

    monkeypatch.setattr(client, "_call", fake_call)
    try:
        assert await client.set_my_commands([("/status", "Show status"), ("new", "New session")]) is True
    finally:
        await client.close()

    assert captured == {
        "method": "setMyCommands",
        "payload": {
            "commands": [
                {"command": "status", "description": "Show status"},
                {"command": "new", "description": "New session"},
            ]
        },
    }


@pytest.mark.asyncio
async def test_telegram_messages_disable_link_previews_by_default(monkeypatch):
    client = TelegramClient("test-token")
    calls = []

    async def fake_call(method, payload=None):
        calls.append((method, payload))
        return {"message_id": 1}

    monkeypatch.setattr(client, "_call", fake_call)
    try:
        await client.send_message(123, "https://www.w3.org/2000/svg")
        await client.edit_message_text(123, 1, "https://example.com")
    finally:
        await client.close()

    assert calls[0][1]["link_preview_options"] == {"is_disabled": True}
    assert calls[1][1]["link_preview_options"] == {"is_disabled": True}
