from __future__ import annotations

import html
import re

_FENCE_RE = re.compile(r"```([\w.+#-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")


def markdown_to_telegram_html(text: str) -> str:
    """Render the small Markdown subset coding-agent replies need most."""
    placeholders: list[str] = []

    def replace_fence(match: re.Match[str]) -> str:
        language = match.group(1).strip()
        code = html.escape(match.group(2), quote=False)
        cls = f' class="language-{html.escape(language, quote=True)}"' if language else ""
        placeholders.append(f"<pre><code{cls}>{code}</code></pre>")
        return f"\x00BLOCK{len(placeholders) - 1}\x00"

    without_blocks = _FENCE_RE.sub(replace_fence, text)
    escaped = html.escape(without_blocks, quote=False)
    escaped = _INLINE_CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", escaped)
    escaped = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    for idx, rendered in enumerate(placeholders):
        escaped = escaped.replace(html.escape(f"\x00BLOCK{idx}\x00"), rendered)
    return escaped


def split_markdown(text: str, limit: int = 3500) -> list[str]:
    """Split while keeping fenced code blocks syntactically valid across chunks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    fence_open = False
    fence_header = "```"

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        if fence_open:
            current.append("```\n")
        chunks.append("".join(current).rstrip())
        current = [fence_header + "\n"] if fence_open else []
        current_len = len(current[0]) if current else 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not fence_open:
                fence_open = True
                fence_header = stripped
            else:
                fence_open = False
        if current_len + len(line) > limit and current:
            flush()
        while len(line) > limit - current_len and limit - current_len > 100:
            take = limit - current_len - (4 if fence_open else 0)
            current.append(line[:take])
            line = line[take:]
            flush()
        current.append(line)
        current_len += len(line)
    flush()
    return [chunk for chunk in chunks if chunk]
