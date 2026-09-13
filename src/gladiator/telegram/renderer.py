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


def telegram_message_parts(text: str, rendered_limit: int = 3800) -> list[str]:
    """Return numbered Markdown parts that remain safely below Telegram's message limit.

    Telegram ultimately receives HTML, so raw-Markdown length alone is not enough: escaping
    ``<``, ``>``, and ``&`` can substantially expand the transmitted text. Re-split with a
    progressively smaller raw limit until every rendered part is under the conservative
    rendered-size ceiling. The extra headroom also covers the ``Part N/M`` label.
    """
    if not text:
        return [""]

    label_reserve = 64
    chunk_rendered_limit = max(512, rendered_limit - label_reserve)
    raw_limit = min(3200, max(512, len(text)))

    while True:
        chunks = split_markdown(text, limit=raw_limit)
        if all(len(markdown_to_telegram_html(chunk)) <= chunk_rendered_limit for chunk in chunks):
            break
        if raw_limit <= 256:
            # HTML escaping expands one source character by at most a small constant factor;
            # this floor is intentionally very conservative for pathological content.
            chunks = split_markdown(text, limit=192)
            break
        raw_limit = max(256, int(raw_limit * 0.72))

    if len(chunks) == 1:
        return chunks

    total = len(chunks)
    parts = [f"Part {index}/{total}\n\n{chunk}" for index, chunk in enumerate(chunks, 1)]
    # Defensive assertion: keep the contract local to this helper so callers can send directly.
    if any(len(markdown_to_telegram_html(part)) > rendered_limit for part in parts):
        # This is only reachable for extremely pathological escaping. Retry once with much
        # smaller source chunks rather than risking a Telegram-side rejection/truncation.
        chunks = split_markdown(text, limit=160)
        total = len(chunks)
        parts = [f"Part {index}/{total}\n\n{chunk}" for index, chunk in enumerate(chunks, 1)]
    return parts
