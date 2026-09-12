# GladiatorAgent

GladiatorAgent is a lightweight, Telegram-first coding-agent runtime built on top of Stanford/SWE-agent's `mini-swe-agent`.
The goal is Hermes-like usability on the go while preserving mini-swe-agent's small, coding-focused context footprint.

## Design rules

- **YOLO by default.** Routine commands, edits, tests, and implementation choices do not ask for confirmation.
- **Escalation is exceptional.** Gladiator only pauses for user input when a materially consequential choice remains genuinely unresolved after investigation. If the user does not answer within one hour, Gladiator resumes with the explicitly identified conservative option.
- **UI is not agent memory.** Telegram typing state, progress traces, message IDs, transport metadata, and formatting never enter model context.
- **Filesystem is external memory.** Large outputs and files stay on disk; the model gets bounded relevant excerpts and paths.
- **No automatic skills.** Skills are only created or modified after an explicit user request.
- **300k compaction target.** The effective threshold is the smaller of 300k tokens or roughly 82% of the selected model's known context window.
- **Cache-friendly history.** API-visible history is append-only within a session and excludes UI/reasoning metadata. OpenRouter sessions use a stable routing/session key and response caching for identical retries.

## Setup

```bash
uv tool install git+https://github.com/invisiblestrangler/GladiatorAgent

gladiator setup
gladiator run
```

The interactive setup wizard asks for:

- OpenAI-compatible endpoint and API key
- model and reasoning level
- Telegram bot token
- web search mode
- optional local-only SearXNG
- optional Docker installation when local SearXNG is selected and Docker is missing
- optional `browser-use` installation for heavy browser automation

No manual config-file editing is required.

## Telegram controls

- `/status` — provider/model/reasoning, approximate context size, session, and provider-reported prompt-cache metrics
- `/model [id]` — show or change model
- `/reasoning [off|minimal|low|medium|high|xhigh]`
- `/trace [off|milestones|verbose]`
- `/provider [endpoint] [api-key]` — show/change OpenAI-compatible provider
- `/compact` — compact at the next safe agent boundary
- `/new` — archive the current conversational state and start a clean session while keeping workspace files, skills, and settings
- `/stop` — stop the current run

## Sessions and caching

Gladiator keeps one persistent session ID per conversation session. On OpenRouter that ID is sent both as `session_id` and `x-session-id`, improving sticky routing to the provider that already has the prompt prefix cached. The system prompt, tool schema, and previous messages stay byte-stable as the conversation grows; Telegram/UI events and `message.extra` metadata are not sent to the model.

The current trajectory and session ID are persisted under the workspace's `.gladiator/` directory. Restarting Gladiator restores that trajectory, so a process restart does not automatically throw away conversational state or the stable cache-routing key.

`/new` intentionally rotates the session ID and clears conversation history. The previous trajectory and compact handoff are archived under `.gladiator/sessions/`; workspace files, global skills, and configuration are left alone.

At the compaction boundary Gladiator writes `.gladiator/contextAfterCompact.md`, validates it, and replaces the old message history with a small stable resume context. The first request after compaction is naturally a new prompt prefix; subsequent turns can cache that new prefix normally.

## Web and files

- `gladiator web search ...` uses configured SearXNG with bounded results.
- `gladiator web fetch URL` performs lightweight deterministic extraction.
- Browser automation is optional and kept outside the core dependency set.
- Uploaded files stay on disk instead of being automatically dumped into context.
- Images can be supplied as multimodal input.
- `gladiator send PATH` sends an image or file back through Telegram without exposing Telegram internals to the model.

## Skills

Skills are lazy external memory. The agent sees names first and reads one relevant skill at a time. Skill creation/update/delete is runtime-locked unless the current user request explicitly asks to create or change a skill; there is no automatic learning or automatic skill creation.

## Development status

The bootstrap runtime includes Telegram streaming and typing feedback, OpenAI-compatible streaming, image/file transfer, provider/model/reasoning controls, context compaction, persistent sessions, OpenRouter cache telemetry, lazy explicit skills, SearXNG/web extraction, optional Browser Use installation, YOLO execution, and conservative one-hour escalation fallback.
