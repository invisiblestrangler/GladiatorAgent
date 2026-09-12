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
- **Cache-friendly history.** API-visible history is append-only within a session, keeps the system/tool prefix stable, and excludes UI/reasoning metadata. Gladiator does not inject provider-specific cache or routing controls.

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

- `/status` — provider/model/reasoning, approximate context size, local session, TODO count, and provider-reported prompt-cache metrics when available
- `/model [id]` — show or change model
- `/reasoning [off|minimal|low|medium|high|xhigh]`
- `/trace [off|milestones|verbose]`
- `/provider [endpoint] [api-key]` — show/change OpenAI-compatible provider
- `/compact` — compact at the next safe agent boundary
- `/new` — archive the current conversational state and start a clean session while keeping workspace files, skills, and settings
- `/todo` — show the current external task ledger
- `/stop` — stop the current run

## Sessions and caching

Gladiator keeps the system prompt, tool schema, and previous API-visible messages byte-stable as a session grows. New information is appended at the tail instead of rewriting prior messages. Telegram/UI events, timestamps, reasoning traces, and `message.extra` metadata are not sent back to the model.

This is intentionally provider-agnostic: Gladiator does not send sticky-routing IDs, cache headers, provider-selection hints, or other vendor-specific cache controls. If a compatible backend implements prefix caching, it can reuse the stable prompt prefix naturally.

The current trajectory and local session ID are persisted under the workspace's `.gladiator/` directory. Restarting Gladiator restores that trajectory. `/new` rotates the local session ID and clears conversation/TODO state; the previous trajectory, compact handoff, and TODO ledger are archived under `.gladiator/sessions/`. Workspace files, global skills, and configuration are left alone.

At the compaction boundary Gladiator writes `.gladiator/contextAfterCompact.md`, validates it, and replaces the old message history with a small stable resume context. The first request after compaction is naturally a new prompt prefix; subsequent turns can cache that prefix normally.

If the provider reports cache-token usage using OpenAI-compatible usage fields, `/status` surfaces it. Missing cache telemetry is treated as "not reported", not as a cache miss.

## Task ledger

For multi-step work, Gladiator can maintain a tiny external TODO ledger at `.gladiator/todo.json`. It stays outside normal model context and is read only when needed.

Agent-facing commands:

```bash
gladiator todo show
gladiator todo add "inspect failing auth test"
gladiator todo done 1
gladiator todo clear
```

The agent is instructed to use this for multi-step work, keep it concise, and avoid creating TODOs for trivial one-step tasks. `/todo` lets the Telegram user inspect the current ledger without adding it to model context. The ledger survives process restarts and compaction, and `/new` archives then clears it.

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

The bootstrap runtime includes Telegram streaming and typing feedback, provider-agnostic OpenAI-compatible streaming, image/file transfer, provider/model/reasoning controls, context compaction, persistent sessions, provider-reported cache telemetry when available, the external TODO ledger, lazy explicit skills, SearXNG/web extraction, optional Browser Use installation, YOLO execution, and conservative one-hour escalation fallback.
