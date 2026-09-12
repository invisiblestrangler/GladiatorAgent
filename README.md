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

## Setup target

```bash
uv tool install git+https://github.com/invisiblestrangler/GladiatorAgent

gladiator setup
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

## Development status

The bootstrap runtime currently includes configuration storage, the setup wizard, local SearXNG provisioning, optional Docker/browser installation hooks, context/output limiting, and the YOLO escalation policy. Telegram streaming, mini-swe-agent lifecycle integration, skills, and compaction are being layered on next.
