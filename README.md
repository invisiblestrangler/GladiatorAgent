# GladiatorAgent

GladiatorAgent is a lightweight, Telegram-first coding-agent runtime built on top of Stanford/SWE-agent's `mini-swe-agent`.
The goal is Hermes-like usability on the go while preserving mini-swe-agent's small, coding-focused context footprint.

You run Gladiator beside a local project, message it from Telegram, and let the agent inspect/edit/test the workspace using an OpenAI-compatible model endpoint. Telegram stays a UI rather than becoming agent memory.

## Highlights

- **Telegram-first coding agent** with image/file input and file/image return.
- **Native Telegram command menu** registered automatically, so typing `/` shows the available controls.
- **Live, compact progress cards** with model-intent previews, filenames, tool targets, test status, and an inline Stop button.
- **Supervised background mode** with automatic restart after failures and autostart on Linux reboot / macOS login.
- **YOLO by default** for routine coding work.
- **OpenAI-compatible and provider-agnostic** model transport.
- **Configurable model and reasoning level** from Telegram.
- **Persistent sessions** with `/new` for a clean conversational reset.
- **Automatic context compaction** around 300k tokens, or earlier when required by the selected model window.
- **External TODO ledger** for multi-step work without bloating model context.
- **Explicit-only skills**: Gladiator never auto-learns or creates skills on its own.
- **Optional SearXNG and browser automation** for web work.
- **Context-bounded observations** so giant shell/web outputs stay on disk instead of flooding the prompt.

## Install

### Requirements

You need:

- macOS or Linux recommended; Windows should work for foreground use anywhere the Python dependencies and shell environment are supported
- Git
- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- a Telegram bot token
- an OpenAI-compatible API endpoint, API key, and model ID

Docker is **optional**. Gladiator only needs it if you choose the local SearXNG option during setup. Browser automation is also optional.

### 1. Install `uv`

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new shell, or make sure the directory printed by the installer is on your `PATH`.

If you already have `uv`, skip this step.

### 2. Install GladiatorAgent

Install directly from GitHub:

```bash
uv tool install --python 3.11 git+https://github.com/invisiblestrangler/GladiatorAgent.git
```

Check it:

```bash
gladiator version
```

To upgrade later:

```bash
uv tool install --force --python 3.11 git+https://github.com/invisiblestrangler/GladiatorAgent.git
```

If Gladiator is installed as a background service, restart it after upgrading:

```bash
gladiator service restart
```

### 3. Create a Telegram bot

In Telegram:

1. Open **@BotFather**.
2. Run `/newbot`.
3. Choose a name and username.
4. Copy the bot token BotFather gives you.

You will paste this token into `gladiator setup`. Gladiator verifies the token before saving the configuration.

### 4. Run the setup wizard

```bash
gladiator setup
```

The wizard asks for:

- OpenAI-compatible endpoint — include `/v1` when your provider expects it
- API key
- model ID; Gladiator can try to fetch `/models` for you
- reasoning level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `ultra`
- Telegram bot token
- web search mode: none, local SearXNG, or an existing SearXNG instance
- optional Docker installation when local SearXNG is selected and Docker is missing
- optional isolated `browser-use` installation for heavier browser automation

Reasoning effort is passed through to the selected OpenAI-compatible endpoint; individual providers/models may support only a subset of these values.

Secrets are stored in Gladiator's local user configuration and are not inserted into model context.

### 5. Start Gladiator once in the foreground

From the repository/project you want Gladiator to work on:

```bash
cd /path/to/your/project
gladiator run
```

The current directory is the workspace. You can also specify one explicitly:

```bash
gladiator run --workspace /path/to/your/project
```

`gladiator run` is intentionally the foreground/debug command. It stays attached to the terminal until you stop it with Ctrl-C.

Workspace-local runtime state is kept under:

```text
.gladiator/
```

That directory contains things such as the current trajectory, bounded tool-output files, uploaded Telegram attachments, TODO state, and compaction handoff data.

### 6. Pair your Telegram account

On the first foreground run Gladiator prints a six-digit pairing code in the terminal.

Open your bot in Telegram and send:

```text
/pair 123456
```

using the code printed by your Gladiator process.

After pairing, simply send the bot a coding task. You can also attach an image or file with the request.

Gladiator registers its native Telegram command menu every time the bot starts. After the bot is online, typing `/` in Telegram shows commands such as `/status`, `/model`, `/reasoning`, `/new`, `/todo`, and `/stop`.

A typical run looks like:

```text
Working…
💭 I’ll inspect the auth middleware and its tests first.
✓ Reading src/auth/middleware.py
✓ Testing tests/test_auth.py

↓ when finished

✓ Done
💭 I’ll inspect the auth middleware and its tests first.
✓ Reading src/auth/middleware.py
✓ Testing tests/test_auth.py
```

The separate final Telegram message contains the actual answer. Progress/UI text never becomes model memory.

### 7. Install the persistent background service (recommended)

Once pairing works, stop the foreground process with Ctrl-C and install the supervised service from the project workspace:

```bash
cd /path/to/your/project
gladiator service install
```

Or provide the workspace explicitly:

```bash
gladiator service install --workspace /path/to/your/project
```

On **Linux**, Gladiator uses systemd. The service is enabled for automatic startup and uses `Restart=on-failure`. For a non-root user Gladiator also enables systemd lingering so it can start after reboot without an interactive login. If the host requires elevated permission to enable lingering, Gladiator prints the exact `sudo loginctl enable-linger ...` command required.

On **macOS**, Gladiator installs a LaunchAgent with `RunAtLoad` and restart-on-failure behavior. It starts automatically when that user logs in after reboot.

Manage the service with:

```bash
gladiator service status
gladiator service restart
gladiator service stop
gladiator service start
gladiator service uninstall
```

So normal usage is background/supervised; `gladiator run` remains useful when you want foreground logs or are debugging setup.

## Quick start

If `uv` is already installed and you already have a Telegram bot token:

```bash
uv tool install --python 3.11 git+https://github.com/invisiblestrangler/GladiatorAgent.git
gladiator setup
cd /path/to/project
gladiator run
```

Pair once from Telegram, press Ctrl-C, then make it persistent:

```bash
gladiator service install
```

After that, the service runs independently of your shell and is supervised by the operating system.

## Telegram controls

Typing `/` in Telegram displays the registered command list while Gladiator is running.

- `/start` or `/help` — show available controls
- `/status` — provider/model/reasoning, approximate context size, local session, TODO count, and provider-reported prompt-cache metrics when available
- `/model [id]` — show or change model
- `/reasoning [off|minimal|low|medium|high|xhigh|max|ultra]`
- `/trace [off|milestones|verbose]`
- `/provider [endpoint] [api-key]` — show/change OpenAI-compatible provider; when a key is included Gladiator attempts to remove that Telegram message immediately
- `/compact` — compact at the next safe agent boundary
- `/new` — archive the current conversational state and start a clean session while keeping workspace files, skills, and settings
- `/todo` — show the current external task ledger
- `/stop` — stop the current run

The default `milestones` trace mode is intended for normal use: enough information to observe what the model is doing without dumping its entire reasoning or shell transcript into Telegram.

## Background service behavior

Gladiator does not hide a second internal "gateway" process. The long-running Telegram/model runtime is the process managed by systemd or launchd.

- `gladiator run` — foreground process; closing the shell/process stops it.
- `gladiator service install` — installs and starts the supervised background process.
- Linux systemd service — starts after reboot and restarts after unexpected failures.
- macOS LaunchAgent — starts at user login and restarts after unexpected failures.
- Planned/manual stops are respected; the supervisor does not immediately resurrect a service that you intentionally stopped through the service manager.

The current conversation trajectory is persisted under `.gladiator/`, so a supervised process restart restores the saved agent session rather than intentionally creating a new one.

## Provider compatibility and tool-roundtrip errors

Gladiator is intentionally provider-agnostic and uses the OpenAI-compatible chat-completions/tool-call contract rather than provider-specific routing or cache controls.

For multi-turn tool work, Gladiator validates the local transcript before every request. An assistant tool call must have a non-empty unique tool-call ID and must be followed by the matching `role: "tool"` result before the next assistant/user turn is sent upstream. If a streaming provider omits a tool-call ID, Gladiator generates a short unique `call_...` ID and uses the same ID for the matching tool result.

Terminal completion is also closed as a valid tool round trip. mini-swe signals `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` by raising a terminal exception from inside tool execution; Gladiator records the corresponding `role: "tool"` observation before persisting the final submission so the next user request can continue the same session cleanly.

Older Gladiator versions could save a successful completion as `assistant(tool_call) -> exit` without the matching tool observation. When continuing one of those specifically completed sessions, current Gladiator repairs that exact legacy terminal gap automatically before adding the next user turn. It does **not** fabricate tool results for crashes, cancellations, or unrelated malformed history.

This catches malformed or poisoned tool history locally instead of repeatedly sending a transcript that a stricter compatible endpoint will reject.

When a provider returns HTTP 4xx/5xx, Gladiator preserves a **sanitized provider error body** and surfaces request/provider metadata when the endpoint supplies it. Telegram should therefore show the actual provider reason instead of only a generic `httpx` error and documentation link.

If a genuinely malformed session from an older version still fails local transcript validation, update Gladiator, restart the service, then use `/new` once to discard the old conversational transcript while keeping workspace files, skills, and settings:

```bash
uv tool install --force --python 3.11 git+https://github.com/invisiblestrangler/GladiatorAgent.git
gladiator service restart
```

Then in Telegram:

```text
/new
```

If the provider still rejects a valid tool round trip, the error message should contain the provider's actionable detail and, when available, a request ID. That information is the right starting point for diagnosing endpoint-specific incompatibilities.

Telegram link previews are disabled for Gladiator progress/final bot messages by default, and long compound shell commands are summarized into useful labels such as `Writing project/output.svg` rather than exposing embedded URLs or dumping full heredocs into the chat.

## Design rules

- **YOLO by default.** Routine commands, edits, tests, and implementation choices do not ask for confirmation.
- **Escalation is exceptional.** Gladiator only pauses for user input when a materially consequential choice remains genuinely unresolved after investigation. If the user does not answer within one hour, Gladiator resumes with the explicitly identified conservative option.
- **UI is not agent memory.** Telegram typing state, progress traces, message IDs, transport metadata, and formatting never enter model context.
- **Filesystem is external memory.** Large outputs and files stay on disk; the model gets bounded relevant excerpts and paths.
- **No automatic skills.** Skills are only created or modified after an explicit user request.
- **300k compaction target.** The effective threshold is the smaller of 300k tokens or roughly 82% of the selected model's known context window.
- **Cache-friendly history.** API-visible history is append-only within a session, keeps the system/tool prefix stable, and excludes UI/reasoning metadata. Gladiator does not inject provider-specific cache or routing controls.

## Sessions and caching

Gladiator keeps the system prompt, tool schema, and previous API-visible messages stable as a session grows. New information is appended at the tail instead of rewriting prior messages. Telegram/UI events, timestamps, reasoning traces, and `message.extra` metadata are not sent back to the model.

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
- Images can be supplied as multimodal input when the selected model supports vision.
- `gladiator send PATH` sends an image or file back through Telegram without exposing Telegram internals to the model.

## Skills

Skills are lazy external memory. The agent sees names first and reads one relevant skill at a time. Skill creation/update/delete is runtime-locked unless the current user request explicitly asks to create or change a skill; there is no automatic learning or automatic skill creation.

Agent-facing commands:

```bash
gladiator skill list
gladiator skill read NAME
gladiator skill write NAME SOURCE_PATH
gladiator skill delete NAME
```

Writes and deletes only succeed when the current user request explicitly authorizes a skill change.

## Development

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/invisiblestrangler/GladiatorAgent.git
cd GladiatorAgent
uv sync --extra dev
```

Run tests and lint checks:

```bash
uv run pytest
uv run ruff check src tests
```

The repository also includes a manual GitHub Actions Live E2E workflow for real model/Telegram testing. It is intentionally separate from normal CI so regular pull requests do not require provider or Telegram secrets.

## Status

GladiatorAgent currently includes the native Telegram command menu, supervised background-service support, Telegram progress/typing feedback, inline cancellation, strict local tool-transcript validation with terminal completion repair, provider error-body diagnostics, provider-agnostic OpenAI-compatible streaming, image/file transfer, provider/model/reasoning controls, context compaction, persistent sessions, provider-reported cache telemetry when available, the external TODO ledger, lazy explicit skills, SearXNG/web extraction, optional Browser Use installation, YOLO execution, and conservative one-hour escalation fallback.
