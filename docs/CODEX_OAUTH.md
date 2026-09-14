# Codex subscription OAuth

Gladiator can use a Codex subscription directly while keeping Gladiator/mini-swe as the only agent harness.

This mode does **not** launch Codex CLI, Codex app-server, or another coding-agent loop. Only the model transport changes. Codex function calls are translated back into Gladiator's existing `bash` tool contract, so `GladiatorAgent`, `GladiatorLocalEnvironment`, TODOs, goals, skills, bounded observations, compaction, Telegram progress, and artifact delivery continue to work normally.

## Connect from Telegram

Send the bot:

```text
/codex connect YOUR_OAUTH_ACCESS_TOKEN
```

Gladiator tries to read the ChatGPT account/workspace id from the JWT claims. If the token does not carry a usable account id, provide it explicitly:

```text
/codex connect YOUR_OAUTH_ACCESS_TOKEN YOUR_ACCOUNT_ID
```

The bot attempts to delete the Telegram message containing the token immediately after reading it. The token is then stored only in Gladiator's local configuration, which Gladiator writes with user-only file permissions where supported. The token is never inserted into agent/model context, status output, trajectories, or model serialization.

Use:

```text
/codex
```

to see whether a Codex credential is stored and whether the direct Codex transport is active.

## Switch back to the API provider

```text
/codex off
```

This switches back to the previously configured OpenAI-compatible endpoint and API key while keeping the Codex credential available for later reuse.

To remove the locally stored Codex credential as well:

```text
/codex clear
```

Supplying a new `/provider ENDPOINT API_KEY` also activates the ordinary OpenAI-compatible transport.

## Models and reasoning

The existing commands still control the active transport:

```text
/model MODEL_ID
/reasoning high
```

Available reasoning strings in Gladiator remain `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`. The selected Codex model/account ultimately determines which model and reasoning combinations are accepted.

## Transport behavior

The Codex subscription mode sends Responses-style streaming requests directly to the ChatGPT Codex responses endpoint using the OAuth bearer token plus the ChatGPT account id. It uses `store: false` and carries returned encrypted reasoning items forward locally when needed for subsequent tool turns.

Gladiator still owns the conversation, tool execution, local history, retry policy, cancellation, compaction, TODO/goal state, and Telegram UI. There is no second harness controlling the workspace.

## Token lifetime

The current integration accepts an **OAuth access token**. It does not store or automatically use an OAuth refresh token. If the access token expires or is rejected, reconnect with a fresh token using `/codex connect ...`.

Because Telegram message deletion is best-effort, treat a pasted OAuth token as a secret even though Gladiator attempts to remove the source message immediately. If deletion fails because of Telegram permissions or transport errors, delete that message manually.
