# Mentor mode

Mentor mode is an optional, rare second-model consultation path for Gladiator. It is **not** a second autonomous agent and does not replace Gladiator's execution harness.

The main agent remains responsible for inspecting the workspace, editing files, running commands, testing, and deciding what work to do next. The mentor receives one question plus only the files or logs the main agent explicitly selects for that consultation, returns advice, and disappears.

## Telegram controls

```text
/mentor
/mentor on
/mentor off
/mentor model MODEL_ID
/mentor model default
/mentor reasoning high
```

`/mentor` shows the current state. Mentor mode is disabled by default. If no mentor model override is configured, it uses the currently selected main model. `model default` returns to that behavior.

The mentor uses the currently active model transport. If Gladiator is using the normal OpenAI-compatible provider, mentor requests use that provider. If Gladiator is using direct Codex OAuth mode, mentor requests use the same direct Codex OAuth transport. The mentor model can still be selected independently with `/mentor model ...`.

## Agent invocation

The main agent has one small stable capability hint in its normal runtime policy. It may invoke the mentor only when it is genuinely stuck after meaningful investigation or when an unusually complex algorithm/design deserves a final expert review:

```bash
gladiator mentor --question 'Review the lock-free queue correctness and identify any ABA risk' \
  --file src/queue.py \
  --file tests/test_queue.py \
  --log .gladiator/tool-output/benchmark.log
```

The command must be a sole bash action. `--file` and `--log` may be repeated.

## Just-in-time context design

The mentor does not receive the main conversation, TODO ledger, Telegram state, tool history, or an automatic workspace dump. At consultation time Gladiator builds a one-shot request containing:

- a dedicated mentor-only advisory policy;
- the main agent's question;
- only explicitly selected files/logs, with per-file and total context bounds.

The mentor has no shell/tools and no autonomous loop. Its response is returned to the main agent as one normal tool observation. This keeps mentor-only restrictions out of the main model's stable prompt prefix and avoids paying context/cache costs on turns where the mentor is not used.

Mentor consultations are intentionally uncommon. They are for hard reasoning, root-cause help, algorithm/design review, complexity/performance review, or identifying the smallest next experiment when the main model is stuck—not routine implementation or grunt work.
