from __future__ import annotations

import json
from pathlib import Path

from gladiator.events import AgentEvent, EventKind, EventSink, null_event_sink
from gladiator.runtime.prompt import GLADIATOR_RUNTIME_POLICY
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded


SYSTEM_TEMPLATE = GLADIATOR_RUNTIME_POLICY + r"""

You can interact with the computer through the bash tool. Work autonomously until the user's task is complete.
Use targeted inspection rather than dumping large files. Execute one focused action at a time and verify changes with tests.
For lightweight public-web research, use `gladiator web search QUERY` and `gladiator web fetch URL` as sole bash commands. Prefer these over browser automation.
User-created skills are lazy external memory. Use `gladiator skill list` to see names and `gladiator skill read NAME` only when a listed skill is relevant. Never read all skills by default.
Only when the CURRENT user request explicitly asks you to create or change a skill may you write a skill: first create a SKILL.md candidate in the workspace, then call `gladiator skill write NAME PATH` as the sole bash command. The runtime enforces this permission.
To send a generated image or file to the user, call bash with `gladiator send PATH` as the sole command.
If and ONLY if you are exceptionally uncertain about a materially consequential choice after investigating, you may ask the user with a sole bash command of this form:
`gladiator ask --question '...' --option 'choice A' --option 'choice B' --conservative 'choice A' --reason 'why this cannot be resolved safely'`
The conservative value MUST be one of the options. The runtime may return it automatically if the user does not answer within the configured timeout.

To finish a task, call bash with a command whose FIRST output line is exactly:
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Any following output becomes the final message shown to the user. Markdown, including fenced code blocks, is allowed there.
"""

INSTANCE_TEMPLATE = r"""
User task:
{{ task }}

Work on this task autonomously. Investigate before making consequential assumptions. Do not ask for routine approvals.
""".strip()

COMPACTION_TEMPLATE = r"""
Context compaction is required before continuing. Create or overwrite this file now:
{path}

Write a concise but complete Markdown handoff for YOUR future self. Preserve only information that matters for continuing work:
- current user objective and explicit requirements
- important architectural decisions and constraints
- relevant repository/workspace state and files changed
- tests run and their results
- important errors, root causes, failed approaches, and successful fixes
- current provider/tool assumptions only if relevant
- exact unfinished work and next actions
- anything that must not be repeated or forgotten

Do not include generic narration, Telegram/UI state, typing/progress messages, or other transport metadata.
The file is working memory, not a transcript. Write it using a bash command, then continue only after it exists.
""".strip()


class GladiatorAgentConfig(AgentConfig):
    system_template: str = SYSTEM_TEMPLATE
    instance_template: str = INSTANCE_TEMPLATE
    cost_limit: float = 0.0
    context_after_compact_path: Path
    compact_threshold_tokens: int = 300_000
    compact_fraction_of_model_window: float = 0.82
    model_context_window: int | None = None
    max_compaction_steps: int = 4
    min_compaction_file_chars: int = 300


class GladiatorAgent(DefaultAgent):
    """Persistent mini-swe-agent with safe-boundary context compaction."""

    def __init__(self, *args, event_sink: EventSink = null_event_sink, config_class=GladiatorAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.event_sink = event_sink
        self._last_compact_at_call = -1
        self._force_compact_requested = False

    def request_compaction(self) -> None:
        self._force_compact_requested = True

    @property
    def effective_compact_threshold(self) -> int:
        configured = self.config.compact_threshold_tokens
        window = self.config.model_context_window
        if not window:
            return configured
        return min(configured, int(window * self.config.compact_fraction_of_model_window))

    def estimate_context_tokens(self) -> int:
        """Conservative provider-agnostic estimate; intentionally excludes message.extra/UI data."""
        api_messages = []
        for message in self.messages:
            if message.get("role") == "exit":
                continue
            clean = {key: value for key, value in message.items() if key != "extra"}
            api_messages.append(clean)
        chars = len(json.dumps(api_messages, ensure_ascii=False, default=str))
        return max(1, int(chars / 3.5))

    def _start_or_continue_task(
        self, task: str, *, image_paths: list[Path] | None = None, file_paths: list[Path] | None = None
    ) -> None:
        self.extra_template_vars["task"] = task
        while self.messages and self.messages[-1].get("role") == "exit":
            self.messages.pop()
        if not self.messages:
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(
                    role="user", content=self._user_content(self._initial_user_content(task), image_paths, file_paths)
                ),
            )
            return
        self.add_messages(
            self.model.format_message(
                role="user", content=self._user_content(f"New user request:\n{task}", image_paths, file_paths)
            )
        )

    @staticmethod
    def _user_content(
        text: str, image_paths: list[Path] | None, file_paths: list[Path] | None
    ) -> str | list[dict]:
        file_paths = file_paths or []
        image_paths = image_paths or []
        if file_paths:
            text += "\n\nUploaded files are available locally at:\n" + "\n".join(
                f"- {path.resolve()}" for path in file_paths
            )
        if not image_paths:
            return text
        content: list[dict] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append({"type": "gladiator_image_path", "path": str(path.resolve())})
        return content

    def _initial_user_content(self, task: str) -> str:
        rendered = self._render_template(self.config.instance_template)
        compact_path = self.config.context_after_compact_path
        if compact_path.exists() and compact_path.stat().st_size > 0:
            rendered += (
                "\n\nA previous compacted working-memory file exists. Before proceeding, read it with bash:\n"
                f"{compact_path}\nUse it only as prior task context; the current user request above has priority."
            )
        return rendered

    def run_task(
        self,
        task: str,
        *,
        image_paths: list[Path] | None = None,
        file_paths: list[Path] | None = None,
        **kwargs,
    ) -> dict:
        """Run one user task while preserving history across Telegram turns."""
        self.extra_template_vars |= kwargs
        self._start_or_continue_task(task, image_paths=image_paths, file_paths=file_paths)
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
                self._maybe_compact_at_safe_boundary()
            except FormatError as exc:
                self.cost += exc.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *exc.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*exc.messages)
            except (LimitsExceeded, TimeExceeded, InterruptAgentFlow) as exc:
                self.add_messages(*exc.messages)
            except Exception as exc:
                self.handle_uncaught_exception(exc)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages and self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def _maybe_compact_at_safe_boundary(self) -> None:
        current_tokens = self.estimate_context_tokens()
        if not self._force_compact_requested and current_tokens < self.effective_compact_threshold:
            return
        if self._last_compact_at_call == self.n_calls:
            return
        self._last_compact_at_call = self.n_calls
        self._force_compact_requested = False
        self._compact_context(current_tokens)

    def _compact_context(self, current_tokens: int) -> None:
        path = self.config.context_after_compact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        before_mtime = path.stat().st_mtime_ns if path.exists() else -1
        self.event_sink(
            AgentEvent(
                EventKind.COMPACTION_STARTED,
                f"Compacting context at ~{current_tokens:,} tokens",
                {"estimated_tokens": current_tokens, "path": str(path)},
            )
        )
        self.add_messages(
            self.model.format_message(role="user", content=COMPACTION_TEMPLATE.format(path=path.resolve()))
        )

        success = False
        for _ in range(self.config.max_compaction_steps):
            try:
                self.step()
            except FormatError as exc:
                self.add_messages(*exc.messages)
                continue
            except InterruptAgentFlow as exc:
                self.add_messages(*exc.messages)
            if path.exists():
                new_mtime = path.stat().st_mtime_ns
                if new_mtime != before_mtime and len(path.read_text(encoding="utf-8", errors="replace")) >= self.config.min_compaction_file_chars:
                    success = True
                    break

        if not success:
            self.event_sink(
                AgentEvent(
                    EventKind.WARNING,
                    "Context compaction was requested but contextAfterCompact was not written; keeping existing history.",
                )
            )
            return

        self.save(self.config.output_path, {"info": {"pre_compaction_estimated_tokens": current_tokens}})
        self.messages = [
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(
                role="user",
                content=(
                    "Context was compacted. Resume the unfinished work. First read the working-memory file with bash:\n"
                    f"{path.resolve()}\nDo not ask the user to repeat information preserved there."
                ),
            ),
        ]
        self.event_sink(
            AgentEvent(
                EventKind.COMPACTION_FINISHED,
                "Context compacted successfully",
                {"path": str(path), "estimated_tokens_before": current_tokens},
            )
        )
