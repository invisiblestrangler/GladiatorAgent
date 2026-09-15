from __future__ import annotations

import shlex
from pathlib import Path
from typing import Callable

from gladiator.events import AgentEvent, EventKind, EventSink, null_event_sink
from gladiator.mentor import MentorRequest
from gladiator.runtime.context import ObservationLimiter
from gladiator.runtime.decision import DecisionRequest, DecisionResult
from gladiator.skills import SkillManager
from gladiator.webtools import WebTools
from minisweagent.environments.local import LocalEnvironment

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DecisionHandler = Callable[[DecisionRequest], DecisionResult]
MentorHandler = Callable[[MentorRequest], str]


class GladiatorLocalEnvironment(LocalEnvironment):
    def __init__(
        self,
        *,
        output_dir: Path,
        workspace_root: Path,
        observation_char_limit: int = 12_000,
        event_sink: EventSink = null_event_sink,
        decision_handler: DecisionHandler | None = None,
        mentor_handler: MentorHandler | None = None,
        web_tools: WebTools | None = None,
        skill_manager: SkillManager | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.limiter = ObservationLimiter(observation_char_limit, output_dir)
        self.event_sink = event_sink
        self.workspace_root = workspace_root.resolve()
        self.decision_handler = decision_handler
        self.mentor_handler = mentor_handler
        self.web_tools = web_tools
        self.skill_manager = skill_manager
        self.skill_write_authorized = False

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = str(action.get("command", ""))
        special = (
            self._artifact_command(command, cwd=cwd)
            or self._decision_command(command)
            or self._mentor_command(command, cwd=cwd)
            or self._web_command(command)
            or self._skill_command(command, cwd=cwd)
        )
        if special is not None:
            return special

        self.event_sink(AgentEvent(EventKind.TOOL_STARTED, command, {"command": command}))
        result = super().execute(action, cwd=cwd, timeout=timeout)
        limited = self.limiter.limit(str(result.get("output", "")), label="shell")
        result["output"] = limited.text
        extra = result.setdefault("extra", {})
        if limited.saved_path:
            extra["full_output_path"] = str(limited.saved_path)
        extra["output_truncated"] = limited.truncated
        extra["original_output_chars"] = limited.original_chars
        self.event_sink(
            AgentEvent(
                EventKind.TOOL_FINISHED,
                data={
                    "command": command,
                    "returncode": result.get("returncode"),
                    "truncated": limited.truncated,
                },
            )
        )
        return result

    @staticmethod
    def _reserved_pair_index(args: list[str], operation: str) -> int | None:
        for index in range(max(0, len(args) - 1)):
            if args[index : index + 2] == ["gladiator", operation]:
                return index
        return None

    def _artifact_command(self, command: str, *, cwd: str) -> dict | None:
        try:
            args = shlex.split(command)
        except ValueError:
            return None
        pair_index = self._reserved_pair_index(args, "send")
        if pair_index is None:
            return None
        if pair_index != 0:
            return self._bad_artifact_command("gladiator send must be the sole command, not part of a compound shell command")
        if len(args) < 3:
            return self._bad_artifact_command("Usage: gladiator send PATH [PATH ...]")

        base = Path(cwd or self.config.cwd or self.workspace_root)
        paths: list[Path] = []
        for raw_path in args[2:]:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            try:
                path.relative_to(self.workspace_root)
            except ValueError:
                return {
                    "output": f"Refused to send file outside workspace root: {path}",
                    "returncode": 2,
                    "exception_info": "artifact path outside workspace",
                }
            if not path.is_file():
                return {
                    "output": f"File does not exist: {path}",
                    "returncode": 2,
                    "exception_info": "artifact not found",
                }
            paths.append(path)

        for path in paths:
            self.event_sink(
                AgentEvent(
                    EventKind.ARTIFACT_READY,
                    str(path),
                    {"path": str(path), "is_image": path.suffix.lower() in _IMAGE_EXTENSIONS},
                )
            )
        if len(paths) == 1:
            output = f"Queued file for user delivery: {paths[0]}"
        else:
            output = "Queued files for user delivery:\n" + "\n".join(f"- {path}" for path in paths)
        return {"output": output, "returncode": 0, "exception_info": ""}

    @staticmethod
    def _bad_artifact_command(message: str) -> dict:
        return {"output": message, "returncode": 2, "exception_info": "invalid artifact send command"}

    def _decision_command(self, command: str) -> dict | None:
        try:
            args = shlex.split(command)
        except ValueError:
            return None
        if len(args) < 3 or args[:2] != ["gladiator", "ask"]:
            return None

        question = ""
        reason = ""
        conservative = ""
        options: list[str] = []
        index = 2
        while index < len(args):
            key = args[index]
            if index + 1 >= len(args):
                return self._bad_decision_command(f"Missing value for {key}")
            value = args[index + 1]
            if key == "--question":
                question = value
            elif key == "--reason":
                reason = value
            elif key == "--option":
                options.append(value)
            elif key == "--conservative":
                conservative = value
            else:
                return self._bad_decision_command(f"Unknown ask argument: {key}")
            index += 2

        try:
            request = DecisionRequest(
                question=question,
                options=tuple(options),
                conservative_choice=conservative,
                reason=reason or "Agent reports exceptional uncertainty.",
            )
        except ValueError as exc:
            return self._bad_decision_command(str(exc))
        if len(options) < 2:
            return self._bad_decision_command("At least two --option values are required")
        if self.decision_handler is None:
            result = DecisionResult(choice=conservative, timed_out=True)
        else:
            result = self.decision_handler(request)
        prefix = "No user response before timeout; conservative choice" if result.timed_out else "User choice"
        return {
            "output": f"{prefix}: {result.choice}",
            "returncode": 0,
            "exception_info": "",
            "extra": {"decision_timed_out": result.timed_out, "decision_choice": result.choice},
        }

    def _mentor_command(self, command: str, *, cwd: str) -> dict | None:
        try:
            args = shlex.split(command)
        except ValueError:
            return None
        pair_index = self._reserved_pair_index(args, "mentor")
        if pair_index is None:
            return None
        if pair_index != 0:
            return self._bad_mentor_command("gladiator mentor must be the sole command")
        if len(args) < 4:
            return self._bad_mentor_command(
                "Usage: gladiator mentor --question QUESTION [--file PATH ...] [--log PATH ...]"
            )

        question = ""
        files: list[Path] = []
        logs: list[Path] = []
        base = Path(cwd or self.config.cwd or self.workspace_root)
        index = 2
        while index < len(args):
            key = args[index]
            if index + 1 >= len(args):
                return self._bad_mentor_command(f"Missing value for {key}")
            value = args[index + 1]
            if key == "--question":
                question = value.strip()
            elif key in {"--file", "--log"}:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = base / path
                path = path.resolve()
                (files if key == "--file" else logs).append(path)
            else:
                return self._bad_mentor_command(f"Unknown mentor argument: {key}")
            index += 2
        if not question:
            return self._bad_mentor_command("--question is required")
        if self.mentor_handler is None:
            return {
                "output": "Mentor is unavailable in this runtime.",
                "returncode": 2,
                "exception_info": "mentor unavailable",
            }

        self.event_sink(AgentEvent(EventKind.TOOL_STARTED, command, {"command": command, "mentor": True}))
        try:
            advice = self.mentor_handler(MentorRequest(question=question, files=tuple(files), logs=tuple(logs)))
        except Exception as exc:
            self.event_sink(
                AgentEvent(EventKind.TOOL_FINISHED, data={"command": command, "returncode": 1, "mentor": True})
            )
            return {
                "output": f"Mentor consultation failed: {exc}",
                "returncode": 1,
                "exception_info": type(exc).__name__,
            }
        self.event_sink(AgentEvent(EventKind.TOOL_FINISHED, data={"command": command, "returncode": 0, "mentor": True}))
        return {
            "output": f"<mentor_advice>\n{advice}\n</mentor_advice>",
            "returncode": 0,
            "exception_info": "",
            "extra": {"mentor_advice": True},
        }

    @staticmethod
    def _bad_mentor_command(message: str) -> dict:
        return {"output": f"Invalid gladiator mentor command: {message}", "returncode": 2, "exception_info": "invalid mentor request"}

    def _web_command(self, command: str) -> dict | None:
        try:
            args = shlex.split(command)
        except ValueError:
            return None
        if len(args) < 4 or args[:2] != ["gladiator", "web"]:
            return None
        if self.web_tools is None:
            return {"output": "Web tools are not configured.", "returncode": 2, "exception_info": "web disabled"}
        operation = args[2]
        try:
            if operation == "search":
                output = self.web_tools.search(" ".join(args[3:]))
            elif operation == "fetch" and len(args) == 4:
                output = self.web_tools.fetch(args[3])
            else:
                return {
                    "output": "Usage: gladiator web search QUERY | gladiator web fetch URL",
                    "returncode": 2,
                    "exception_info": "invalid web command",
                }
        except Exception as exc:
            return {"output": f"Web operation failed: {exc}", "returncode": 1, "exception_info": type(exc).__name__}
        return {"output": output, "returncode": 0, "exception_info": ""}

    def _skill_command(self, command: str, *, cwd: str) -> dict | None:
        try:
            args = shlex.split(command)
        except ValueError:
            return None
        if len(args) < 3 or args[:2] != ["gladiator", "skill"]:
            return None
        if self.skill_manager is None:
            return {"output": "Skills are unavailable.", "returncode": 2, "exception_info": "skills unavailable"}
        operation = args[2]
        try:
            if operation == "list" and len(args) == 3:
                names = self.skill_manager.list()
                output = "Available skills:\n" + "\n".join(f"- {name}" for name in names) if names else "No user-created skills."
            elif operation == "read" and len(args) == 4:
                output = self.skill_manager.read(args[3])
            elif operation == "write" and len(args) == 5:
                if not self.skill_write_authorized:
                    return {
                        "output": "Skill writes are locked. The current user turn did not explicitly request creating or changing a skill.",
                        "returncode": 3,
                        "exception_info": "skill write not user-authorized",
                    }
                base = Path(cwd or self.config.cwd or self.workspace_root)
                source = Path(args[4]).expanduser()
                if not source.is_absolute():
                    source = base / source
                source = source.resolve()
                try:
                    source.relative_to(self.workspace_root)
                except ValueError:
                    return {
                        "output": f"Skill source must be inside workspace: {source}",
                        "returncode": 2,
                        "exception_info": "skill source outside workspace",
                    }
                target = self.skill_manager.write_from_file(args[3], source)
                output = f"Skill saved: {target}"
            elif operation == "delete" and len(args) == 4:
                if not self.skill_write_authorized:
                    return {
                        "output": "Skill changes are locked. The current user turn did not explicitly request changing a skill.",
                        "returncode": 3,
                        "exception_info": "skill write not user-authorized",
                    }
                self.skill_manager.delete(args[3])
                output = f"Skill deleted: {args[3]}"
            else:
                return {
                    "output": "Usage: gladiator skill list | read NAME | write NAME SOURCE_PATH | delete NAME",
                    "returncode": 2,
                    "exception_info": "invalid skill command",
                }
        except Exception as exc:
            return {"output": f"Skill operation failed: {exc}", "returncode": 1, "exception_info": type(exc).__name__}
        return {"output": output, "returncode": 0, "exception_info": ""}

    @staticmethod
    def _bad_decision_command(message: str) -> dict:
        return {"output": f"Invalid gladiator ask command: {message}", "returncode": 2, "exception_info": "invalid decision request"}
