from __future__ import annotations

import getpass
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SERVICE_NAME = "gladiator-agent"
LAUNCHD_LABEL = "xyz.gladiator.agent"


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def _systemd_escape(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def systemd_unit(*, executable: Path, workspace: Path, path_env: str, user: str | None = None) -> str:
    lines = [
        "[Unit]",
        "Description=GladiatorAgent Telegram coding agent",
        "Wants=network-online.target",
        "After=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={_systemd_escape(str(workspace))}",
        f"ExecStart={_systemd_escape(str(executable))} run --workspace {_systemd_escape(str(workspace))}",
        "Restart=on-failure",
        "RestartSec=3",
        "TimeoutStopSec=30",
        f"Environment={_systemd_escape('PATH=' + path_env)}",
        "Environment=PYTHONUNBUFFERED=1",
    ]
    if user:
        lines.append(f"User={user}")
    lines.extend(["", "[Install]", "WantedBy=multi-user.target" if user else "WantedBy=default.target", ""])
    return "\n".join(lines)


def launchd_plist(*, executable: Path, workspace: Path, path_env: str, log_dir: Path) -> bytes:
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [str(executable), "run", "--workspace", str(workspace)],
        "WorkingDirectory": str(workspace),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 3,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PATH": path_env, "PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(log_dir / "service.log"),
        "StandardErrorPath": str(log_dir / "service.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


@dataclass(slots=True)
class ServiceStatus:
    installed: bool
    active: bool
    enabled: bool
    detail: str


class BackgroundServiceManager:
    def __init__(self, workspace: Path, *, executable: Path | None = None):
        self.workspace = workspace.expanduser().resolve()
        found = shutil.which("gladiator")
        resolved = executable or (Path(found) if found else None)
        if resolved is None:
            resolved = Path(sys.argv[0])
        self.executable = resolved.expanduser().resolve()
        self.path_env = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

    @property
    def platform(self) -> str:
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "darwin":
            return "macos"
        return "unsupported"

    @property
    def is_root(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0

    @property
    def linux_user_unit(self) -> Path:
        return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"

    @property
    def linux_system_unit(self) -> Path:
        return Path("/etc/systemd/system") / f"{SERVICE_NAME}.service"

    @property
    def launchd_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

    @property
    def launchd_domain(self) -> str:
        return f"gui/{os.getuid()}"

    @property
    def launchd_target(self) -> str:
        return f"{self.launchd_domain}/{LAUNCHD_LABEL}"

    def install(self) -> str:
        if not self.workspace.is_dir():
            raise RuntimeError(f"Workspace does not exist: {self.workspace}")
        if not self.executable.exists():
            raise RuntimeError(f"Gladiator executable was not found: {self.executable}")
        if self.platform == "linux":
            return self._install_linux()
        if self.platform == "macos":
            return self._install_macos()
        raise RuntimeError("Background service installation is currently supported on Linux and macOS.")

    def start(self) -> str:
        if self.platform == "linux":
            result = _run(self._systemctl_args("start"))
            return result.stdout.strip() or "Gladiator service started."
        if self.platform == "macos":
            if not self.launchd_path.exists():
                raise RuntimeError("Gladiator launchd service is not installed.")
            _run(["launchctl", "bootstrap", self.launchd_domain, str(self.launchd_path)], check=False)
            _run(["launchctl", "kickstart", "-k", self.launchd_target])
            return "Gladiator service started."
        raise RuntimeError("Unsupported platform.")

    def stop(self) -> str:
        if self.platform == "linux":
            result = _run(self._systemctl_args("stop"))
            return result.stdout.strip() or "Gladiator service stopped."
        if self.platform == "macos":
            if not self.launchd_path.exists():
                return "Gladiator service is not installed."
            _run(["launchctl", "bootout", self.launchd_domain, str(self.launchd_path)], check=False)
            return "Gladiator service stopped."
        raise RuntimeError("Unsupported platform.")

    def restart(self) -> str:
        if self.platform == "linux":
            result = _run(self._systemctl_args("restart"))
            return result.stdout.strip() or "Gladiator service restarted."
        if self.platform == "macos":
            if not self.launchd_path.exists():
                raise RuntimeError("Gladiator launchd service is not installed.")
            _run(["launchctl", "bootout", self.launchd_domain, str(self.launchd_path)], check=False)
            _run(["launchctl", "bootstrap", self.launchd_domain, str(self.launchd_path)])
            _run(["launchctl", "kickstart", "-k", self.launchd_target])
            return "Gladiator service restarted."
        raise RuntimeError("Unsupported platform.")

    def uninstall(self) -> str:
        if self.platform == "linux":
            unit = self.linux_system_unit if self.is_root else self.linux_user_unit
            _run(self._systemctl_args("disable", "--now"), check=False)
            try:
                unit.unlink()
            except FileNotFoundError:
                pass
            _run(self._systemctl_args("daemon-reload"), check=False)
            return f"Removed Gladiator service: {unit}"
        if self.platform == "macos":
            _run(["launchctl", "bootout", self.launchd_domain, str(self.launchd_path)], check=False)
            try:
                self.launchd_path.unlink()
            except FileNotFoundError:
                pass
            return f"Removed Gladiator service: {self.launchd_path}"
        raise RuntimeError("Unsupported platform.")

    def status(self) -> ServiceStatus:
        if self.platform == "linux":
            unit = self.linux_system_unit if self.is_root else self.linux_user_unit
            active = _run(self._systemctl_args("is-active"), check=False)
            enabled = _run(self._systemctl_args("is-enabled"), check=False)
            detail = _run(self._systemctl_args("status", "--no-pager", "--lines=5"), check=False)
            return ServiceStatus(
                installed=unit.exists(),
                active=active.returncode == 0 and active.stdout.strip() == "active",
                enabled=enabled.returncode == 0 and enabled.stdout.strip() == "enabled",
                detail=(detail.stdout or detail.stderr).strip(),
            )
        if self.platform == "macos":
            result = _run(["launchctl", "print", self.launchd_target], check=False)
            return ServiceStatus(
                installed=self.launchd_path.exists(),
                active=result.returncode == 0,
                enabled=self.launchd_path.exists(),
                detail=(result.stdout or result.stderr).strip(),
            )
        raise RuntimeError("Unsupported platform.")

    def _install_linux(self) -> str:
        if shutil.which("systemctl") is None:
            raise RuntimeError("systemd/systemctl is not available on this Linux host.")

        if self.is_root:
            unit_path = self.linux_system_unit
            unit = systemd_unit(
                executable=self.executable,
                workspace=self.workspace,
                path_env=self.path_env,
                user=getpass.getuser(),
            )
            unit_path.write_text(unit, encoding="utf-8")
            _run(["systemctl", "daemon-reload"])
            _run(["systemctl", "enable", "--now", f"{SERVICE_NAME}.service"])
            return (
                f"Installed {unit_path}. Gladiator is running in the background, enabled at boot, "
                "and configured to restart after failures."
            )

        unit_path = self.linux_user_unit
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(
            systemd_unit(executable=self.executable, workspace=self.workspace, path_env=self.path_env),
            encoding="utf-8",
        )
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME}.service"])

        user = getpass.getuser()
        linger = _run(["loginctl", "show-user", user, "-p", "Linger", "--value"], check=False)
        if linger.stdout.strip().lower() != "yes":
            enabled = _run(["loginctl", "enable-linger", user], check=False)
            if enabled.returncode != 0:
                raise RuntimeError(
                    "The service is running, but systemd lingering could not be enabled. "
                    f"Run `sudo loginctl enable-linger {shlex.quote(user)}` so Gladiator starts after reboot "
                    "without an interactive login."
                )
        return (
            f"Installed {unit_path}. Gladiator is running in the background, enabled at boot via systemd lingering, "
            "and configured to restart after failures."
        )

    def _install_macos(self) -> str:
        log_dir = Path.home() / "Library" / "Logs" / "GladiatorAgent"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.launchd_path.parent.mkdir(parents=True, exist_ok=True)
        self.launchd_path.write_bytes(
            launchd_plist(
                executable=self.executable,
                workspace=self.workspace,
                path_env=self.path_env,
                log_dir=log_dir,
            )
        )
        _run(["launchctl", "bootout", self.launchd_domain, str(self.launchd_path)], check=False)
        _run(["launchctl", "bootstrap", self.launchd_domain, str(self.launchd_path)])
        _run(["launchctl", "enable", self.launchd_target])
        _run(["launchctl", "kickstart", "-k", self.launchd_target])
        return (
            f"Installed {self.launchd_path}. Gladiator runs in the background, restarts after failures, "
            "and starts automatically when this macOS user logs in."
        )

    def _systemctl_args(self, action: str, *extra: str) -> list[str]:
        prefix = ["systemctl"] if self.is_root else ["systemctl", "--user"]
        if action == "daemon-reload":
            return prefix + [action]
        return prefix + [action, f"{SERVICE_NAME}.service", *extra]
