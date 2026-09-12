from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    returncode: int


class DockerManager:
    @staticmethod
    def available() -> bool:
        return shutil.which("docker") is not None

    @staticmethod
    def install() -> CommandResult:
        """Install Docker only after the setup wizard has obtained explicit consent."""
        if sys.platform == "darwin":
            if not shutil.which("brew"):
                raise RuntimeError("Homebrew is required for automatic Docker installation on macOS")
            command = ["brew", "install", "--cask", "docker"]
        elif sys.platform.startswith("linux"):
            if shutil.which("apt-get"):
                command = ["apt-get", "install", "-y", "docker.io"]
            elif shutil.which("dnf"):
                command = ["dnf", "install", "-y", "docker"]
            elif shutil.which("pacman"):
                command = ["pacman", "-S", "--noconfirm", "docker"]
            else:
                raise RuntimeError("Automatic Docker installation is not supported on this Linux distribution")
        else:
            raise RuntimeError(f"Automatic Docker installation is not supported on {sys.platform}")

        if sys.platform.startswith("linux") and os.geteuid() != 0:
            if not shutil.which("sudo"):
                raise RuntimeError("Docker installation requires root privileges or sudo")
            command = ["sudo", *command]
        proc = subprocess.run(command, check=False)
        if proc.returncode == 0 and sys.platform.startswith("linux"):
            starter = ["systemctl", "enable", "--now", "docker"] if shutil.which("systemctl") else ["service", "docker", "start"]
            if os.geteuid() != 0 and shutil.which("sudo"):
                starter = ["sudo", *starter]
            subprocess.run(starter, check=False)
        return CommandResult(command=command, returncode=proc.returncode)
