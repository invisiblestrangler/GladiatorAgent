from __future__ import annotations

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
                command = ["sudo", "apt-get", "install", "-y", "docker.io", "docker-compose-v2"]
            elif shutil.which("dnf"):
                command = ["sudo", "dnf", "install", "-y", "docker", "docker-compose-plugin"]
            elif shutil.which("pacman"):
                command = ["sudo", "pacman", "-S", "--noconfirm", "docker", "docker-compose"]
            else:
                raise RuntimeError("Automatic Docker installation is not supported on this Linux distribution")
        else:
            raise RuntimeError(f"Automatic Docker installation is not supported on {sys.platform}")

        proc = subprocess.run(command, check=False)
        return CommandResult(command=command, returncode=proc.returncode)
