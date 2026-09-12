from __future__ import annotations

import shutil
import subprocess


def install_browser_use() -> str:
    """Install browser-use as an isolated uv tool, then install its local Chromium runtime.

    Keeping it outside Gladiator's Python environment prevents browser-use's large,
    tightly pinned dependency set from destabilizing mini-swe-agent.
    Returns the command Gladiator should use later.
    """
    uv = shutil.which("uv")
    uvx = shutil.which("uvx")
    if not uv or not uvx:
        raise RuntimeError("Optional browser-use setup requires uv/uvx so it can be installed in an isolated environment")
    subprocess.run([uv, "tool", "install", "--force", "browser-use"], check=True)
    command = shutil.which("browser-use")
    if command:
        subprocess.run([command, "install"], check=True)
        return command
    subprocess.run([uvx, "browser-use", "install"], check=True)
    return f"{uvx} browser-use"
