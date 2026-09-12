from __future__ import annotations

import shutil
import subprocess
import sys


def install_browser_use() -> None:
    """Install browser-use into Gladiator's current Python environment on explicit opt-in."""
    if uv := shutil.which("uv"):
        subprocess.run([uv, "pip", "install", "--python", sys.executable, "browser-use"], check=True)
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "browser-use"], check=True)
