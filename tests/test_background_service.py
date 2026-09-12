import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gladiator.setup.service import LAUNCHD_LABEL, launchd_plist, systemd_unit


def test_systemd_unit_restarts_and_keeps_workspace_and_path():
    unit = systemd_unit(
        executable=Path("/home/test/.local/bin/gladiator"),
        workspace=Path("/srv/My Project"),
        path_env="/home/test/.local/bin:/usr/bin:/bin",
    )
    assert "WorkingDirectory=/srv/My\\x20Project" in unit
    assert 'WorkingDirectory="/srv/My Project"' not in unit
    assert 'ExecStart="/home/test/.local/bin/gladiator" run --workspace "/srv/My Project"' in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=3" in unit
    assert "WantedBy=default.target" in unit
    assert 'Environment="PATH=/home/test/.local/bin:/usr/bin:/bin"' in unit


def test_root_systemd_unit_is_boot_enabled_system_service():
    unit = systemd_unit(
        executable=Path("/root/.local/bin/gladiator"),
        workspace=Path("/root/project"),
        path_env="/root/.local/bin:/usr/bin:/bin",
        user="root",
    )
    assert "WorkingDirectory=/root/project" in unit
    assert 'WorkingDirectory="/root/project"' not in unit
    assert "User=root" in unit
    assert "WantedBy=multi-user.target" in unit


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("systemd-analyze") is None,
    reason="systemd-analyze is only available on systemd Linux hosts",
)
def test_generated_systemd_unit_passes_systemd_analyze_verify(tmp_path):
    workspace = tmp_path / "Workspace With Spaces"
    workspace.mkdir()
    unit_path = tmp_path / "gladiator-agent.service"
    unit_path.write_text(
        systemd_unit(
            executable=Path("/usr/bin/true"),
            workspace=workspace,
            path_env="/usr/local/bin:/usr/bin:/bin",
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["systemd-analyze", "verify", str(unit_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_launchd_plist_runs_at_login_and_restarts_failed_process(tmp_path):
    payload = plistlib.loads(
        launchd_plist(
            executable=Path("/Users/test/.local/bin/gladiator"),
            workspace=Path("/Users/test/project"),
            path_env="/Users/test/.local/bin:/usr/bin:/bin",
            log_dir=tmp_path,
        )
    )
    assert payload["Label"] == LAUNCHD_LABEL
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ProgramArguments"] == [
        "/Users/test/.local/bin/gladiator",
        "run",
        "--workspace",
        "/Users/test/project",
    ]
