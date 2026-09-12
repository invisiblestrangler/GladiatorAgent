import plistlib
from pathlib import Path

from gladiator.setup.service import LAUNCHD_LABEL, launchd_plist, systemd_unit


def test_systemd_unit_restarts_and_keeps_workspace_and_path():
    unit = systemd_unit(
        executable=Path("/home/test/.local/bin/gladiator"),
        workspace=Path("/srv/My Project"),
        path_env="/home/test/.local/bin:/usr/bin:/bin",
    )
    assert 'WorkingDirectory="/srv/My Project"' in unit
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
    assert "User=root" in unit
    assert "WantedBy=multi-user.target" in unit


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
