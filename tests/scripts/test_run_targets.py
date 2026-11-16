"""The deployment-target registry (run/targets/) and the identity check the run/ scripts make with it."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "run"

_spec = importlib.util.spec_from_file_location("run_targets", RUN / "targets.py")
targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(targets)


def _target(**overrides):
    base = {
        "station": "station9",
        "user": "nvidia",
        "address": "10.0.0.9",
        "hostname": "thor-nine",
        "repo_dir": "/home/nvidia/lerobot",
        "profile": "thor",
        "config": "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml",
        "gateway_port": 8765,
        "sandboxes": True,
        "calibration_in_tree": False,
        "box_net_addr": None,
    }
    base.update(overrides)
    return base


def _registry(tmp_path: Path, files: dict, local: dict | None = None):
    tdir = tmp_path / "targets"
    tdir.mkdir(parents=True)
    for name, body in files.items():
        (tdir / f"{name}.json").write_text(json.dumps(body))
    local_path = tmp_path / "targets.local.json"
    if local is not None:
        local_path.write_text(json.dumps(local))
    return targets.load_targets(tdir, local_path)


def test_committed_registry_is_valid_and_thor_still_means_thor1():
    reg = targets.load_targets(RUN / "targets", Path("/nonexistent/targets.local.json"))
    thor = targets.resolve("thor", reg)
    assert thor["name"] == "thor1" and thor["ssh"] == "nvidia@192.168.111.122"
    assert thor["calibration_in_tree"] is True
    # Only one machine may claim the legacy BOX address: two Thors on one LAN would fight over it.
    assert [n for n, t in reg.items() if t["box_net_addr"]] == ["thor1"]
    assert [n for n, t in reg.items() if t["calibration_in_tree"]] == ["thor1"]


def test_no_run_script_hard_codes_a_machine_address():
    for script in RUN.glob("*.sh"):
        assert "192.168." not in script.read_text(), f"{script.name}: addresses belong in run/targets/"


def test_two_targets_cannot_share_a_hostname(tmp_path):
    with pytest.raises(targets.TargetError, match="both claim hostname"):
        _registry(tmp_path, {"thor8": _target(hostname="upai"), "thor9": _target(hostname="upai")})


def test_a_name_and_an_alias_cannot_collide(tmp_path):
    with pytest.raises(targets.TargetError, match="names both"):
        _registry(tmp_path, {"thor8": _target(hostname="a", aliases=["thor9"]), "thor9": _target(hostname="b")})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"surprise": 1}, "unknown key"),
        ({"profile": "workstation", "sandboxes": True}, "sandboxes exist only"),
        ({"repo_dir": "relative/path"}, "absolute path"),
        ({"box_net_addr": "192.168.2.45"}, "box_net_addr"),
        ({"gateway_port": True}, "gateway_port"),
    ],
)
def test_bad_target_files_are_refused(tmp_path, change, message):
    with pytest.raises(targets.TargetError, match=message):
        _registry(tmp_path, {"thor9": _target(**change)})


def test_local_override_changes_only_how_a_machine_is_reached(tmp_path):
    reg = _registry(
        tmp_path,
        {"thor9": _target()},
        local={"thor9": {"ssh": "thor9-via-lab", "gateway_url": "http://127.0.0.1:18765/"}},
    )
    t = reg["thor9"]
    assert t["ssh"] == "thor9-via-lab" and t["gateway_url"] == "http://127.0.0.1:18765"
    assert t["hostname"] == "thor-nine" and t["overridden"] == ["ssh", "gateway_url"]
    with pytest.raises(targets.TargetError, match="may only override"):
        _registry(tmp_path / "x", {"thor9": _target()}, local={"thor9": {"hostname": "other"}})
    with pytest.raises(targets.TargetError, match="ssh destination"):
        _registry(tmp_path / "y", {"thor9": _target()}, local={"thor9": {"ssh": "host; rm -rf /"}})


def _fake_bin(tmp_path: Path, hostname: str) -> dict:
    """PATH with an ssh that answers `hostname` with HOSTNAME and an rsync that must never run."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ssh").write_text(f'#!/bin/sh\nfor a; do last="$a"; done\n[ "$last" = hostname ] && echo {hostname} && exit 0\nexit 9\n')
    (fake / "rsync").write_text(f'#!/bin/sh\ntouch {tmp_path}/rsync_ran\nexit 0\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    return {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}


def _identity(env: dict) -> subprocess.CompletedProcess:
    script = f'. "{RUN}/targets_common.sh"; resolve_target thor1; require_target_identity'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def test_identity_check_accepts_the_right_machine(tmp_path):
    proc = _identity(_fake_bin(tmp_path, "upai-pro03"))
    assert proc.returncode == 0, proc.stderr
    assert "thor1 (station1)" in proc.stdout


def test_deploy_to_a_machine_that_is_not_the_target_stops_before_rsync(tmp_path):
    env = _fake_bin(tmp_path, "some-clone")
    assert _identity(env).returncode == 1
    proc = subprocess.run(
        ["bash", str(RUN / "deploy.sh"), "thor1", "--sync-only"],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=REPO,
    )
    assert proc.returncode == 1
    assert "is the machine 'some-clone'" in proc.stderr
    assert not (tmp_path / "rsync_ran").exists()


def test_deploy_rejects_an_unknown_target_without_touching_the_network(tmp_path):
    proc = subprocess.run(
        ["bash", str(RUN / "deploy.sh"), "thor7"],
        env=_fake_bin(tmp_path, "upai-pro03"),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=REPO,
    )
    assert proc.returncode == 2 and "unknown deployment target 'thor7'" in proc.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["deploy.sh"],
        ["deploy.sh", "--no-frontend"],
        ["sync_to_target.sh", "-n"],
        ["sync_to_thor.sh"],
        ["kill_thor_gateway.sh", "--force"],
        ["sandbox.sh", "list"],
    ],
)
def test_no_script_picks_a_machine_by_default(tmp_path, argv):
    env = _fake_bin(tmp_path, "upai-pro03")
    proc = subprocess.run(
        ["bash", str(RUN / argv[0]), *argv[1:]],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=REPO,
    )
    assert proc.returncode == 2, proc.stderr
    assert "there is no default target" in proc.stderr and "thor2" in proc.stderr
    assert not (tmp_path / "rsync_ran").exists()
