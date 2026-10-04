from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

from run import patch_thor_panda_py


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def deployment_tree(tmp_path):
    """Exercise the real deployment scripts without a network or services."""
    checkout = tmp_path / "checkout"
    (checkout / "run").mkdir(parents=True)
    for name in ("deploy.sh", "sync_to_target.sh", "setup_thor_fr3.sh", "setup_thor_spacemouse.sh"):
        shutil.copy2(REPO / "run" / name, checkout / "run" / name)
    (checkout / "run/start_host_fr3.sh").write_text(
        '#!/bin/bash\nprintf "host-fr3 start\\n" >> "$DEPLOY_TEST_LOG"\n'
    )
    vite = checkout / "tools/data_collection_gui/frontend/node_modules/.bin/vite"
    vite.parent.mkdir(parents=True)
    vite.write_text("#!/bin/sh\nexit 0\n")
    vite.chmod(0o755)
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "command.log"
    remote_script = tmp_path / "remote.sh"
    fakes = {
        "ssh": """#!/bin/bash
printf 'ssh %s\\n' "$*" >> "$DEPLOY_TEST_LOG"
if [[ "$*" == *"flock -n"* ]]; then
  cat > "$DEPLOY_REMOTE_SCRIPT"
  exit "${DEPLOY_RESTART_RC:-0}"
fi
printf '%s\\n' '---POINTERS---' '---WORLD---'
""",
        "rsync": "#!/bin/bash\nprintf 'rsync %s\\n' \"$*\" >> \"$DEPLOY_TEST_LOG\"\n",
        "npm": "#!/bin/bash\nprintf 'npm %s target=%s\\n' \"$*\" \"$GUI_API_TARGET\" >> \"$DEPLOY_TEST_LOG\"\n",
        "sleep": "#!/bin/bash\nprintf 'sleep %s\\n' \"$*\" >> \"$DEPLOY_TEST_LOG\"\n",
        "kill": "#!/bin/bash\nprintf 'kill %s\\n' \"$*\" >> \"$DEPLOY_TEST_LOG\"\n",
    }
    for name, content in fakes.items():
        path = commands / name
        path.write_text(content)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "DEPLOY_TEST_LOG": str(log),
        "DEPLOY_REMOTE_SCRIPT": str(remote_script),
    }
    # Calibration input availability is intentionally unrelated to this test.
    env.pop("REQUIRE_EE_CALIBRATION", None)
    return checkout, env, log, remote_script


def _deploy(tree, *args, restart_rc=0):
    checkout, env, log, script = tree
    result = subprocess.run(
        ["bash", str(checkout / "run/deploy.sh"), *args],
        cwd=checkout,
        env={**env, "DEPLOY_RESTART_RC": str(restart_rc)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, log.read_text() if log.exists() else "", script


def test_default_deploy_only_contacts_thor_and_starts_box_frontend(deployment_tree):
    result, commands, script = _deploy(deployment_tree)
    assert result.returncode == 0, result.stderr
    assert "nvidia@192.168.111.122" in commands
    assert "192.168.100.155" not in commands
    assert "hph@" not in commands
    assert "tools/thor/gmsl2/thor_fr3_teleop.yaml" in commands
    assert "npm run dev -- --host 0.0.0.0 --port 5173 target=http://192.168.111.122:8765" in commands
    assert "--check" not in script.read_text()
    assert "apt-get" not in commands + script.read_text()
    assert "uv sync" not in commands + script.read_text()
    assert "fr3_control_worker --" not in script.read_text()
    assert "host-fr3 start" in commands
    assert commands.index("flock -n") < commands.index("host-fr3 start") < commands.index("npm run")


def test_box_only_preserves_original_profile(deployment_tree):
    result, commands, _ = _deploy(deployment_tree, "--box-only", "--no-frontend")
    assert result.returncode == 0, result.stderr
    assert "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml" in commands
    assert "tools/thor/gmsl2/thor_fr3_teleop.yaml" not in commands
    assert "npm " not in commands
    assert "192.168.100.155" not in commands
    assert "host-fr3 start" not in commands


def test_sync_only_does_not_restart_or_start_robot(deployment_tree):
    result, commands, script = _deploy(deployment_tree, "--sync-only")
    assert result.returncode == 0, result.stderr
    assert "rsync " in commands
    assert "--delete-delay" in commands
    assert "flock" not in commands
    assert not script.exists()
    assert "npm " not in commands
    assert "host-fr3 start" not in commands


def test_restart_lock_conflict_remains_distinct(deployment_tree):
    result, commands, _ = _deploy(deployment_tree, "--no-frontend", restart_rc=75)
    assert result.returncode == 75
    assert "another deploy" in result.stderr
    assert "npm " not in commands


def test_box_only_rejects_explicit_legacy_workstation(deployment_tree):
    result, commands, _ = _deploy(deployment_tree, "workstation", "--box-only")
    assert result.returncode == 2
    assert "Thor deployment option" in result.stderr
    assert not commands


@pytest.mark.parametrize("helper", ["stop_pids", "stop_fr3_workers"])
def test_redeploy_allows_eight_seconds_for_owned_robot_shutdown(deployment_tree, helper):
    checkout, env, log, _ = deployment_tree
    deploy = (checkout / "run/deploy.sh").read_text()
    definitions = deploy[deploy.index("stop_pids() {"):deploy.index("# gateway.py redirects")]
    pid_source = "fake_pids" if helper == "stop_pids" else "fr3_worker_pids"
    script = "set -euo pipefail\n" + definitions + f"\n{pid_source}() {{ echo 999999; }}\n"
    script += "stop_pids fake_pids\n" if helper == "stop_pids" else "stop_fr3_workers\n"
    result = subprocess.run(
        ["bash", "-c", script], env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    events = log.read_text().splitlines()
    assert events[0] == "kill 999999"
    assert events[1:9] == ["sleep 1"] * 8
    assert events[-1] == "kill -9 999999"


def test_fr3_profile_keeps_box_camera_dataset_configuration_exactly():
    baseline = REPO / "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml"
    profile = REPO / "tools/thor/gmsl2/thor_fr3_teleop.yaml"
    assert profile.read_text().startswith(baseline.read_text())
    old = yaml.safe_load(baseline.read_text())
    new = yaml.safe_load(profile.read_text())
    assert {key: new[key] for key in old} == old
    assert new["sensors"]["cameras"]["detect_all"] is True
    assert new["sensors"]["cameras"]["sensor_ids"] == []
    assert "wrist_camera" not in new["fr3_teleop"]
    assert "arm_host" not in new["fr3_teleop"]
    assert new["fr3_teleop"]["runtime_python"] == ".venv-fr3/bin/python"
    assert new["fr3_teleop"]["command_timeout_s"] == 0.2
    urdf = ET.parse(REPO / new["robot"]["urdf_path"])
    assert any(link.attrib["name"] == new["robot"]["target_frame_name"] for link in urdf.findall("link"))
    assert new["robot"]["cameras"] == {}
    assert new["robot"]["use_otg"] is True
    assert new["robot"]["otg_min_position"] == [-2.64, -1.57, -2.70, -2.84, -2.70, 0.60, -2.70]
    assert new["robot"]["otg_max_position"] == [2.64, 1.57, 2.70, -0.27, 2.70, 3.65, 2.70]


def test_check_missing_fr3_runtime_leaves_sensor_workflow_available(deployment_tree):
    checkout, env, log, _ = deployment_tree
    result = subprocess.run(
        ["bash", str(checkout / "run/setup_thor_fr3.sh"), "--check"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 1
    assert "C and camera/BOX collection remain available" in result.stderr
    assert not log.exists()


def test_setup_requires_matching_wheel_before_any_install(deployment_tree):
    checkout, env, log, _ = deployment_tree
    result = subprocess.run(
        ["bash", str(checkout / "run/setup_thor_fr3.sh"), "--install-system-deps", "--install-python"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2
    assert "requires --panda-wheel" in result.stderr
    assert not log.exists()


def test_explicit_check_uses_local_native_preflight_without_ipc(deployment_tree):
    checkout, env, log, _ = deployment_tree
    runtime = checkout / ".venv-fr3/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text('#!/bin/bash\nprintf "python %s\\n" "$*" >> "$DEPLOY_TEST_LOG"\n')
    runtime.chmod(0o755)
    result = subprocess.run(
        ["bash", str(checkout / "run/setup_thor_fr3.sh"), "--check"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert "-m tools.thor.fr3_control_worker --check --config-path tools/thor/gmsl2/thor_fr3_teleop.yaml" in commands
    assert "--ipc-fd" not in commands
    assert "ssh " not in commands


def test_setup_rejects_unpatched_native_capability_before_worker_preflight(deployment_tree):
    checkout, env, log, _ = deployment_tree
    runtime = checkout / ".venv-fr3/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text('''#!/bin/bash
printf "python %s\\n" "$*" >> "$DEPLOY_TEST_LOG"
if [[ "$1" == "-" ]]; then
  input="$(cat)"
  if [[ "$input" == *FR3_NO_AUTOMATIC_ERROR_RECOVERY* ]]; then exit 1; fi
fi
''')
    runtime.chmod(0o755)
    result = subprocess.run(
        ["bash", str(checkout / "run/setup_thor_fr3.sh"), "--check"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 1
    assert "patch_thor_panda_py.py" in result.stderr
    assert "-m tools.thor.fr3_control_worker" not in log.read_text()


@pytest.mark.parametrize("modern_api", [True, False])
def test_spacemouse_component_check_supports_owning_device_api(deployment_tree, tmp_path, modern_api):
    checkout, env, log, _ = deployment_tree
    stub_root = tmp_path / "usb_stub"
    stub_root.mkdir()
    common = '''from pathlib import Path
import os
from types import SimpleNamespace
def record(name):
    with Path(os.environ["DEPLOY_TEST_LOG"]).open("a") as stream: stream.write(name + "\\n")
def state():
    return SimpleNamespace(x=0.25, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=[1,0], t=1.0)
'''
    if modern_api:
        api = '''def get_connected_devices():
    record("enumerate")
    return ["SpaceMouse Compact"]
class SpaceMouseDevice:
    def read(self):
        record("device.read")
        return state()
    def close(self): record("device.close")
def open():
    record("open")
    return SpaceMouseDevice()
'''
    else:
        api = '''def list_devices():
    record("enumerate")
    return ["SpaceMouse Compact"]
def open():
    record("open")
    return True
def read():
    record("module.read")
    return state()
def close(): record("module.close")
'''
    (stub_root / "pyspacemouse.py").write_text(common + api)
    result = subprocess.run(
        ["bash", str(checkout / "run/setup_thor_spacemouse.sh"), "--check"],
        env={**env, "THOR_PYTHON": sys.executable, "PYTHONPATH": str(stub_root)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "SpaceMouse Compact" in result.stdout
    assert "'x': 0.25" in result.stdout
    suffix = "device" if modern_api else "module"
    assert log.read_text().splitlines() == ["enumerate", "open", f"{suffix}.read", f"{suffix}.close"]


@pytest.fixture
def panda_source(tmp_path):
    source = tmp_path / "panda-py"
    (source / "src").mkdir(parents=True)
    (source / "src/panda.cpp").write_text("#include <memory>\n" + patch_thor_panda_py.ORIGINAL_RECOVER + "\nvoid Panda::stopController() {\n" + patch_thor_panda_py.ORIGINAL_STOP_STATE + "\n}\n")
    (source / "src/_core.cpp").write_text(patch_thor_panda_py.MODULE_DECLARATION + "\n" + patch_thor_panda_py.ORIGINAL_STOP_BINDING + "\n}\n")
    return source


def test_native_patch_removes_recovery_and_exports_compiled_capability(panda_source):
    assert patch_thor_panda_py.main([str(panda_source)]) == 0
    panda = (panda_source / "src/panda.cpp").read_text()
    core = (panda_source / "src/_core.cpp").read_text()
    assert "automaticErrorRecovery(" not in panda
    assert "throw std::runtime_error" in panda
    assert "state.current_errors" in panda
    assert "#include <stdexcept>" in panda
    assert patch_thor_panda_py.PATCHED_STOP_STATE in panda
    assert patch_thor_panda_py.CAPABILITY in core
    assert patch_thor_panda_py.PATCHED_STOP_BINDING in core
    assert patch_thor_panda_py.main(["--check", str(panda_source)]) == 0
    assert patch_thor_panda_py.main([str(panda_source)]) == 0
    assert (panda_source / "src/_core.cpp").read_text() == core


def test_native_patch_refuses_extra_native_recovery_without_editing(panda_source):
    (panda_source / "src/other.cpp").write_text("void recoverAnotherRobot() { robot.automaticErrorRecovery(); }\n")
    before = (panda_source / "src/panda.cpp").read_text()
    with pytest.raises(ValueError, match="Additional native automaticErrorRecovery"):
        patch_thor_panda_py.prepare_sources(panda_source)
    assert (panda_source / "src/panda.cpp").read_text() == before
    assert "FR3_NO_AUTOMATIC_ERROR_RECOVERY" not in (panda_source / "src/_core.cpp").read_text()


def test_native_patch_refuses_uninspected_recovery_implementation(panda_source):
    panda = panda_source / "src/panda.cpp"
    panda.write_text(panda.read_text().replace("auto state = robot_->readOnce();", "auto state = unknown_state();"))
    with pytest.raises(ValueError, match="Unknown Panda::recover"):
        patch_thor_panda_py.prepare_sources(panda_source)


def test_native_patch_refuses_capability_without_native_change(panda_source):
    core = panda_source / "src/_core.cpp"
    core.write_text(core.read_text().replace("{", "{\n" + patch_thor_panda_py.CAPABILITY, 1))
    with pytest.raises(ValueError, match="marker lacks|marker does not match"):
        patch_thor_panda_py.prepare_sources(panda_source)
