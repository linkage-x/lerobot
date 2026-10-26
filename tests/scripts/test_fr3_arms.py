"""fr3_arms (roadmap B2): one entry per FR3, and no default arm."""

import copy
import json

import numpy as np
import pytest

from tools.fr3 import fr3_arms as fa, fr3_kinematics as kin
from tools.thor.gmsl2 import embodiment


def test_the_registry_agrees_with_the_replay_yaml_and_the_embodiments() -> None:
    registry = fa.load()
    assert fa.in_service(registry) == ["right", "left"]
    dual = embodiment.load(embodiment.EMBODIMENT_DIR / "fr3_dual.json")["arms"]
    single = embodiment.load(embodiment.EMBODIMENT_DIR / "fr3_single.json")["arms"]
    for name, arm in registry["arms"].items():
        assert dual[name]["replay_config"] == arm["replay_config"]
        start = np.asarray(arm["start_joints_rad"])
        assert np.allclose(start, kin.Q_READY)  # what Panda.move_to_start() did before B2
        assert np.all(start > kin.Q_LOWER) and np.all(start < kin.Q_UPPER)
    # 10-26: the registered arm (fr3_base) is the left one; the new one on its right has a bare flange
    assert registry["arms"]["left"]["robot_ip"] == "192.168.11.102" and dual["left"]["root"] == "fr3_base"
    assert single["left"]["replay_config"] == registry["arms"]["left"]["replay_config"]
    assert registry["arms"]["right"]["corenetic_device_id"] is None


def test_a_registry_that_disagrees_with_its_yaml_is_refused() -> None:
    registry = json.loads((fa._REPO_ROOT / fa.REGISTRY).read_text())
    wrong = copy.deepcopy(registry)
    wrong["arms"]["left"]["corenetic_device_id"] = 1819152274  # the handheld BOX (10-09)
    assert any("corenetic_device_id" in p and "says 596523097" in p for p in fa.validate(wrong))
    twins = copy.deepcopy(registry)
    twins["arms"]["right"] = dict(twins["arms"]["left"])
    assert any(p.startswith("two arms share robot_ip") for p in fa.validate(twins))
    silent = copy.deepcopy(registry)
    silent["arms"]["left"].update(in_service=False, why_not="")
    assert "arm left: not in service and no why_not" in fa.validate(silent)
    bare = copy.deepcopy(registry)
    bare["arms"]["right"]["no_gripper_why"] = ""
    assert "arm right: no gripper BOX (corenetic_device_id null) needs no_gripper_why" in fa.validate(bare)


def test_there_is_no_default_arm() -> None:
    registry = fa.load()
    with pytest.raises(
        fa.ArmConfigError, match="choose an arm \\(in service: right, left\\); there is no default arm"
    ):
        fa.resolve(registry)
    retired = copy.deepcopy(registry)
    retired["arms"]["right"].update(in_service=False, why_not="B1: not cabled")
    with pytest.raises(fa.ArmConfigError, match="right arm is not in service: B1"):
        fa.resolve(retired, "right")
    with pytest.raises(fa.ArmConfigError, match="is no FR3"):
        fa.resolve(registry, robot_ip="192.168.1.208")
    with pytest.raises(fa.ArmConfigError, match="the left arm is 192.168.11.102, not 192.168.11.103"):
        fa.resolve(registry, "left", "192.168.11.103")
    assert fa.resolve(registry, robot_ip="192.168.11.102")["arm"] == "left"
    assert fa.resolve(registry, robot_ip="192.168.1.210")["arm"] == "right"
    assert fa.resolve(registry, "LEFT")["corenetic_device_id"] == 596523097


def test_the_payload_check_compares_desk_with_the_declared_load() -> None:
    entry = {"arm": "right", "payload": None}
    ok, details = fa.payload_check(entry, 1.1108, [0.0168, -0.0222, 0.0609])
    assert ok and "Desk holds m_ee 1.111 kg, F_x_Cee (16.8, -22.2, 60.9) mm; no payload declared" in details
    entry["payload"] = {"mass_kg": 1.11, "com_m": [0.017, -0.022, 0.061], "source": "test"}
    assert fa.payload_check(entry, 1.1108, [0.0168, -0.0222, 0.0609])[0]
    ok, details = fa.payload_check(entry, 0.73, [0.0, 0.0, 0.04])  # the gripper taken off, Desk not updated
    assert not ok and "does NOT match" in details
