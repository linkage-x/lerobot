"""fr3_arms (roadmap B2): one entry per FR3, and no default arm."""

import copy
import json

import numpy as np
import pytest

from tools.fr3 import fr3_arms as fa, fr3_kinematics as kin
from tools.thor.gmsl2 import embodiment


def test_the_registry_agrees_with_the_replay_yaml_and_the_embodiments() -> None:
    registry = fa.load()
    assert fa.in_service(registry) == ["right"]
    right = registry["arms"]["right"]
    for name in ("fr3_single", "fr3_dual"):
        arms = embodiment.load(embodiment.EMBODIMENT_DIR / f"{name}.json")["arms"]
        assert arms["right"]["replay_config"] == right["replay_config"]
    start = np.asarray(right["start_joints_rad"])
    assert np.allclose(start, kin.Q_READY)  # what Panda.move_to_start() did before B2
    assert np.all(start > kin.Q_LOWER) and np.all(start < kin.Q_UPPER)


def test_a_registry_that_disagrees_with_its_yaml_is_refused() -> None:
    registry = json.loads((fa._REPO_ROOT / fa.REGISTRY).read_text())
    wrong = copy.deepcopy(registry)
    wrong["arms"]["right"]["corenetic_device_id"] = 1819152274  # the handheld BOX (10-09)
    assert any("corenetic_device_id" in p and "says 596523097" in p for p in fa.validate(wrong))
    twins = copy.deepcopy(registry)
    twins["arms"]["left"] = dict(twins["arms"]["right"])
    assert any(p.startswith("two arms share robot_ip") for p in fa.validate(twins))
    silent = copy.deepcopy(registry)
    silent["arms"]["left"]["why_not"] = ""
    assert "arm left: not in service and no why_not" in fa.validate(silent)


def test_there_is_no_default_arm() -> None:
    registry = fa.load()
    with pytest.raises(fa.ArmConfigError, match="choose an arm \\(in service: right\\); there is no default arm"):
        fa.resolve(registry)
    with pytest.raises(fa.ArmConfigError, match="left arm is not in service: B1"):
        fa.resolve(registry, "left")
    with pytest.raises(fa.ArmConfigError, match="is no FR3"):
        fa.resolve(registry, robot_ip="192.168.1.208")
    with pytest.raises(fa.ArmConfigError, match="the right arm is 192.168.11.102, not 192.168.11.103"):
        fa.resolve(registry, "right", "192.168.11.103")
    assert fa.resolve(registry, robot_ip="192.168.11.102")["arm"] == "right"
    assert fa.resolve(registry, "RIGHT")["corenetic_device_id"] == 596523097


def test_the_payload_check_compares_desk_with_the_declared_load() -> None:
    entry = {"arm": "right", "payload": None}
    ok, details = fa.payload_check(entry, 1.1108, [0.0168, -0.0222, 0.0609])
    assert ok and "Desk holds m_ee 1.111 kg, F_x_Cee (16.8, -22.2, 60.9) mm; no payload declared" in details
    entry["payload"] = {"mass_kg": 1.11, "com_m": [0.017, -0.022, 0.061], "source": "test"}
    assert fa.payload_check(entry, 1.1108, [0.0168, -0.0222, 0.0609])[0]
    ok, details = fa.payload_check(entry, 0.73, [0.0, 0.0, 0.04])  # the gripper taken off, Desk not updated
    assert not ok and "does NOT match" in details
