#!/usr/bin/env python

"""Real replay drives the pivot insert's socket centre, not the vendor TCP.

The camera EE trajectories put their TCP at the socket of insert v2 (bundle 0929, d = 0 by
definition) and label it with corenetic_gripper_ee's axes. That socket sits 10.3 mm up the tool axis
from corenetic_gripper_ee: the laser tracker measured -10.7 mm on 10-12 and -9.9 mm on 10-13, comparing
replay with recording. Driving corenetic_gripper_ee onto the trajectory therefore placed every replayed
pose 10 mm too far along the tool. The real replay, its MuJoCo preview and the IK QC must all drive
the same socket frame, and must keep the EE axes so that no replayed orientation changes.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3"
URDF = ASSETS / "fr3_corenetic_gripper.urdf"
MJCF = ASSETS / "fr3_corenetic_gripper.xml"
REPLAY = REPO / "third_party/opencv_kalibr/fr3_data_collection_replay"
THOR_CONFIGS = (
    REPLAY / "replay_cube_pose_in_robot_base.thor.yaml",
    REPO / "third_party/opencv_kalibr/verification/verify_fr3_cube_pose_ik.thor.yaml",
)
SOCKET_IN_EE_M = np.array([0.0, 0.0, -0.0103])
PROBE_Q = (0.31, -0.52, 0.18, -2.05, 0.12, 1.83, 0.71)


@pytest.mark.parametrize("config", THOR_CONFIGS, ids=lambda p: p.name)
def test_thor_replay_and_ik_qc_drive_the_socket_frame(config):
    if not config.is_file():
        pytest.skip("opencv_kalibr submodule not checked out")
    assert yaml.safe_load(config.read_text())["robot"]["target_frame_name"] == "corenetic_socket_ee"


def test_urdf_and_mjcf_put_the_socket_on_the_ee_axes_10_3_mm_up_the_tool():
    pin = pytest.importorskip("pinocchio")
    mujoco = pytest.importorskip("mujoco")

    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q = pin.neutral(model)
    q[:7] = PROBE_Q
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ee = data.oMf[model.getFrameId("corenetic_gripper_ee")]
    socket = ee.actInv(data.oMf[model.getFrameId("corenetic_socket_ee")])
    np.testing.assert_allclose(socket.translation, SOCKET_IN_EE_M, atol=1e-9)
    np.testing.assert_allclose(socket.rotation, np.eye(3), atol=1e-9)

    sim = mujoco.MjModel.from_xml_path(str(MJCF))
    sim_data = mujoco.MjData(sim)
    for index, value in enumerate(PROBE_Q, start=1):
        sim_data.qpos[sim.jnt_qposadr[sim.joint(f"fr3_joint{index}").id]] = value
    mujoco.mj_forward(sim, sim_data)
    ee_id, socket_id = sim.body("corenetic_gripper_ee").id, sim.body("corenetic_socket_ee").id
    ee_rot = sim_data.xmat[ee_id].reshape(3, 3)
    np.testing.assert_allclose(ee_rot.T @ (sim_data.xpos[socket_id] - sim_data.xpos[ee_id]), SOCKET_IN_EE_M, atol=1e-9)
    np.testing.assert_allclose(sim_data.xmat[socket_id].reshape(3, 3), ee_rot, atol=1e-9)
    # and the same place in the world as the URDF says
    np.testing.assert_allclose(sim_data.xpos[socket_id], data.oMf[model.getFrameId("corenetic_socket_ee")].translation, atol=1e-6)


def _replay_module():
    path = REPLAY / "replay_cube_pose_in_robot_base.py"
    if not path.is_file():
        pytest.skip("opencv_kalibr submodule not checked out")
    pytest.importorskip("lerobot.robots.franka_research3")
    spec = importlib.util.spec_from_file_location("replay_cube_pose_in_robot_base_socket_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg(urdf: Path, frame: str):
    return SimpleNamespace(robot=SimpleNamespace(urdf_path=str(urdf), target_frame_name=frame))


def test_replay_contract_accepts_the_socket_frame_only_as_a_pure_translation(tmp_path):
    replay = _replay_module()
    replay._validate_replay_target_frame_contract(_cfg(URDF, "corenetic_socket_ee"))

    # A turned socket frame would rotate every replayed orientation by the same amount, silently.
    turned = tmp_path / "turned.urdf"
    turned.write_text(
        URDF.read_text().replace('<origin rpy="0 0 0" xyz="0 0 -0.0103" />', '<origin rpy="0 0 0.1" xyz="0 0 -0.0103" />')
    )
    with pytest.raises(ValueError, match="pure translation"):
        replay._validate_replay_target_frame_contract(_cfg(turned, "corenetic_socket_ee"))
