"""Monte02 URDF with the GMSL2 camera: generated file up to date and its frames equal to the FR3's.

Roadmap docs/monte02_left_arm_roadmap.html, M1.1/M1.2. The description lives in the private opencv_kalibr
submodule; without it these tests skip.
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DESC = REPO / "third_party" / "opencv_kalibr" / "monte02_h_v1_description"
FR3_URDF = REPO / "src" / "lerobot" / "robots" / "franka_research3" / "assets" / "franka_fr3" / "fr3_corenetic_gripper.urdf"

pytestmark = pytest.mark.skipif(not (DESC / "build_gmsl2cam_urdf.py").exists(), reason="opencv_kalibr submodule not checked out")


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("monte02_build", DESC / "build_gmsl2cam_urdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_urdf_passes_check(builder):
    assert builder.check(FR3_URDF, builder.OUT_URDF, ("left",)) == []


def test_checked_in_urdf_is_what_the_builder_makes(builder, tmp_path):
    out = tmp_path / "rebuilt.urdf"
    builder.build(FR3_URDF, ("left",)).write(out, encoding="utf-8", xml_declaration=True)
    assert out.read_bytes() == builder.OUT_URDF.read_bytes(), "rerun build_gmsl2cam_urdf.py"


def test_socket_point_in_sensor_frame(builder):
    import numpy as np

    out = ET.parse(builder.OUT_URDF).getroot()
    t_sensor_socket = builder._chain(out, "lt_corenetic_socket_ee", "link_l_sensor_ft")
    np.testing.assert_allclose(t_sensor_socket[:3, 3], [-0.00206, -0.00096, 0.18324], atol=1e-5)


def test_check_catches_a_moved_camera(builder, tmp_path):
    tree = builder.build(FR3_URDF, ("left",))
    for j in tree.getroot().findall("joint"):
        if j.get("name") == "joint_lt_corenetic_cam_frame":
            j.find("origin").set("xyz", "0 0 -0.03")
    bad = tmp_path / "bad.urdf"
    tree.write(bad, encoding="utf-8", xml_declaration=True)
    # meshes resolve relative to the URDF; point the check at a copy next to the real one
    sibling = builder.OUT_URDF.parent / "_test_bad.urdf"
    sibling.write_bytes(bad.read_bytes())
    try:
        problems = builder.check(FR3_URDF, sibling, ("left",))
    finally:
        sibling.unlink()
    assert any("camera bracket" in p for p in problems)
