#!/usr/bin/env python3
"""Build a dual-FR3 URDF from one FR3 + Corenetic gripper model.

The two fixed base poses are read from the P1 and P2 calibration JSON files.
Both calibration matrices have the convention::

    p_old_world = T_old_world_position_base @ p_position_base

Therefore each matrix can be used directly as the URDF fixed-joint origin from
the common ``world`` link to the corresponding prefixed single-arm ``base``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent


def find_repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root above {path}")


REPO_ROOT = find_repo_root(HERE)
DEFAULT_SINGLE_ARM = HERE / "fr3_corenetic_gripper.urdf"
DEFAULT_P1 = REPO_ROOT / "outputs/calibration_reports/P1_20260827/selected_calibration.json"
DEFAULT_P2 = REPO_ROOT / "outputs/calibration_reports/P2_20260827/selected_calibration.json"
DEFAULT_OUTPUT = HERE / "dual_fr3_corenetic_gripper_p1_p2.urdf"


def load_transform(path: Path, key: str) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload[key]
    matrix = item["matrix_4x4"] if isinstance(item, dict) else item
    matrix = [[float(value) for value in row] for row in matrix]
    validate_transform(matrix, f"{path}:{key}")
    return matrix


def validate_transform(matrix: list[list[float]], label: str) -> None:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError(f"{label} is not a 4x4 matrix")
    expected_last = [0.0, 0.0, 0.0, 1.0]
    if any(abs(a - b) > 1e-9 for a, b in zip(matrix[3], expected_last)):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            target = 1.0 if i == j else 0.0
            if abs(dot - target) > 1e-5:
                raise ValueError(f"{label} rotation is not orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError(f"{label} rotation determinant is {determinant}, expected +1")


def matrix_to_xyz_rpy(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    pitch = math.atan2(-rotation[2][0], math.hypot(rotation[0][0], rotation[1][0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return [matrix[0][3], matrix[1][3], matrix[2][3]], [roll, pitch, yaw]


def values_text(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def prefixed_copy(element: ET.Element, prefix: str) -> ET.Element:
    clone = copy.deepcopy(element)
    for item in clone.iter():
        if item.tag in {"link", "joint"} and "name" in item.attrib:
            item.set("name", prefix + item.get("name", ""))
        if item.tag in {"visual", "collision"} and item.get("name"):
            item.set("name", prefix + item.get("name", ""))
        if item.tag in {"parent", "child"} and "link" in item.attrib:
            item.set("link", prefix + item.get("link", ""))
        if item.tag == "mimic" and "joint" in item.attrib:
            item.set("joint", prefix + item.get("joint", ""))
    # MuJoCo derives geom names from URDF visual/collision names.  The source
    # contains several unnamed visuals on one link, which become duplicate
    # ``<link>_visual`` names.  Assign deterministic names to every geometry.
    for link in clone.iter("link"):
        link_name = link.get("name", "link")
        for geometry_tag in ("visual", "collision"):
            for index, geometry in enumerate(link.findall(geometry_tag)):
                geometry.set("name", f"{link_name}_{geometry_tag}_{index}")
    return clone


def fixed_world_joint(name: str, child: str, matrix: list[list[float]]) -> ET.Element:
    xyz, rpy = matrix_to_xyz_rpy(matrix)
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "world"})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": values_text(xyz), "rpy": values_text(rpy)})
    return joint


def validate_urdf(root: ET.Element, output_path: Path) -> dict[str, int]:
    links = [item.get("name", "") for item in root.findall("link")]
    joints = [item.get("name", "") for item in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError("Generated URDF contains duplicate link names")
    if len(joints) != len(set(joints)):
        raise ValueError("Generated URDF contains duplicate joint names")
    link_set = set(links)
    children: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')} is missing parent or child")
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        if parent_name not in link_set or child_name not in link_set:
            raise ValueError(f"Joint {joint.get('name')} references an unknown link")
        if child_name in children:
            raise ValueError(f"Link {child_name} has more than one parent joint")
        children[child_name] = parent_name
    roots = sorted(link_set - set(children))
    if roots != ["world"]:
        raise ValueError(f"Expected only world as the root link, got {roots}")
    joint_set = set(joints)
    for mimic in root.iter("mimic"):
        if mimic.get("joint") not in joint_set:
            raise ValueError(f"Mimic references unknown joint {mimic.get('joint')}")
    missing_meshes = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if "://" not in filename and not (output_path.parent / filename).is_file():
            missing_meshes.append(filename)
    if missing_meshes:
        raise ValueError(f"Missing mesh files: {sorted(set(missing_meshes))}")
    return {"links": len(links), "joints": len(joints), "meshes": len(list(root.iter("mesh")))}


def build(args: argparse.Namespace) -> dict[str, int]:
    source_root = ET.parse(args.single_arm_urdf).getroot()
    p1_transform = load_transform(args.p1_calibration, args.p1_key)
    p2_transform = load_transform(args.p2_calibration, args.p2_key)

    output_root = ET.Element("robot", {"name": args.output.stem})
    mujoco = source_root.find("mujoco")
    if mujoco is not None:
        mujoco_copy = copy.deepcopy(mujoco)
        compiler = mujoco_copy.find("compiler")
        if compiler is not None:
            # Mesh filenames in this URDF already include either ``assets/`` or
            # ``URDF_franka_corenetic_gripper/meshes/``.  Keeping the source
            # meshdir="assets" would make MuJoCo look for assets/assets/... .
            compiler.set("meshdir", ".")
            # Keep fixed alias links such as both gripper TCPs addressable in
            # MuJoCo instead of fusing them into their nearest moving body.
            compiler.set("fusestatic", "false")
        output_root.append(mujoco_copy)
    output_root.append(
        ET.Comment(
            " P1=right arm, P2=left arm. Fixed origins are calibrated in the common old_world frame. "
        )
    )
    ET.SubElement(output_root, "link", {"name": "world"})

    for position, prefix, transform in (
        ("p1", "p1_", p1_transform),
        ("p2", "p2_", p2_transform),
    ):
        output_root.append(fixed_world_joint(f"world_to_{position}_base", f"{prefix}base", transform))
        for item in source_root:
            if item.tag in {"link", "joint"}:
                output_root.append(prefixed_copy(item, prefix))

    ET.indent(output_root, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(args.output, encoding="utf-8", xml_declaration=True)
    return validate_urdf(ET.parse(args.output).getroot(), args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-arm-urdf", type=Path, default=DEFAULT_SINGLE_ARM)
    parser.add_argument("--p1-calibration", type=Path, default=DEFAULT_P1)
    parser.add_argument("--p2-calibration", type=Path, default=DEFAULT_P2)
    parser.add_argument("--p1-key", default="T_old_world_P1_base")
    parser.add_argument("--p2-key", default="T_old_world_P2_base")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = build(arguments)
    print(f"output={arguments.output}")
    print(json.dumps(summary, indent=2))
