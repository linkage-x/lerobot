#!/usr/bin/env python3
"""Extract calibrated P1 and P2 single-arm URDFs from a dual-arm URDF."""

from __future__ import annotations

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "dual_fr3_corenetic_gripper_v2_p1_p2.urdf"
DEFAULT_P0_OUTPUT = HERE / "fr3_corenetic_gripper_v2_p0.urdf"
DEFAULT_P1_OUTPUT = HERE / "fr3_corenetic_gripper_v2_p1.urdf"
DEFAULT_P2_OUTPUT = HERE / "fr3_corenetic_gripper_v2_p2.urdf"
DEFAULT_P0_CALIBRATION = (
    HERE.parents[5]
    / "outputs"
    / "calibration_reports"
    / "single_camera_latest_production"
    / "selected_calibration.json"
)


def load_transform(path: Path, key: str) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get(key)
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ValueError(f"{path} has no 4x4 transform at {key!r}")
    values = [[float(value) for value in row] for row in matrix]
    if any(len(row) != 4 for row in values):
        raise ValueError(f"{path}:{key} is not a 4x4 transform")
    if any(abs(values[3][index] - expected) > 1e-9 for index, expected in enumerate((0, 0, 0, 1))):
        raise ValueError(f"{path}:{key} has an invalid homogeneous last row")
    return values


def matrix_xyz_rpy(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    """Convert a homogeneous transform to URDF xyz and fixed-axis roll/pitch/yaw."""
    xyz = [matrix[index][3] for index in range(3)]
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return xyz, [roll, pitch, yaw]


def values_text(values: list[float]) -> str:
    return " ".join(format(float(value), ".16g") for value in values)


def strip_prefix(element: ET.Element, prefix: str) -> ET.Element:
    clone = copy.deepcopy(element)
    for item in clone.iter():
        if item.tag in {"link", "joint", "visual", "collision"} and item.get("name", "").startswith(prefix):
            item.set("name", item.get("name", "").removeprefix(prefix))
        if item.tag in {"parent", "child"} and item.get("link", "").startswith(prefix):
            item.set("link", item.get("link", "").removeprefix(prefix))
        if item.tag == "mimic" and item.get("joint", "").startswith(prefix):
            item.set("joint", item.get("joint", "").removeprefix(prefix))
    return clone


def validate(root: ET.Element, output: Path) -> dict[str, int]:
    links = [item.get("name", "") for item in root.findall("link")]
    joints = [item.get("name", "") for item in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError("Duplicate link names in extracted URDF")
    if len(joints) != len(set(joints)):
        raise ValueError("Duplicate joint names in extracted URDF")

    link_set = set(links)
    joint_set = set(joints)
    children: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')} has incomplete references")
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        if parent_name not in link_set or child_name not in link_set:
            raise ValueError(f"Joint {joint.get('name')} references an absent link")
        if child_name in children:
            raise ValueError(f"Link {child_name} has more than one parent")
        children.add(child_name)
    if sorted(link_set - children) != ["world"]:
        raise ValueError(f"Expected world as the only root, got {sorted(link_set - children)}")

    for mimic in root.iter("mimic"):
        if mimic.get("joint") not in joint_set:
            raise ValueError(f"Mimic references absent joint {mimic.get('joint')}")
    missing_meshes = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if "://" not in filename and not (output.parent / filename).is_file():
            missing_meshes.append(filename)
    if missing_meshes:
        raise ValueError(f"Missing meshes: {sorted(set(missing_meshes))}")
    return {
        "links": len(links),
        "joints": len(joints),
        "meshes": len(list(root.iter("mesh"))),
    }


def extract(
    source_root: ET.Element,
    side: str,
    output: Path,
    *,
    position_label: str | None = None,
    base_transform: list[list[float]] | None = None,
) -> dict[str, int | str]:
    prefix = f"{side}_"
    position = position_label or side
    output_root = ET.Element("robot", {"name": output.stem})
    mujoco = source_root.find("mujoco")
    if mujoco is not None:
        output_root.append(copy.deepcopy(mujoco))
    output_root.append(
        ET.Comment(
            f" Extracted from {DEFAULT_INPUT.name}; {position.upper()} base pose is expressed in the common world frame. "
        )
    )
    ET.SubElement(output_root, "link", {"name": "world"})

    world_joint_name = f"world_to_{side}_base"
    matching = [joint for joint in source_root.findall("joint") if joint.get("name") == world_joint_name]
    if len(matching) != 1:
        raise ValueError(f"Expected one {world_joint_name}, got {len(matching)}")
    world_joint = strip_prefix(matching[0], prefix)
    world_joint.set("name", "world_to_base")
    if base_transform is not None:
        xyz, rpy = matrix_xyz_rpy(base_transform)
        origin = world_joint.find("origin")
        if origin is None:
            origin = ET.SubElement(world_joint, "origin")
        origin.set("xyz", values_text(xyz))
        origin.set("rpy", values_text(rpy))
    output_root.append(world_joint)

    for item in source_root:
        if item.tag not in {"link", "joint"}:
            continue
        if not item.get("name", "").startswith(prefix):
            continue
        output_root.append(strip_prefix(item, prefix))

    ET.indent(output_root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output, encoding="utf-8", xml_declaration=True)
    summary: dict[str, int | str] = validate(ET.parse(output).getroot(), output)
    origin = world_joint.find("origin")
    summary.update(
        {
            "side": side,
            "position": position,
            "output": str(output),
            "base_xyz_m": origin.get("xyz", "") if origin is not None else "",
            "base_rpy_rad": origin.get("rpy", "") if origin is not None else "",
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--p0-output", type=Path, default=DEFAULT_P0_OUTPUT)
    parser.add_argument("--p1-output", type=Path, default=DEFAULT_P1_OUTPUT)
    parser.add_argument("--p2-output", type=Path, default=DEFAULT_P2_OUTPUT)
    parser.add_argument("--p0-calibration", type=Path, default=DEFAULT_P0_CALIBRATION)
    parser.add_argument("--p0-transform-key", default="T_old_base_new_base")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = ET.parse(args.input).getroot()
    p0_transform = load_transform(args.p0_calibration, args.p0_transform_key)
    results = [
        extract(source, "p1", args.p0_output, position_label="p0", base_transform=p0_transform),
        extract(source, "p1", args.p1_output),
        extract(source, "p2", args.p2_output),
    ]
    print(json.dumps(results, indent=2))
