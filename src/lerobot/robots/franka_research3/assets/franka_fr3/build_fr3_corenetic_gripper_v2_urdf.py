#!/usr/bin/env python3
"""Combine the existing FR3 arm with the supplied Corenetic gripper V2 URDF.

The vendor archive contains only the flange/gripper assembly, not the Franka
arm.  It also contains two stale joints whose child links are absent and one
collision mesh path that is not present in the archive.  This builder keeps the
archive untouched and produces a loadable composite model by:

* reusing the FR3 arm chain from ``fr3_corenetic_gripper.urdf``;
* omitting the two stale vendor joints;
* using the available visual mesh for the missing gripper-base collision mesh;
* adding collision geometry from the supplied visual mesh where the vendor
  URDF omitted collision geometry; and
* adding ``corenetic_gripper_ee`` as a compatibility alias for replay tools.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_ARM_SOURCE = HERE / "fr3_corenetic_gripper.urdf"
DEFAULT_GRIPPER_SOURCE = (
    HERE
    / "gripper_w_v2_franka_description"
    / "urdf"
    / "gripper_w_v2_franka_description.urdf"
)
DEFAULT_OUTPUT = HERE / "fr3_corenetic_gripper_v2.urdf"

ARM_LINKS = {"base", "fr3_ee", *(f"fr3_link{i}" for i in range(9))}
ARM_JOINTS = {
    "fr3_base_joint",
    "fr3_ee_joint",
    *(f"fr3_joint{i}" for i in range(1, 9)),
}
STALE_VENDOR_JOINTS = {
    "joint_franka_corenetic_gripper_4",
    "joint_lt_gripper_base",
}
PACKAGE_PREFIX = "package://gripper_w_v2_franka_description/"
V2_ASSET_PREFIX = "gripper_w_v2_franka_description/"


def rewrite_mesh_paths(root: ET.Element) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(PACKAGE_PREFIX):
            filename = V2_ASSET_PREFIX + filename.removeprefix(PACKAGE_PREFIX)
        if filename == V2_ASSET_PREFIX + "meshes/collision/link_gripper_base.STL":
            filename = V2_ASSET_PREFIX + "meshes/link_gripper_base.STL"
        mesh.set("filename", filename)


def add_missing_collision_geometry(root: ET.Element) -> int:
    added = 0
    for link in root.findall("link"):
        if link.find("collision") is not None:
            continue
        visual = link.find("visual")
        if visual is None or visual.find("geometry") is None:
            continue
        collision = ET.Element("collision")
        origin = visual.find("origin")
        if origin is not None:
            collision.append(copy.deepcopy(origin))
        collision.append(copy.deepcopy(visual.find("geometry")))
        link.append(collision)
        added += 1
    return added


def sanitize_placeholder_inertias(root: ET.Element) -> int:
    """Make the vendor's near-zero placeholder inertia tensors positive definite."""
    sanitized = 0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue
        if float(mass.get("value", "0")) > 1e-12:
            continue
        # MuJoCo rejects the supplied 1e-16 full-inertia tensor even after the
        # off-diagonal terms are removed.  Keep these links dynamically
        # negligible while using values safely above the compiler tolerance.
        mass.set("value", "1e-6")
        inertia.set("ixx", "1e-9")
        inertia.set("iyy", "1e-9")
        inertia.set("izz", "1e-9")
        inertia.set("ixy", "0")
        inertia.set("ixz", "0")
        inertia.set("iyz", "0")
        sanitized += 1
    return sanitized


def add_missing_drive_inertia(root: ET.Element) -> int:
    link = next(
        (item for item in root.findall("link") if item.get("name") == "link_gripper_drive"),
        None,
    )
    if link is None or link.find("inertial") is not None:
        return 0
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1e-6"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-9", "ixy": "0", "ixz": "0", "iyy": "1e-9", "iyz": "0", "izz": "1e-9"},
    )
    link.insert(0, inertial)
    return 1


def name_geometries(root: ET.Element) -> None:
    for link in root.findall("link"):
        link_name = link.get("name", "link")
        for tag in ("visual", "collision"):
            for index, geometry in enumerate(link.findall(tag)):
                geometry.set("name", f"{link_name}_{tag}_{index}")


def validate_urdf(root: ET.Element, output_path: Path) -> dict[str, int]:
    links = [item.get("name", "") for item in root.findall("link")]
    joints = [item.get("name", "") for item in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError("Generated URDF contains duplicate link names")
    if len(joints) != len(set(joints)):
        raise ValueError("Generated URDF contains duplicate joint names")

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
            raise ValueError(
                f"Joint {joint.get('name')} references {parent_name} -> {child_name}, "
                "but at least one link is absent"
            )
        if child_name in children:
            raise ValueError(f"Link {child_name} has more than one parent")
        children.add(child_name)

    roots = sorted(link_set - children)
    if roots != ["base"]:
        raise ValueError(f"Expected base as the only root link, got {roots}")
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

    return {
        "links": len(links),
        "joints": len(joints),
        "visuals": len(list(root.iter("visual"))),
        "collisions": len(list(root.iter("collision"))),
        "meshes": len(list(root.iter("mesh"))),
    }


def build(args: argparse.Namespace) -> dict[str, int]:
    arm_root = ET.parse(args.arm_source).getroot()
    gripper_root = ET.parse(args.gripper_source).getroot()

    output_root = ET.Element("robot", {"name": args.output.stem})
    mujoco = arm_root.find("mujoco")
    if mujoco is not None:
        mujoco_copy = copy.deepcopy(mujoco)
        compiler = mujoco_copy.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", ".")
            compiler.set("fusestatic", "false")
        output_root.append(mujoco_copy)

    output_root.append(
        ET.Comment(
            " FR3 arm from fr3_corenetic_gripper.urdf; gripper from "
            "gripper_w_v2_franka_description.zip (SHA256 "
            "6c03f2266e5b66cb9d55875c49b58189c78c751d8aac17ffbdd51ed9aea35ee3). "
        )
    )
    for item in arm_root:
        if item.tag == "link" and item.get("name") in ARM_LINKS:
            output_root.append(copy.deepcopy(item))
        elif item.tag == "joint" and item.get("name") in ARM_JOINTS:
            output_root.append(copy.deepcopy(item))

    mount = ET.SubElement(output_root, "joint", {"name": "fr3_corenetic_gripper_v2_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", {"rpy": "0 0 -0.7853981633974483", "xyz": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "fr3_link8"})
    ET.SubElement(mount, "child", {"link": "base_link"})

    for item in gripper_root:
        if item.tag == "joint" and item.get("name") in STALE_VENDOR_JOINTS:
            continue
        if item.tag in {"link", "joint"}:
            output_root.append(copy.deepcopy(item))

    rewrite_mesh_paths(output_root)
    collisions_added = add_missing_collision_geometry(output_root)
    inertias_sanitized = sanitize_placeholder_inertias(output_root)
    drive_inertias_added = add_missing_drive_inertia(output_root)

    ET.SubElement(output_root, "link", {"name": "corenetic_gripper_ee"})
    alias = ET.SubElement(output_root, "joint", {"name": "fr3_hand_tcp_joint", "type": "fixed"})
    ET.SubElement(alias, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    ET.SubElement(alias, "parent", {"link": "link_gripper_tcp"})
    ET.SubElement(alias, "child", {"link": "corenetic_gripper_ee"})

    name_geometries(output_root)
    ET.indent(output_root, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(args.output, encoding="utf-8", xml_declaration=True)
    summary = validate_urdf(ET.parse(args.output).getroot(), args.output)
    summary["collision_geometries_added"] = collisions_added
    summary["placeholder_inertias_sanitized"] = inertias_sanitized
    summary["drive_inertias_added"] = drive_inertias_added
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-source", type=Path, default=DEFAULT_ARM_SOURCE)
    parser.add_argument("--gripper-source", type=Path, default=DEFAULT_GRIPPER_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build(arguments)
    print(f"output={arguments.output}")
    print(json.dumps(result, indent=2))
