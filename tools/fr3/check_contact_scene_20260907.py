"""Offline mesh/self-collision and +110 mm height diagnostics. No hardware.

Height overlap is NOT a tabletop collision proof: tabletop footprint and
orientation are not surveyed. Two plane conventions are reported explicitly.
"""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from scipy.optimize import brentq
from replay_ik_trajectory_guarded import JOINTS, read_contact_ik, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-height-m", type=float, default=.110)
    parser.add_argument("--height-only", action="store_true",
                        help="Only inspect heights; explicitly skip self-collision queries")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.table_height_m) or args.table_height_m < 0:
        raise ValueError("Invalid table height")
    episodes = read_contact_ik(args.csv)
    model = pin.buildModelFromUrdf(str(args.urdf))
    geom = pin.buildGeomFromUrdf(model, str(args.urdf), pin.GeometryType.COLLISION,
                                 package_dirs=[str(args.urdf.parent)])
    data = model.createData()
    tree = ET.parse(args.urdf)
    adjacent = {frozenset((j.find("parent").attrib["link"], j.find("child").attrib["link"]))
                for j in tree.findall("joint")}
    excluded = []
    for i, j in itertools.combinations(range(len(geom.geometryObjects)), 2):
        a, b = geom.geometryObjects[i], geom.geometryObjects[j]
        links = (model.frames[a.parentFrame].name, model.frames[b.parentFrame].name)
        reason = ("same rigid body" if a.parentJoint == b.parentJoint else
                  "direct URDF parent-child; structural exclusion requires physical review"
                  if frozenset(links) in adjacent else None)
        if reason:
            excluded.append(dict(pair=[a.name, b.name], reason=reason))
        else:
            geom.addCollisionPair(pin.CollisionPair(i, j))
    gd = pin.GeometryData(geom)
    q = pin.neutral(model)
    arm_idx = [model.joints[model.getJointId(name)].idx_q for name in JOINTS]
    grip_names = [name for name in model.names if name.startswith("joint_gripper_")]
    if len(grip_names) != 7:
        raise ValueError("Unexpected V2 gripper DOF mapping")
    grip_idx = [model.joints[model.getJointId(name)].idx_q for name in grip_names]
    maximum_drive = float(np.min(model.upperPositionLimit[grip_idx]))
    left = model.getFrameId("link_gripper_contact_left")
    right = model.getFrameId("link_gripper_contact_right")
    base = model.getFrameId("base")
    if max(left, right, base) >= model.nframes:
        raise ValueError("Missing required frames")
    pin.framesForwardKinematics(model, data, q)
    base_pose = data.oMf[base].copy()
    vertices = {}
    for i, item in enumerate(geom.geometryObjects):
        if not hasattr(item.geometry, "vertices"):
            raise ValueError(f"Unsupported non-mesh collision object: {item.name}")
        vertices[i] = np.asarray(item.geometry.vertices()).copy()
    grip_geoms = [i for i, item in enumerate(geom.geometryObjects)
                  if "gripper" in model.frames[item.parentFrame].name
                  or model.frames[item.parentFrame].name in ("base_link", "link_sensor_ft")]

    def opening(angle):
        q[grip_idx] = angle
        pin.framesForwardKinematics(model, data, q)
        return float(np.linalg.norm(data.oMf[left].translation - data.oMf[right].translation))

    report = dict(input_sha256=sha256(args.csv), urdf_sha256=sha256(args.urdf),
                  script_sha256=sha256(__file__), pinocchio_version=pin.__version__,
                  collision_meshes=len(vertices), checked_self_pairs=0 if args.height_only else len(geom.collisionPairs),
                  excluded_pairs=excluded, tabletop_height_above_mount_m=args.table_height_m,
                  assumptions=["URDF base origin is the physical mounting-surface origin.",
                               "Plane A: tabletop parallel to base XY; normal is base Z.",
                               "Plane B: tabletop parallel to old-world XY at base-origin world Z plus measured height."],
                  environment_collision_certified=False, hardware_ready=False,
                  limitations=["No measured tabletop footprint or obstacle poses; plane heights are diagnostics only.",
                               "Raw recorded frames only: neither inter-frame nor start-transit collision certified.",
                               "URDF tree does not encode all V2 closed-loop physical connections; flagged pairs need CAD review.",
                               "Excluded parent-child pairs are listed, not silently treated as a final safety whitelist.",
                               "Single-arm scene; other robot, cables, table sides/supports, BOX and payload not modeled."],
                  base_pose_world=base_pose.homogeneous.tolist(), episodes=[])
    for ep, seq in episodes.items():
        records = []
        collisions = {}
        max_target_error = 0.
        min_distance = float("inf")
        for sample in seq:
            q[arm_idx] = sample["q"]
            angle = brentq(lambda a: opening(a) - sample["width_m"], 0., maximum_drive, xtol=1e-12)
            q[grip_idx] = angle
            pin.framesForwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, geom, gd, q)
            center = (data.oMf[left].translation + data.oMf[right].translation) / 2
            target = np.array([float(sample["original"][f"contact_target_{axis}_m"]) for axis in "xyz"])
            error = float(np.linalg.norm(center - target))
            if not np.isfinite(error) or error > .001:
                raise ValueError(f"Contact frame mapping inconsistent with input target: {error}")
            max_target_error = max(max_target_error, error)
            points = np.concatenate([vertices[i] @ gd.oMg[i].rotation.T + gd.oMg[i].translation
                                     for i in grip_geoms])
            base_points = (points - base_pose.translation) @ base_pose.rotation
            center_base = base_pose.actInv(center)
            height_a = float(base_points[:, 2].min() - args.table_height_m)
            height_b = float(points[:, 2].min() - base_pose.translation[2] - args.table_height_m)
            if not args.height_only:
                pin.computeCollisions(geom, gd, False)
                pin.computeDistances(geom, gd)
            for pair, cr, dr in ([] if args.height_only else zip(geom.collisionPairs, gd.collisionResults, gd.distanceResults)):
                if not np.isfinite(dr.min_distance):
                    raise ValueError("Nonfinite mesh distance")
                min_distance = min(min_distance, float(dr.min_distance))
                if cr.isCollision():
                    name = geom.geometryObjects[pair.first].name + " / " + geom.geometryObjects[pair.second].name
                    entry = collisions.setdefault(name, dict(frames=0, first_frame=sample["frame_index"]))
                    entry["frames"] += 1
                    entry["last_frame"] = sample["frame_index"]
            records.append(dict(frame=sample["frame_index"], contact_center_base_m=center_base.tolist(),
                                contact_height_above_plane_a_m=float(center_base[2] - args.table_height_m),
                                lowest_gripper_above_plane_a_m=height_a,
                                lowest_gripper_above_plane_b_m=height_b))
        result = dict(episode=ep, frames=len(seq), first_frame=records[0],
                      min_gripper_height_a_m=min(r["lowest_gripper_above_plane_a_m"] for r in records),
                      min_gripper_height_b_m=min(r["lowest_gripper_above_plane_b_m"] for r in records),
                      frames_with_gripper_below_height_a=sum(r["lowest_gripper_above_plane_a_m"] < 0 for r in records),
                      frames_with_gripper_below_height_b=sum(r["lowest_gripper_above_plane_b_m"] < 0 for r in records),
                      maximum_contact_fk_error_m=max_target_error, candidate_self_collision_pairs=collisions,
                      self_collision_checked=not args.height_only,
                      minimum_checked_mesh_distance_m=None if args.height_only else min_distance, samples=records)
        report["episodes"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
