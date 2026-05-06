#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile

import numpy as np

from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig


def _box_bottom_z(center: np.ndarray, rot_mat: np.ndarray, half_sizes: np.ndarray) -> float:
    extent_z = float(np.abs(rot_mat[2]) @ half_sizes)
    return float(center[2] - extent_z)


def _materialize_scene_xml(left_collision_mode: str, explicit_xml_path: str | None) -> str | None:
    if explicit_xml_path is not None:
        return explicit_xml_path
    if left_collision_mode == "mesh":
        return None

    base_xml_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "lerobot"
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "quest3_pika_gripper_scene.xml"
    )
    xml_text = base_xml_path.read_text()
    xml_text = xml_text.replace('meshdir="assets/"', f'meshdir="{(base_xml_path.parent / "assets").as_posix()}/"', 1)
    mesh_geom = (
        '<geom name="gripper_left_collision" type="mesh" mesh="pika_gripper_left_link" '
        'class="collision" friction="2.0 0.4 0.03" condim="4" solref="0.001 0.6" '
        'solimp="0.995 0.9995 0.00005" priority="2"/>'
    )

    if left_collision_mode == "stable_box":
        replacement = (
            '<geom name="gripper_left_collision_box" type="box" pos="0.022 -0.027 0.008" '
            'euler="0 0 1.012" size="0.0025 0.010 0.020" '
            'friction="2.0 0.4 0.03" condim="4" solref="0.001 0.6" '
            'solimp="0.995 0.9995 0.00005" priority="2"/>'
        )
    elif left_collision_mode == "stable_box_pair":
        replacement = (
            '<geom name="gripper_left_collision_box_main" type="box" pos="0.022 -0.027 0.008" '
            'euler="0 0 1.012" size="0.0025 0.010 0.020" '
            'friction="2.0 0.4 0.03" condim="4" solref="0.001 0.6" '
            'solimp="0.995 0.9995 0.00005" priority="2"/>'
            '<geom name="gripper_left_collision_box_support" type="box" pos="0.016 -0.021 0.020" '
            'euler="0 0 1.012" size="0.002 0.006 0.012" '
            'friction="2.0 0.4 0.03" condim="4" solref="0.001 0.6" '
            'solimp="0.995 0.9995 0.00005" priority="2"/>'
        )
    else:
        raise ValueError(f"Unknown left collision mode: {left_collision_mode}")

    if mesh_geom not in xml_text:
        raise ValueError("Could not locate left finger mesh collision geom in base Quest3 scene XML.")
    xml_text = xml_text.replace(mesh_geom, replacement, 1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="quest3-pika-left-collision-"))
    tmp_xml_path = tmp_dir / base_xml_path.name
    tmp_xml_path.write_text(xml_text)
    return str(tmp_xml_path)


@dataclass
class ContactRecord:
    pair: str
    dist: float
    pos: list[float]
    normal_world: list[float]
    wrench_local: list[float]


def _collect_object_contacts(env: Quest3PikaMujocoEnv) -> list[ContactRecord]:
    contacts: list[ContactRecord] = []
    mujoco = env._mujoco
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom1 is None or geom2 is None:
            continue
        if "workspace_object" not in (geom1, geom2):
            continue
        wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, wrench_local)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        contacts.append(
            ContactRecord(
                pair=f"{geom1}<->{geom2}",
                dist=float(contact.dist),
                pos=np.asarray(contact.pos, dtype=np.float64).round(6).tolist(),
                normal_world=frame[0].round(6).tolist(),
                wrench_local=wrench_local.round(6).tolist(),
            )
        )
    return contacts


def _snapshot(env: Quest3PikaMujocoEnv, phase: str, step: int, target_tcp_z: float | None) -> dict:
    mujoco = env._mujoco
    tendon_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_TENDON, "split")
    object_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "workspace_object_body")
    table_geom_id = env._table_geom_id
    object_geom_id = env._workspace_object_geom_id

    object_center = np.asarray(env.data.xpos[object_body_id], dtype=np.float64)
    object_rot = np.asarray(env.data.geom_xmat[object_geom_id], dtype=np.float64).reshape(3, 3)
    object_half_sizes = np.asarray(env.model.geom_size[object_geom_id], dtype=np.float64)
    object_bottom_z = _box_bottom_z(object_center, object_rot, object_half_sizes)

    table_center = np.asarray(env.data.geom_xpos[table_geom_id], dtype=np.float64)
    table_half_sizes = np.asarray(env.model.geom_size[table_geom_id], dtype=np.float64)
    table_top_z = float(table_center[2] + table_half_sizes[2])

    contacts = _collect_object_contacts(env)
    return {
        "phase": phase,
        "step": int(step),
        "target_tcp_z": None if target_tcp_z is None else float(target_tcp_z),
        "tcp_z": float(env._current_tcp_pose()[2, 3]),
        "mocap_z": float(env.data.mocap_pos[env._mocap_id][2]),
        "object_center_z": float(object_center[2]),
        "object_bottom_z": object_bottom_z,
        "object_bottom_minus_table_top": float(object_bottom_z - table_top_z),
        "left_qpos": float(env.data.qpos[env._gripper_joint_indices["left"]]),
        "right_qpos": float(env.data.qpos[env._gripper_joint_indices["right"]]),
        "actuator_ctrl": float(env.data.ctrl[env._gripper_actuator_id]),
        "actuator_force": float(env.data.actuator_force[env._gripper_actuator_id]),
        "tendon_length": float(env.data.ten_length[tendon_id]) if tendon_id >= 0 else None,
        "object_contacts": [asdict(contact) for contact in contacts],
    }


def _run_scenario(
    x: float,
    y: float,
    close_z: float,
    lift_zs: list[float],
    *,
    sim_xml_path: str | None,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
    sample_every: int,
) -> dict:
    cfg = Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False)
    if sim_xml_path is not None:
        cfg.sim_xml_path = sim_xml_path
    env = Quest3PikaMujocoEnv(cfg)
    try:
        env._debug_gripper_state = lambda _command: None
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        pose = env._initial_tcp_pose()

        trace: list[dict] = []
        trace.append(_snapshot(env, "reset", 0, None))

        pose[:3, 3] = np.array([x, y, close_z], dtype=np.float64)
        env._apply_tcp_pose(pose)
        env._step_physics(settle_steps)
        trace.append(_snapshot(env, "preclose", settle_steps, close_z))

        env._set_gripper_command(0.0)
        elapsed = 0
        while elapsed < close_steps:
            step_chunk = min(sample_every, close_steps - elapsed)
            env._step_physics(step_chunk)
            elapsed += step_chunk
            trace.append(_snapshot(env, "close", elapsed, close_z))

        for lift_index, lift_z in enumerate(lift_zs, start=1):
            pose[:3, 3] = np.array([x, y, lift_z], dtype=np.float64)
            env._apply_tcp_pose(pose)
            elapsed = 0
            while elapsed < lift_steps:
                step_chunk = min(sample_every, lift_steps - elapsed)
                env._step_physics(step_chunk)
                elapsed += step_chunk
                trace.append(_snapshot(env, f"lift_{lift_index}", elapsed, lift_z))

        object_zs = [entry["object_center_z"] for entry in trace]
        bottom_offsets = [entry["object_bottom_minus_table_top"] for entry in trace]
        final_contacts = trace[-1]["object_contacts"]
        return {
            "scenario": {
                "x": float(x),
                "y": float(y),
                "close_z": float(close_z),
                "lift_zs": [float(z) for z in lift_zs],
                "settle_steps": int(settle_steps),
                "close_steps": int(close_steps),
                "lift_steps": int(lift_steps),
                "sample_every": int(sample_every),
            },
            "summary": {
                "final_object_center_z": float(object_zs[-1]),
                "max_object_center_z": float(max(object_zs)),
                "min_object_center_z": float(min(object_zs)),
                "min_object_bottom_minus_table_top": float(min(bottom_offsets)),
                "max_object_bottom_minus_table_top": float(max(bottom_offsets)),
                "final_contact_count": len(final_contacts),
                "lift_succeeded_gt_0p46": bool(object_zs[-1] > 0.46),
            },
            "trace": trace,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Quest3 Pika close-then-lift contact mechanics.")
    parser.add_argument("--sim-xml-path", type=str, default=None, help="Optional explicit scene XML path.")
    parser.add_argument(
        "--left-collision-mode",
        choices=("mesh", "stable_box", "stable_box_pair"),
        default="mesh",
        help="Temporarily replace the left finger collision mesh with a simpler approximation for diagnostics.",
    )
    parser.add_argument("--xs", type=float, nargs="+", default=[0.47], help="TCP x positions to test.")
    parser.add_argument("--ys", type=float, nargs="+", default=[0.0], help="TCP y positions to test.")
    parser.add_argument("--close-zs", type=float, nargs="+", default=[0.30], help="TCP z used during close.")
    parser.add_argument(
        "--lift-zs",
        type=float,
        nargs="+",
        default=[0.32, 0.34, 0.36, 0.38],
        help="TCP z targets used during lift.",
    )
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--lift-steps", type=int, default=240)
    parser.add_argument("--sample-every", type=int, default=40)
    parser.add_argument("--trace-limit", type=int, default=0, help="If >0, truncate each scenario trace to this many records.")
    parser.add_argument("--summary-only", action="store_true", help="Print one-line summaries without the full JSON trace payload.")
    parser.add_argument("--json-only", action="store_true", help="Print only the JSON payload without summary lines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim_xml_path = _materialize_scene_xml(args.left_collision_mode, args.sim_xml_path)
    results = []
    for x in args.xs:
        for y in args.ys:
            for close_z in args.close_zs:
                result = _run_scenario(
                    x,
                    y,
                    close_z,
                    list(args.lift_zs),
                    sim_xml_path=sim_xml_path,
                    settle_steps=args.settle_steps,
                    close_steps=args.close_steps,
                    lift_steps=args.lift_steps,
                    sample_every=args.sample_every,
                )
                if args.trace_limit > 0:
                    result["trace"] = result["trace"][: args.trace_limit]
                results.append(result)
                summary = result["summary"]
                if not args.json_only:
                    print(
                        "SUMMARY "
                        f"x={x:.3f} y={y:.3f} close_z={close_z:.4f} "
                        f"final_z={summary['final_object_center_z']:.6f} "
                        f"max_z={summary['max_object_center_z']:.6f} "
                        f"min_clearance={summary['min_object_bottom_minus_table_top']:.6f} "
                        f"contacts={summary['final_contact_count']} "
                        f"lifted={summary['lift_succeeded_gt_0p46']} "
                        f"left_collision_mode={args.left_collision_mode}"
                    )
    if not args.summary_only:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
