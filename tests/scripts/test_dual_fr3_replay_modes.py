"""dual_fr3_replay_modes (roadmap C2): absolute and relative targets for two arms, and what each rests on."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import dual_fr3_replay_modes as md, fr3_kinematics as kin
from tools.thor.gmsl2 import bimanual_qc as bq, embodiment as em, world_provenance as wp

REPO = Path(__file__).resolve().parents[2]
TOOL = kin.urdf_fixed_transform("corenetic_socket_ee")
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
WORLD = "world_test"
#: the camera world 1 m in front of the right base, turned 90 deg
T_BASE_WORLD = np.array(
    [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
)
SPACING = 0.8
EXPORT_EDGE = {
    "from_world_frame_id": WORLD,
    "to_world_frame_id": wp.FR3_BASE_WORLD_ID,
    "T_to_from": T_BASE_WORLD.tolist(),
    "method": "fr3_ball_pivot_check: test",
    "created_utc": "2026-10-15T03:08:47Z",
    "sigma_mm": 2.0,
}


def T_right_left(spacing: float = SPACING) -> np.ndarray:
    T = np.eye(4)
    T[1, 3] = spacing
    return T


def graph(spacing: float = SPACING, *, export_edge: dict | None = None) -> dict:
    return {
        "version": 1,
        "nodes": [
            {"world_frame_id": WORLD, "reason": "frozen_reference"},
            {"world_frame_id": wp.FR3_BASE_WORLD_ID, "reason": "robot_base"},
            {"world_frame_id": "fr3_base_left", "reason": "robot_base"},
        ],
        "edges": [
            export_edge or EXPORT_EDGE,
            {
                "from_world_frame_id": "fr3_base_left",
                "to_world_frame_id": wp.FR3_BASE_WORLD_ID,
                "T_to_from": T_right_left(spacing).tolist(),
                "method": "fr3_arm_arm_ball: test",
                "created_utc": "2026-10-27T00:00:00Z",
                "sigma_mm": 1.0,
            },
        ],
    }


def description() -> dict:
    d = em.load(em.EMBODIMENT_DIR / "fr3_dual.json", repo_root=REPO)
    d["arms"]["left"]["replay_config"] = d["arms"]["right"]["replay_config"]
    return d


def joint_paths(n: int, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A right-arm joint path near the ready pose and the left arm's mirror image."""
    s = np.sin(np.linspace(0.0, 2 * np.pi, n) + phase)
    q = np.tile(kin.Q_READY, (n, 1))
    q[:, 0] += 0.25 * s
    q[:, 3] += 0.2 * s
    return q, q * MIRROR


def hands_in_right_base(n: int, phase: float = 0.0, spacing: float = SPACING) -> dict[str, np.ndarray]:
    q_r, q_l = joint_paths(n, phase)
    return {"right": kin.NOMINAL.fk(q_r) @ TOOL, "left": T_right_left(spacing) @ kin.NOMINAL.fk(q_l) @ TOOL}


def _pose7(T: np.ndarray) -> list[float]:
    return [*T[:3, 3].tolist(), *wp._quat_from_matrix(T[:3, :3].tolist())]


def write_session(
    root: Path,
    name: str,
    episodes: list[dict[str, np.ndarray]],
    *,
    in_world: bool = False,
    lost: dict[str, list[int]] | None = None,
) -> Path:
    """A two-hand session; poses given in the right base, written there (re-expressed, as export does) or in
    the camera world."""
    session = root / name
    fields = [
        "schema_version", "timestamp_s", "num_cameras", "solve_ok", "camera_set", "camera_set_changed",
        "frame_global_index", "episode_index", "frame_index",
        "state_x_m", "state_y_m", "state_z_m", "state_qx", "state_qy", "state_qz", "state_qw",
    ]  # fmt: skip
    to_file = np.linalg.inv(T_BASE_WORLD) if in_world else np.eye(4)
    (session / "derived" / bq.SIDECAR).mkdir(parents=True)
    for hand in ("right", "left"):
        rows, g = [], 0
        for e, hands in enumerate(episodes):
            for k, T in enumerate(hands[hand]):
                ok = k not in ((lost or {}).get(hand) or [])
                rows.append(
                    dict(
                        zip(
                            fields,
                            [3, k / 60.0, 5, ok, "06,07,08,09,13", False, g, e, k, *_pose7(to_file @ T)],
                            strict=True,
                        )
                    )
                )
                g += 1
        with (session / "derived" / bq.SIDECAR / f"state_action.{hand}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    world_frame = (
        {"world_frame_id": WORLD, "status": "ok"}
        if in_world
        else {
            "world_frame_id": wp.FR3_BASE_WORLD_ID,
            "status": "ok",
            "edges_by_source_world": {
                WORLD: [
                    {
                        **{
                            k: EXPORT_EDGE[k]
                            for k in ("from_world_frame_id", "to_world_frame_id", "method", "created_utc")
                        },
                        "traversed_reversed": False,
                    }
                ]
            },
        }
    )
    (session / "meta").mkdir()
    (session / "meta" / "info.json").write_text(json.dumps({"fps": 60, "world_frame": world_frame}))
    return session


def test_a_session_reads_as_both_hands_on_shared_frames_holding_lost_ones(tmp_path: Path) -> None:
    hands = hands_in_right_base(50)
    session = write_session(tmp_path, "s", [hands], lost={"left": [10, 11]})
    ep = md.read_episode(session, 0)
    assert len(ep.frames) == 50 and ep.fps == 60.0 and ep.dataset_world == wp.FR3_BASE_WORLD_ID
    assert not ep.seen["left"][10] and ep.seen["left"][12]
    assert np.allclose(ep.poses["left"][11], ep.poses["left"][9])  # held
    assert np.allclose(ep.poses["right"], hands["right"], atol=1e-9)


def test_absolute_puts_each_hand_in_its_base_through_the_graph(tmp_path: Path) -> None:
    hands = hands_in_right_base(40)
    ep = md.read_episode(write_session(tmp_path, "s", [hands]), 0)
    plan = md.plan("absolute", ep, description(), graph())
    assert plan.ok and plan.align == {"kind": "none"}
    assert np.allclose(plan.targets["right"], hands["right"], atol=1e-9)
    assert np.allclose(plan.targets["left"], np.linalg.inv(T_right_left()) @ hands["left"], atol=1e-9)
    assert np.allclose(plan.T_right_left, T_right_left())
    assert plan.placement["sigma_mm"] == pytest.approx(1.0) and not plan.placement["assumed"]
    assert [e["to_world_frame_id"] for e in plan.arms["left"]["edges"]] == ["fr3_base"]
    with pytest.raises(TypeError):
        json.dumps(plan.targets)
    json.dumps(plan.record())


def test_absolute_is_refused_once_the_export_edge_is_no_longer_the_graph_s(tmp_path: Path) -> None:
    ep = md.read_episode(write_session(tmp_path, "s", [hands_in_right_base(30)]), 0)
    moved = copy.deepcopy(EXPORT_EDGE)
    moved["created_utc"] = "2026-11-01T00:00:00Z"  # the base was re-registered
    plan = md.plan("absolute", ep, description(), graph(export_edge=moved))
    assert not plan.ok and "re-export" in plan.reason
    # relative replay does not rest on the world edge, only on the base-to-base one
    assert md.plan("relative", ep, description(), graph(export_edge=moved)).ok


def test_absolute_takes_no_alignment(tmp_path: Path) -> None:
    ep = md.read_episode(write_session(tmp_path, "s", [hands_in_right_base(30)]), 0)
    plan = md.plan("absolute", ep, description(), graph(), align="fit")
    assert not plan.ok and "no alignment" in plan.reason


def test_relative_moves_both_hands_by_one_transform_and_keeps_their_relation(tmp_path: Path) -> None:
    hands = hands_in_right_base(40)
    ep = md.read_episode(write_session(tmp_path, "s", [hands]), 0)
    shift = np.eye(4)
    shift[:3, :3] = kin._rpy_matrix(0.0, 0.0, 0.2)
    shift[:3, 3] = [0.05, -0.02, 0.03]
    plan = md.plan("relative", ep, description(), graph(), align="matrix", T_align=shift)
    assert plan.ok and np.allclose(plan.align["T_align"], shift)
    left_in_right = T_right_left() @ plan.targets["left"]
    # the hands' relation is exactly the recording's: one rigid motion of both
    rel = np.linalg.inv(left_in_right) @ plan.targets["right"]
    rel_recorded = np.linalg.inv(hands["left"]) @ hands["right"]
    assert np.allclose(rel, rel_recorded, atol=1e-9)
    assert np.allclose(plan.targets["right"], shift @ hands["right"], atol=1e-9)


def test_relative_on_a_camera_world_dataset_needs_an_alignment(tmp_path: Path) -> None:
    hands = hands_in_right_base(30)
    ep = md.read_episode(write_session(tmp_path, "s", [hands], in_world=True), 0)
    assert ep.dataset_world == WORLD
    refused = md.plan("relative", ep, description(), graph(), align="identity")
    assert not refused.ok and "align=registered" in refused.reason
    registered = md.plan("relative", ep, description(), graph(), align="registered")
    assert registered.ok and np.allclose(registered.targets["right"], hands["right"], atol=1e-9)
    assert registered.align["edges"]


def test_fit_pulls_a_demonstration_recorded_past_the_reach_back_in(tmp_path: Path) -> None:
    hands = hands_in_right_base(30)
    away = np.eye(4)
    away[0, 3] = 0.55  # both hands past the reach
    ep = md.read_episode(write_session(tmp_path, "s", [{h: away @ T for h, T in hands.items()}]), 0)
    plan = md.plan("relative", ep, description(), graph(), align="fit")
    fit = plan.align["fit"]
    assert plan.ok and fit["reachable_fraction"] > fit["reachable_fraction_unaligned"]
    assert fit["shift_m"][0] < 0.0
    # one T_align for both hands
    T_align = np.asarray(plan.align["T_align"])
    assert np.allclose(
        plan.arms["left"]["T_root_dataset"], np.linalg.inv(T_right_left()) @ T_align, atol=1e-9
    )
