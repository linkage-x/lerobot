"""dual_fr3_replay_runtime (roadmap D1/D3) on fake arms: one start, one stop, and the gates.

Each run spawns two arm processes, as the real runtime will; the trajectories are short (40 frames at
50 Hz) so a run takes about a second.
"""

import numpy as np
import pytest

from tools.fr3 import dual_fr3_replay_runtime as rt, fr3_kinematics as kin

FPS = 50.0
N = 40
TOOL = kin.urdf_fixed_transform("corenetic_socket_ee")
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _path(amplitude: float = 0.15) -> np.ndarray:
    """A smooth joint path near the ready posture: joints 1, 2 and 4 swing."""
    s = np.sin(np.linspace(0.0, np.pi, N))
    q = np.tile(kin.Q_READY, (N, 1))
    q[:, 0] += amplitude * s
    q[:, 1] += 0.5 * amplitude * s
    q[:, 3] += 0.5 * amplitude * s
    return q


def _plans(spacing: float = 0.8, faults=None, amplitude: float = 0.15, clocks=None):
    """Right arm on the path, left arm on its mirror image -- the pseudo_bimanual geometry."""
    q_r = _path(amplitude)
    q_l = q_r * MIRROR
    T_right_left = np.eye(4)
    T_right_left[1, 3] = spacing
    plans = []
    for name, q in (("right", q_r), ("left", q_l)):
        spec = {
            "kind": "fake",
            "tool": TOOL.tolist(),
            "fault": (faults or {}).get(name),
            "clock": (clocks or {}).get(name),
            "start_joints": q[0].tolist(),
        }
        plans.append(rt.ArmPlan(name, kin.NOMINAL.fk(q) @ TOOL, spec))
    return plans, T_right_left, (q_r, q_l)


def _run(tmp_path, plans, T_right_left, **settings):
    return rt.run(
        plans,
        fps=FPS,
        T_right_left=T_right_left,
        run_dir=tmp_path / "run",
        settings=rt.Settings(**settings),
        go=lambda: True,
        start_lead_s=0.1,
    )


def test_both_arms_start_together_and_complete(tmp_path) -> None:
    plans, T_right_left, _ = _plans()
    report = _run(tmp_path, plans, T_right_left)
    assert report["stop"] is None
    for row in report["arms"].values():
        assert row["outcome"] == "completed" and row["frames_commanded"] == N
        assert row["target_error_mm_p95"] < 15.0  # the fake trails its target, as the arm does
    assert report["start_skew_ms"] < 5.0
    assert report["min_gap_mm"] > 300.0


def test_a_fault_in_one_arm_stops_the_other_within_one_period(tmp_path) -> None:
    plans, T_right_left, _ = _plans(faults={"left": {"kind": "raise", "at_frame": 15}})
    report = _run(tmp_path, plans, T_right_left)
    left, right = report["arms"]["left"], report["arms"]["right"]
    assert left["outcome"] == "fault" and "libfranka reflex" in left["reason"]
    assert report["stop"]["by"] == 2  # the left arm wrote the stop word itself
    assert right["outcome"] == "stopped"
    assert right["commands_after_stop"] <= 1 and report["stopped_together"]
    assert right["frames_commanded"] <= 17


def test_an_arm_process_that_dies_is_caught_by_the_coordinator(tmp_path) -> None:
    plans, T_right_left, _ = _plans(faults={"right": {"kind": "exit", "at_frame": 10}})
    report = _run(tmp_path, plans, T_right_left)
    assert report["stop"]["by"] == rt.STOP_BY_COORDINATOR
    assert "exited" in report["stop"]["events"][0]["reason"]
    assert report["arms"]["left"]["outcome"] == "stopped"
    assert report["arms"]["left"]["commands_after_stop"] <= 1


def test_a_hung_arm_is_stopped_on_its_stale_heartbeat(tmp_path) -> None:
    plans, T_right_left, _ = _plans(faults={"left": {"kind": "hang", "at_frame": 10}})
    report = _run(tmp_path, plans, T_right_left)
    assert "hung" in report["stop"]["events"][0]["reason"]
    right = report["arms"]["right"]
    assert right["outcome"] == "stopped" and right["commands_after_stop"] <= 1
    # stale after 3 periods: the right arm got at most ~4 frames past the hang
    assert right["frames_commanded"] <= 10 + 3 + 2


def test_an_arm_that_stops_following_trips_its_own_tracking_gate(tmp_path) -> None:
    plans, T_right_left, _ = _plans(faults={"right": {"kind": "freeze", "at_frame": 5}}, amplitude=0.4)
    report = _run(
        tmp_path, plans, T_right_left, max_command_position_error_m=0.01, tracking_error_grace_frames=5
    )
    right = report["arms"]["right"]
    assert right["outcome"] == "fault" and "tracking error" in right["reason"]
    assert report["arms"]["left"]["outcome"] == "stopped" and report["stopped_together"]


def test_arms_too_close_at_frame_zero_are_refused_before_anything_moves(tmp_path) -> None:
    plans, T_right_left, _ = _plans(spacing=0.15)
    report = _run(tmp_path, plans, T_right_left)
    assert report["stop"]["by"] == rt.STOP_BY_GATE and "at frame 0" in report["stop"]["events"][0]["reason"]
    assert all(row["frames_commanded"] == 0 for row in report["arms"].values())


def test_the_separation_gate_stops_both_when_the_arms_close_in(tmp_path) -> None:
    plans, T_right_left, (q_r, q_l) = _plans(spacing=0.55, amplitude=0.5)
    gaps = [rt.arm_separation(a, b, T_right_left)[0] for a, b in zip(q_r, q_l, strict=True)]
    assert min(gaps) < gaps[0] - 0.05  # the path really closes in
    d_safe = (gaps[0] + min(gaps)) / 2.0
    report = _run(tmp_path, plans, T_right_left, d_safe_m=d_safe)
    assert report["stop"]["by"] == rt.STOP_BY_GATE
    assert all(row["outcome"] == "stopped" for row in report["arms"].values())
    assert report["stopped_together"]
    assert report["min_gap_mm"] < d_safe * 1e3


def test_the_stop_word_keeps_its_first_writer(tmp_path) -> None:
    sync = rt.ArmSync(tmp_path / "sync.bin", create=True, period_ns=20_000_000)
    other = rt.ArmSync(tmp_path / "sync.bin")
    assert not other.stopped()
    assert sync.request_stop(2) and not other.request_stop(1)
    assert other.stopped() and other.stop_by == 2
    sync.publish_joints(1, np.arange(7.0))
    assert np.array_equal(other.joints(1)[1], np.arange(7.0))
    sync.close()
    other.close()


def test_separation_is_symmetric_and_shrinks_with_spacing() -> None:
    q = kin.Q_READY
    T = np.eye(4)
    gaps = []
    for spacing in (1.0, 0.8, 0.6):
        T[1, 3] = spacing
        gaps.append(rt.arm_separation(q, q * MIRROR, T)[0])
    assert gaps[0] > gaps[1] > gaps[2]
    # parallel arms: the gap grows with the spacing -- to 0.1 mm, the fitted capsules being no more
    # mirror-symmetric than the meshes they enclose
    assert gaps[0] - gaps[1] == pytest.approx(0.2, abs=1e-4)


def test_incremental_command_clips_one_step() -> None:
    T0 = kin.NOMINAL.fk(kin.Q_READY)[0]
    T1 = T0.copy()
    T1[:3, 3] += [0.05, 0.0, 0.0]
    T = rt.incremental_command(T0, T1, 0.01, 2.0)
    assert rt.pose_error(T0, T)[0] == pytest.approx(0.01)
