from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tools.thor import p0_native_arm_unchecked as unchecked


def test_episode_knots_extracts_joint_columns() -> None:
    coefficients = np.zeros((2, 2, 8), dtype=float)
    coefficients[0, -1] = np.arange(8)
    coefficients[1, -1] = np.arange(8) + 10
    coefficients[1, 0] = 1
    source = {
        "episodes": [
            {
                "episode": 0,
                "frames": 3,
                "coeff_descending_unit_interval": coefficients.tolist(),
            }
        ]
    }

    result = unchecked._episode_knots(source, 0)

    assert result.shape == (3, 7)
    np.testing.assert_array_equal(result[0], np.arange(7))
    np.testing.assert_array_equal(result[-1], np.arange(7) + 11)


def test_episode_knots_accepts_p1_relocalized_joint_plan() -> None:
    knots = np.arange(21, dtype=float).reshape(3, 7)
    source = {"episodes": [{"episode": 1, "frames": 3, "joint_knots": knots.tolist()}]}

    np.testing.assert_array_equal(unchecked._episode_knots(source, 1), knots)


def test_active_p1_source_is_hash_checked(tmp_path, monkeypatch) -> None:
    plan = tmp_path / "plan.json"
    calibration = tmp_path / "calibration.json"
    active = tmp_path / "active.json"
    plan.write_text(json.dumps({"schema": "p0_relocalized_joint_plan/v1", "episodes": []}), encoding="utf-8")
    calibration.write_text(json.dumps({"status": "passed", "world_frame_id": "world_test"}), encoding="utf-8")
    active.write_text(
        json.dumps(
            {
                "schema": "p1_simple_eye_hand_active/v1",
                "plan_path": str(plan),
                "plan_sha256": unchecked._sha256(plan),
                "calibration_path": str(calibration),
                "calibration_sha256": unchecked._sha256(calibration),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(unchecked, "P1_ACTIVE_PATH", active)

    source_path, source, label = unchecked._resolve_source()

    assert source_path == plan
    assert source["schema"] == "p0_relocalized_joint_plan/v1"
    assert "world_test" in label


def test_fixed_chunks_overlap_only_at_stored_boundary() -> None:
    q = np.arange(401 * 7, dtype=float).reshape(401, 7)

    chunks = list(unchecked._chunk_waypoints(q))

    assert [len(chunk) for chunk in chunks] == [200, 200, 3]
    np.testing.assert_array_equal(chunks[0][-1], chunks[1][0])
    np.testing.assert_array_equal(chunks[1][-1], chunks[2][0])


def test_start_trajectory_uses_measured_pose_and_slow_speed() -> None:
    factory = Mock(return_value=object())
    core = SimpleNamespace(JointTrajectory=factory)
    current_q = np.arange(7, dtype=float)
    target_q = current_q + 0.5

    result = unchecked._start_trajectory(core, current_q, target_q)

    assert result is factory.return_value
    factory.assert_called_once_with(
        [current_q.tolist(), target_q.tolist()],
        0.05,
        0.0,
        30.0,
    )


def test_unchecked_executor_does_not_import_removed_gate_modules() -> None:
    script_path = Path(unchecked.__file__)
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert {
        "arm_runtime",
        "plans",
        "replay_ik_trajectory_guarded",
        "replay_timed_candidate",
        "_native_plan_audit",
    }.isdisjoint(imported)
    assert source.index("trajectories = [") < source.index("core.Panda(")
    assert "HIGH RISK" in unchecked.RISK_BANNER
    assert "slow move" in unchecked.RISK_BANNER
