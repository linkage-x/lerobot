# ruff: noqa: N806 -- matrices keep their maths names (R, T_left_right, CL/CR)
"""Measure the edge between the two FR3 bases from balls both arms pivoted about (roadmap B3).

A ball fixed to the table, pivoted about by the left FR3 and then by the right one, is one physical
point with two FK addresses: ``c_L`` in the left arm's base (``fr3_base``) and ``c_R`` in the right
arm's (``fr3_base_right``). Each comes from ``tools/fr3/fr3_tcp_pivot.py fit`` (or ``fit-joint`` with one
shared centre) -- joint angles and the URDF, nothing else: no camera, no TCP definition (each pivot fits
its own lever arm), no tag. Over four or more ball positions spanning a plane, the rigid fit
``R c_R + t = c_L`` is ``T_fr3_base_fr3_base_right``.

Why this and not two world edges: each world edge carries the camera side's error (labels, extrinsics,
the world's scale), and ``T_LR = inv(T_world_L) T_world_R`` adds both, uncorrelated. Here both sides are
FK; the 10-23 simulation (dual_fr3_registration_sim.py, G3) put T_LR p95 at ~3.0 mm this way against
~4.8 mm through the world, the floor being one arm's FK error (~0.5 mm). The world edges are still
needed for absolute replay (B4); with both present the graph checks the loop (``world_closures``).

The verdict rests on leave one out, as the world ball check's: each position predicted from the others.
The report is a ``register_fr3_base.py`` candidate (``apply`` / the calibration page's FR3 base panel),
written from ``fr3_base`` to ``fr3_base_right`` with an explicit ``sigma_mm`` (its leave-one-out rms), so
a loop through the world is judged against what this edge claims.

Pure numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import fr3_base_ball_check as bc, world_provenance as wp  # noqa: E402

#: ``edge.method`` prefix; world_provenance maps it to the ``arm_arm_ball`` kind.
METHOD = "fr3_arm_arm_ball"
EDGE_REPORT_SCHEMA = bc.EDGE_REPORT_SCHEMA
LEFT_ROOT = wp.FR3_BASE_WORLD_ID
RIGHT_ROOT = "fr3_base_right"
GATE_MIN_POSITIONS = 4
#: Both sides are FK (each centre sub-0.3 mm): a held-out ball predicted worse than this is a ball that
#: moved between the two arms' pivots, or a pivot that did not turn enough -- not FK error. G3's p95 for
#: the whole edge was 3.0 mm at the workspace.
GATE_LOO_MAX_MM = 3.0
GATE_MIN_SPREAD_M = bc.GATE_BALL_EDGE_MIN_SPREAD_M
#: The sigma an edge claims is never below this: a near-zero leave-one-out is luck, not FK.
MIN_SIGMA_MM = 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


def side_row(fit: dict[str, Any], name: str, robot_ip: str, expected_ip: str, who: str) -> dict[str, Any]:
    """One arm's pivot at one ball: its centre in its base, and why it cannot be used."""
    fr3, reasons, warnings, runs = bc.fr3_centre(fit, name)
    row: dict[str, Any] = {
        "fit": name,
        "robot_ip": robot_ip,
        "reasons": list(reasons),
        "warnings": list(warnings),
    }
    if runs is not None:
        row["runs"] = runs
    if fr3 is not None:
        row["fr3"] = fr3
    if expected_ip and robot_ip != expected_ip:
        row["reasons"].append(
            f"{who}的 pivot {name} 是在 {robot_ip or '未记录 IP 的臂'} 上采的，{who}是 {expected_ip}"
        )
    return row


def solve(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    left_root: str = LEFT_ROOT,
    right_root: str = RIGHT_ROOT,
) -> dict[str, Any]:
    """Pairs of (left side_row, right side_row), one per ball position -> a register_fr3_base candidate report."""
    reasons: list[str] = []
    warnings: list[str] = []
    positions = []
    for i, (left, right) in enumerate(pairs, start=1):
        usable = "fr3" in left and "fr3" in right and not left["reasons"] and not right["reasons"]
        positions.append({"left": left, "right": right, "usable": usable})
        for side in (left, right):
            for w in side["warnings"]:
                warnings.append(f"位置 {i}：{w}")
        if not usable:
            reasons.append(f"位置 {i} 不能用：" + "；".join(left["reasons"] + right["reasons"]))
    ips = {p["left"]["robot_ip"] for p in positions} | {p["right"]["robot_ip"] for p in positions}
    if any(p["left"]["robot_ip"] and p["left"]["robot_ip"] == p["right"]["robot_ip"] for p in positions):
        reasons.append("同一位置左右两边是同一台臂的 pivot")
    idx = [i for i, p in enumerate(positions) if p["usable"]]
    report: dict[str, Any] = {
        "schema": EDGE_REPORT_SCHEMA,
        "generated_utc": _now_iso(),
        "positions": positions,
    }
    if len(idx) < 3:
        reasons.append(
            f"只有 {len(idx)} 个可用球位：至少 3 个才定得出旋转，{GATE_MIN_POSITIONS} 个才能留一检验"
        )
        return {**report, "verdict": "rejected", "reasons": reasons, "warnings": warnings}
    CL = np.array([positions[i]["left"]["fr3"]["centre_mm"] for i in idx], dtype=np.float64) / 1e3
    CR = np.array([positions[i]["right"]["fr3"]["centre_mm"] for i in idx], dtype=np.float64) / 1e3
    sv = np.linalg.svd(CR - CR.mean(0), compute_uv=False)
    if sv[1] < GATE_MIN_SPREAD_M:
        reasons.append(f"球的位置几乎共线（第二奇异值 {sv[1] * 1e3:.0f} mm），旋转定不出来")
        return {**report, "verdict": "rejected", "reasons": reasons, "warnings": warnings}
    R, t = bc._kabsch(CR, CL)
    res = np.linalg.norm(CR @ R.T + t - CL, axis=1) * 1e3
    loo: list[float | None] = []
    turn: list[float] = []
    if len(idx) < GATE_MIN_POSITIONS:
        reasons.append(
            f"只有 {len(idx)} 个球位：要 ≥ {GATE_MIN_POSITIONS} 个，每个才能由其余的预测（留一检验）"
        )
    else:
        for j in range(len(idx)):
            keep = [k for k in range(len(idx)) if k != j]
            if bc._spread_m(CR[keep]) < GATE_MIN_SPREAD_M:
                loo.append(None)
                continue
            Rj, tj = bc._kabsch(CR[keep], CL[keep])
            loo.append(float(np.linalg.norm(Rj @ CR[j] + tj - CL[j]) * 1e3))
            cos = (np.trace(Rj @ R.T) - 1) / 2
            turn.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        if None in loo:
            reasons.append(
                f"去掉位置 {[idx[j] + 1 for j, v in enumerate(loo) if v is None]} 后其余几乎共线，预测不了它"
            )
        worst = max((v for v in loo if v is not None), default=0.0)
        if worst > GATE_LOO_MAX_MM:
            reasons.append(
                f"留一预测最差 {worst:.1f} mm（> {GATE_LOO_MAX_MM} mm）：两台臂之间球动过，或某次 pivot 转得不够"
            )
    known = [v for v in loo if v is not None]
    loo_rms = float(np.sqrt(np.mean(np.square(known)))) if known else None
    T_left_right = np.eye(4)
    T_left_right[:3, :3], T_left_right[:3, 3] = R, t
    used = [positions[i] for i in idx]
    return {
        **report,
        "verdict": "ok" if not reasons else "rejected",
        "reasons": reasons,
        "warnings": warnings,
        "edge": {
            "from_world_frame_id": left_root,
            "to_world_frame_id": right_root,
            # takes fr3_base coordinates into fr3_base_right
            "T_to_from": np.linalg.inv(T_left_right).tolist(),
            "covariance_6x6": None,
            "sigma_mm": round(max(MIN_SIGMA_MM, loo_rms or 0.0), 3) if loo_rms is not None else None,
            "method": (
                f"{METHOD}: a ball fixed to the table, pivoted about by each FR3 in turn (FK; centre in its own "
                f"base) at {len(idx)} positions; rigid Kabsch {right_root} -> {left_root}, no scale"
            ),
            "source": {
                "positions": [
                    {
                        "left_pivot": p["left"]["fit"],
                        "left_runs": p["left"].get("runs"),
                        "right_pivot": p["right"]["fit"],
                        "right_runs": p["right"].get("runs"),
                    }
                    for p in used
                ],
                "robot_ips": {
                    left_root: used[0]["left"]["robot_ip"],
                    right_root: used[0]["right"]["robot_ip"],
                },
            },
            "quality": {
                "positions": len(idx),
                "extent_mm": np.round(sv * 1e3, 1).tolist(),
                "residual_mm": np.round(res, 2).tolist(),
                "residual_rms_mm": round(float(np.sqrt(np.mean(res**2))), 2),
                "leave_one_out_mm": [None if v is None else round(v, 2) for v in loo],
                "leave_one_out_rms_mm": None if loo_rms is None else round(loo_rms, 2),
                "leave_one_out_max_mm": round(max(known), 2) if known else None,
                "rotation_spread_deg": round(max(turn), 3) if turn else None,
                "baseline_m": round(float(np.linalg.norm(t)), 4),
            },
            "valid_while": "neither FR3 is re-mounted (moving either cart or base needs the balls again)",
        },
        "target_node": {
            "world_frame_id": right_root,
            "calibration_id": "",
            "parent_world_frame_id": None,
            "reason": "robot_base",
            "definition": f"FR3 base frame (libfranka O frame) of the right arm ({used[0]['right']['robot_ip'] or '?'}) at its current mount",
        },
        # the balls in the right base: where a swap of this edge moves a right-arm target
        "workspace_points_base_m": np.round(CR, 5).tolist(),
        "robot_ips_seen": sorted(ips - {""}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--position",
        nargs=2,
        action="append",
        required=True,
        metavar=("LEFT_PIVOT_FIT", "RIGHT_PIVOT_FIT"),
        help="the left and the right arm's fr3_tcp_pivot.py fit (or fit-joint, one shared centre) about the same "
        "ball; repeat per ball position",
    )
    parser.add_argument("--left-ip", default="", help="refuse a left pivot collected on another arm")
    parser.add_argument("--right-ip", default="", help="refuse a right pivot collected on another arm")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = []
    for left_raw, right_raw in args.position:
        sides = []
        for raw, expected, who in ((left_raw, args.left_ip, "左臂"), (right_raw, args.right_ip, "右臂")):
            path = Path(raw).resolve()
            fit = json.loads(path.read_text(encoding="utf-8"))
            sides.append(side_row(fit, path.name, bc.fit_robot_ip(fit, path), expected, who))
        pairs.append((sides[0], sides[1]))
    report = solve(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for i, p in enumerate(report["positions"], start=1):
        left, right = p["left"], p["right"]
        centre = lambda side: side["fr3"]["centre_mm"] if "fr3" in side else "-"  # noqa: E731
        print(f"position {i}: left {left['fit']} {centre(left)} mm | right {right['fit']} {centre(right)} mm")
    if "edge" in report:
        q = report["edge"]["quality"]
        print(
            f"edge {LEFT_ROOT} -> {RIGHT_ROOT}: {q['positions']} positions, baseline {q['baseline_m']} m, residual rms "
            f"{q['residual_rms_mm']} mm, leave-one-out {q['leave_one_out_mm']} mm (rms {q['leave_one_out_rms_mm']})"
        )
    for w in report["warnings"]:
        print(f"warning: {w}")
    print(
        f"verdict: {report['verdict']}" + (f" ({'; '.join(report['reasons'])})" if report["reasons"] else "")
    )
    print(f"written: {args.out}")
    if "edge" not in report:
        return 1
    return 0 if report["verdict"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
