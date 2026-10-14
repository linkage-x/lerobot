"""Check world -> fr3_base edges at a ball that the FR3 and the handheld UMI both pivoted about.

A ball fixed to the table is one physical point with two independent addresses:

* the FR3's FK pivot about it (``tools/fr3/fr3_tcp_pivot.py collect`` then
  ``fit --json``, or several runs on one ball through ``fit-joint --json`` with one
  shared centre) puts its centre ``c`` in **fr3_base** -- joint angles and the
  URDF, nothing else;
* the handheld UMI pivoted about the same ball, recorded as an ordinary episode,
  puts it at ``b`` in the **camera world** -- production's EE labels, fitted as a
  pivot ``R_i s + t_i = b``.

An edge that is right maps one onto the other: ``e = T_world_base c - b`` is
what that edge is off by *there*, in the world a label is recorded in.  Neither
side uses a TCP definition: each pivot fits its own ``s`` (the socket centre
in that tool's frame), so a wrong ``corenetic_socket_ee`` or a wrong marker->TCP
constant moves ``s``, not the ball.  No laser and no station either -- which is
what makes this a referee between a single-tag edge and a laser edge.

One ball position checks the edge's translation where the ball was; positions
~0.3 m apart also see its rotation (1 deg is 5 mm at 0.3 m).  With three or more
non-collinear positions the pairs are a registration of their own, reported as
``ball_edge`` (rigid, no scale); with four or more each position is also
predicted from the others (leave one out), which is what its verdict rests on.
``edge`` writes a ball edge out as a ``register_fr3_base.py`` candidate report,
so the calibration page can compare it with the live edge and swap it in.

The handheld's ``s`` is one constant -- the same insert in the same gripper at
every position -- so by default it is fitted once over all positions
(``--handheld-s shared``), each position keeping its own centre.  Fitted per
position it wandered by ~10 mm on 2026-10-14 (its z from -3.8 to +6.2 mm): the
centre and ``s`` trade off along directions a hand sweep does not cover, so
label errors that change with the tool's orientation land in ``s`` and drag the
centre along.  Sharing ``s`` took the five-position ball edge's leave-one-out
error from ~7 mm to ~3 mm.  The per-position fit is still reported: its ``s``
is where the ball sits relative to production's TCP, i.e. the marker->TCP
constant's error (0 if the TCP is the socket centre, as bundle 0929 defines it).

Pure numpy; the pivot maths is ``lerobot.robots.franka_research3.tcp_pivot``,
loaded by file so the lerobot package (torch, ...) is not imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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

from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402


def _load_tcp_pivot():
    path = _REPO_ROOT / "src" / "lerobot" / "robots" / "franka_research3" / "tcp_pivot.py"
    spec = importlib.util.spec_from_file_location("_fr3_tcp_pivot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


_tcp_pivot = _load_tcp_pivot()
fit_pivot = _tcp_pivot.fit_pivot
fit_pivot_joint = _tcp_pivot.fit_pivot_joint

SCHEMA = "fr3_base_ball_check/v1"
SIDECAR_NAME = "april_cube_tracking_in_robot_base"
#: fr3_tcp_pivot.py fit-joint's report: ``fk.c_base_mm`` is a list of centres, one if shared.
FR3_JOINT_SCHEMA = "fr3_pivot_joint/v1"
CUBES = ("right", "left", "head")
#: Below these a pivot's centre is coupled to its lever arm and moves with it.
GATE_MIN_SPAN_DEG = 20.0
GATE_MIN_WEAK_SINGULAR = 0.1
#: Jackknife 1-sigma of the centre: FK is sub-0.1 mm, camera labels ~1 mm per frame.
GATE_FR3_CENTRE_SIGMA_MM = 0.3
GATE_HANDHELD_CENTRE_SIGMA_MM = 1.0
GATE_MIN_HANDHELD_FRAMES = 60
#: Kept labels farther than this from one point: not a pivot about a fixed ball (lifted, or the ball moved).
GATE_HANDHELD_RESIDUAL_P95_MM = 6.0
#: Share of *all* solved frames within that distance. The fit keeps one frame per orientation (the
#: first), so a ball moved and pivoted again through the same orientations is invisible to its
#: residual; approach and lift-off at the ends of the episode are why this is not 1.
GATE_MIN_ON_BALL_FRACTION = 0.6
#: Camera labels scatter by mm per frame; the FK pivot's 0.5 mm would drop most of them.
HANDHELD_REJECT_MM = 3.0
#: ...but a frame farther than this many times the kept frames' median residual is off the ball. At 3x a
#: 3 mm label scatter keeps everything: 10-14 the insert slid ~6 mm off the ball for 3-4 s of a 20 s
#: pivot, and those frames moved the ball 8 mm (sigma 5 mm); at 2x they go and sigma is under 1 mm.
HANDHELD_REJECT_K = 2.0
#: shared: one handheld ``s`` for every position of a world (the default); per-position: each its own.
HANDHELD_S_MODES = ("shared", "per-position")
#: A ball edge needs this many positions to be checked: each one predicted from the rest.
GATE_BALL_EDGE_MIN_POSITIONS = 4
#: ...and every held-out position predicted this well. As register_fr3_base.GATE_OOS_P95_MM: the
#: number is what an exported pose would be off by inside the workspace.
GATE_BALL_EDGE_LOO_MAX_MM = 6.0
#: Second singular value of the centres, metres: below it they are nearly on a line and the rotation
#: about that line is free.
GATE_BALL_EDGE_MIN_SPREAD_M = 0.05
#: ``edge.method`` of a ball edge's report (register_fr3_base.py's schema).
BALL_EDGE_METHOD = "fr3_ball_pivot_check"
EDGE_REPORT_SCHEMA = "fr3_base_registration_report/v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _quat_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_labels(csv_path: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """Production's solved TCP poses of one episode, world frame: ``(R (n,3,3), t (n,3))``."""
    R, t = [], []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row.get("episode_index") or 0)) != episode:
                    continue
                if str(row.get("solve_ok", "True")).strip().lower() not in ("true", "1"):
                    continue
                pos = [float(row[f"state_{a}_m"]) for a in "xyz"]
                quat = [float(row[f"state_q{a}"]) for a in "xyzw"]
            except (KeyError, ValueError):
                continue
            if not all(math.isfinite(v) for v in pos + quat):
                continue
            R.append(_quat_matrix(*quat))
            t.append(pos)
    return np.asarray(R).reshape(-1, 3, 3), np.asarray(t, dtype=np.float64).reshape(-1, 3)


def _episode_world(dataset: Path, episode: int) -> str:
    meta_path = dataset / "episodes" / f"episode_{episode:06d}" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return wp.world_frame_id_of(meta.get("world_frame"))


def _pick_cube(sidecar: Path, episode: int, cube: str) -> tuple[str, np.ndarray, np.ndarray]:
    """The requested cube, or the one with the most solved frames (only the BOX on the ball moves)."""
    best = ("", np.zeros((0, 3, 3)), np.zeros((0, 3)))
    for name in (cube,) if cube else CUBES:
        path = sidecar / f"state_action.{name}.csv"
        if path.is_file():
            R, t = read_labels(path, episode)
            if len(t) > len(best[2]):
                best = (name, R, t)
    return best


def _fit_row(fit: Any) -> dict[str, Any]:
    d = fit.as_dict()
    return {
        "s_mm": d["s_ee_mm"],
        "s_sigma_mm": d["s_sigma_mm"],
        "centre_mm": d["c_base_mm"],
        "centre_sigma_mm": d["c_sigma_mm"],
        "residual_rms_mm": d["residual_rms_mm"],
        "residual_p95_mm": d["residual_p95_mm"],
        "n_used": d["n_used"],
        "n_total": d["n_total"],
        "span_deg": d["span_deg"],
        "weak_singular": d["weak_singular"],
    }


def _conditioning_warnings(who: str, row: dict[str, Any], sigma_gate_mm: float) -> list[str]:
    out = []
    if row["span_deg"] < GATE_MIN_SPAN_DEG or row["weak_singular"] < GATE_MIN_WEAK_SINGULAR:
        out.append(
            f"{who} 转得不够（总转角 {row['span_deg']:.0f}°、弱方向 {row['weak_singular']:.2f}；"
            f"要两个方向各 ±{GATE_MIN_SPAN_DEG:.0f}° 以上），球心和杠杆臂分不开"
        )
    sigma = float(np.linalg.norm(row["centre_sigma_mm"]))
    if not math.isfinite(sigma) or sigma > sigma_gate_mm:
        out.append(f"{who} 球心不确定度 {sigma:.2f} mm（> {sigma_gate_mm} mm）")
    return out


def position_row(
    fr3_fit: dict[str, Any], fr3_fit_name: str, dataset: Path, episode: int, *, cube: str = "", sidecar_name: str = SIDECAR_NAME
) -> dict[str, Any]:
    """One ball position: the FR3 pivot's centre in fr3_base, the handheld pivot's centre in the world."""
    row: dict[str, Any] = {"fr3_pivot": fr3_fit_name, "dataset": dataset.name, "episode": episode, "reasons": [], "warnings": []}
    fk = dict(fr3_fit.get("fk") or {})
    if fr3_fit.get("schema") == FR3_JOINT_SCHEMA and "c_base_mm" in fk:
        if not fr3_fit.get("shared_centre") or len(fk["c_base_mm"]) != 1:
            row["reasons"].append(f"{fr3_fit_name} 是每次一个球心的合并拟合，不是一个球位置；用共用球心的合并拟合")
            fk.pop("c_base_mm")
        else:
            fk["c_base_mm"] = fk["c_base_mm"][0]
            fk["c_sigma_mm"] = (fk.get("c_sigma_mm") or [None])[0]
            row["fr3_runs"] = list(fr3_fit.get("runs") or [])
    if "c_base_mm" not in fk:
        if not row["reasons"]:
            row["reasons"].append(f"{fr3_fit_name} 不是 fr3_tcp_pivot.py fit / fit-joint 的结果（没有 fk.c_base_mm）")
    else:
        row["fr3"] = {
            "s_mm": fk.get("s_ee_mm"),
            "s_sigma_mm": fk.get("s_sigma_mm"),
            "centre_mm": fk["c_base_mm"],
            "centre_sigma_mm": fk.get("c_sigma_mm") or [float("nan")] * 3,
            "residual_rms_mm": fk.get("residual_rms_mm"),
            "residual_p95_mm": fk.get("residual_p95_mm"),
            "n_used": fk.get("n_used"),
            "n_total": fk.get("n_total"),
            "span_deg": fk.get("span_deg"),
            "weak_singular": fk.get("weak_singular"),
        }
        socket = fr3_fit.get("socket_frame") or {}
        if "s_ee_mm" in socket:
            row["fr3"]["s_in_socket_frame_mm"] = socket["s_ee_mm"]
        row["warnings"] += _conditioning_warnings("FR3 pivot", row["fr3"], GATE_FR3_CENTRE_SIGMA_MM)

    world = _episode_world(dataset, episode)
    row["world"] = world
    if not world:
        row["reasons"].append(f"{dataset.name} episode {episode} 没有 ok 的 world 戳，不知道标签在哪个世界系")
    name, R, t = _pick_cube(dataset / "derived" / sidecar_name, episode, cube)
    row["cube"] = name
    if len(t) < GATE_MIN_HANDHELD_FRAMES:
        row["reasons"].append(
            f"{dataset.name} episode {episode} 只有 {len(t)} 帧解出的 EE 位姿（< {GATE_MIN_HANDHELD_FRAMES}）；先生成 EE 轨迹"
        )
    else:
        try:
            fit = fit_pivot(R, t, reject_mm=HANDHELD_REJECT_MM, reject_k=HANDHELD_REJECT_K)
            row["handheld"] = _fit_row(fit)
            row["_poses"] = (R, t)  # for the shared-s fit across positions; dropped from the report
            r_all = np.linalg.norm(np.einsum("nij,j->ni", R, fit.s_ee_m) + t - fit.c_base_m, axis=1) * 1e3
            on_ball = float(np.mean(r_all <= GATE_HANDHELD_RESIDUAL_P95_MM))
            row["handheld"]["on_ball_fraction"] = round(on_ball, 3)
            p95 = row["handheld"]["residual_p95_mm"]
            if on_ball < GATE_MIN_ON_BALL_FRACTION:
                row["reasons"].append(
                    f"手持这段只有 {on_ball:.0%} 的帧在拟合出的球心 {GATE_HANDHELD_RESIDUAL_P95_MM:.0f} mm 以内："
                    "球中途动过，或者大部分时间插件不在球上"
                )
            elif p95 > GATE_HANDHELD_RESIDUAL_P95_MM:
                row["reasons"].append(
                    f"手持这段不是绕一个固定点转（残差 p95 {p95:.1f} mm > {GATE_HANDHELD_RESIDUAL_P95_MM} mm）："
                    "插件离开了球，或者球动了"
                )
            row["warnings"] += _conditioning_warnings("手持 pivot", row["handheld"], GATE_HANDHELD_CENTRE_SIGMA_MM)
        except ValueError as exc:
            row["reasons"].append(f"手持 pivot 拟合失败：{exc}")
    row["usable"] = not row["reasons"]
    return row


def shared_handheld(rows: list[dict[str, Any]], world: str) -> dict[str, Any] | None:
    """One handheld ``s`` for every usable position in ``world``, a centre each (``handheld_shared`` per row).

    Rejection as the per-position fit, each position against its own median (labels are noisier where
    fewer cameras saw the tool). ``None`` with fewer than two positions.
    """
    idx = [i for i, r in enumerate(rows) if r["usable"] and r["world"] == world and "_poses" in r]
    if len(idx) < 2:
        return None
    try:
        fit = fit_pivot_joint(
            [rows[i]["_poses"][0] for i in idx],
            [rows[i]["_poses"][1] for i in idx],
            shared_centre=False,
            reject_mm=HANDHELD_REJECT_MM,
            reject_k=HANDHELD_REJECT_K,
            per_run_median=True,
        )
    except ValueError as exc:
        return {"positions": idx, "error": f"共用杠杆臂的手持拟合失败：{exc}；按每个位置各自的杠杆臂算"}
    d = fit.as_dict()
    for k, i in enumerate(idx):
        own = np.asarray(rows[i]["handheld"]["centre_mm"])
        rows[i]["handheld_shared"] = {
            "centre_mm": d["c_base_mm"][k],
            "centre_sigma_mm": d["c_sigma_mm"][k],
            "residual_rms_mm": d["run_rms_mm"][k],
            "n_used": d["run_n_used"][k],
            "shift_mm": np.round(np.asarray(d["c_base_mm"][k]) - own, 2).tolist(),
        }
    return {
        "positions": idx,
        "s_mm": d["s_ee_mm"],
        "s_sigma_mm": d["s_sigma_mm"],
        "s_per_position_mm": [rows[i]["handheld"]["s_mm"] for i in idx],
        "residual_rms_mm": d["residual_rms_mm"],
        "residual_p95_mm": d["residual_p95_mm"],
        "span_deg": d["span_deg"],
        "weak_singular": d["weak_singular"],
    }


def handheld_centre_mm(row: dict[str, Any]) -> list[float]:
    """The ball in the world: from the shared-``s`` fit when there is one, else the position's own."""
    return (row.get("handheld_shared") or row["handheld"])["centre_mm"]


def edge_errors(T_fr3_base_world: np.ndarray, rows: list[dict[str, Any]], world: str) -> dict[str, Any]:
    """``T_world_base c - b`` at every usable position recorded in ``world``, mm, world axes."""
    T_world_base = np.linalg.inv(T_fr3_base_world)
    per = []
    for i, row in enumerate(rows):
        if not row["usable"] or row["world"] != world:
            continue
        c = np.asarray(row["fr3"]["centre_mm"]) / 1e3
        b = np.asarray(handheld_centre_mm(row)) / 1e3
        e = (T_world_base[:3, :3] @ c + T_world_base[:3, 3] - b) * 1e3
        per.append({"position": i, "error_mm": np.round(e, 2).tolist(), "norm_mm": round(float(np.linalg.norm(e)), 2)})
    if not per:
        return {"positions": []}
    E = np.array([p["error_mm"] for p in per])
    n = np.linalg.norm(E, axis=1)
    return {
        "positions": per,
        "mean_mm": np.round(E.mean(0), 2).tolist(),
        "rms_mm": round(float(np.sqrt((n**2).mean())), 2),
        "max_mm": round(float(n.max()), 2),
    }


def _kabsch(C: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid ``(R, t)`` with ``R c + t ~ b``."""
    U, _, Vt = np.linalg.svd((C - C.mean(0)).T @ (B - B.mean(0)))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, B.mean(0) - R @ C.mean(0)


def _spread_m(C: np.ndarray) -> float:
    return float(np.linalg.svd(C - C.mean(0), compute_uv=False)[1])


def ball_edge(rows: list[dict[str, Any]], world: str, *, handheld_s: str = "shared") -> dict[str, Any] | None:
    """The edge the ball pairs alone imply (Kabsch, rigid), when three or more positions span a plane.

    Rigid on purpose: the FR3 base and the camera world are both physical frames. The camera world's own
    scale (0.4-0.7 % short of the FR3's FK, 2026-10-14) is reported, not absorbed -- it is fixed at its
    source, then the edge is refitted from the same balls. The verdict rests on leave one out: each
    position predicted by the edge from the others.
    """
    idx = [i for i, r in enumerate(rows) if r["usable"] and r["world"] == world]
    if len(idx) < 3:
        return None
    C = np.array([rows[i]["fr3"]["centre_mm"] for i in idx]) / 1e3
    B = np.array([handheld_centre_mm(rows[i]) for i in idx]) / 1e3
    sv = np.linalg.svd(C - C.mean(0), compute_uv=False)
    if sv[1] < GATE_BALL_EDGE_MIN_SPREAD_M:
        return {"error": f"球的位置几乎共线（第二奇异值 {sv[1] * 1e3:.0f} mm），旋转定不出来"}
    R, t = _kabsch(C, B)
    res = np.linalg.norm(C @ R.T + t - B, axis=1) * 1e3
    T_world_base = np.eye(4)
    T_world_base[:3, :3], T_world_base[:3, 3] = R, t
    # similarity scale, for reading: how much shorter the camera world reads the same balls
    X, Y = C - C.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    scale = float(np.trace(np.diag(S) @ D) / (X**2).sum())

    reasons: list[str] = []
    warnings: list[str] = []
    loo: list[float | None] = []
    turn: list[float] = []
    if len(idx) < GATE_BALL_EDGE_MIN_POSITIONS:
        reasons.append(f"只有 {len(idx)} 个球位：要 ≥ {GATE_BALL_EDGE_MIN_POSITIONS} 个，每个才能由其余的边预测（留一检验）")
    else:
        for j in range(len(idx)):
            keep = [k for k in range(len(idx)) if k != j]
            if _spread_m(C[keep]) < GATE_BALL_EDGE_MIN_SPREAD_M:
                loo.append(None)
                continue
            Rj, tj = _kabsch(C[keep], B[keep])
            loo.append(float(np.linalg.norm(Rj @ C[j] + tj - B[j]) * 1e3))
            cos = (np.trace(Rj @ R.T) - 1) / 2
            turn.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        if None in loo:
            missing = [idx[j] + 1 for j, v in enumerate(loo) if v is None]
            reasons.append(f"去掉位置 {missing} 后其余几乎共线，预测不了它")
        worst = max((v for v in loo if v is not None), default=0.0)
        if worst > GATE_BALL_EDGE_LOO_MAX_MM:
            reasons.append(f"留一预测最差 {worst:.1f} mm（> {GATE_BALL_EDGE_LOO_MAX_MM} mm）")
    if handheld_s != "shared":
        warnings.append("手持按每个位置各自的杠杆臂拟合：球心随杠杆臂漂，留一误差偏大")
    for i in idx:
        if rows[i]["warnings"]:
            warnings.append(f"位置 {i + 1}：" + "；".join(rows[i]["warnings"]))
    known = [v for v in loo if v is not None]
    return {
        "positions": len(idx),
        "position_indices": idx,
        "extent_mm": np.round(sv * 1e3, 1).tolist(),
        "T_world_base": T_world_base.tolist(),
        "residual_mm": np.round(res, 2).tolist(),
        "residual_rms_mm": round(float(np.sqrt(np.mean(res**2))), 2),
        "leave_one_out_mm": [None if v is None else round(v, 2) for v in loo],
        "leave_one_out_rms_mm": round(float(np.sqrt(np.mean(np.square(known)))), 2) if known else None,
        "leave_one_out_max_mm": round(max(known), 2) if known else None,
        "rotation_spread_deg": round(max(turn), 3) if turn else None,
        "scale_pct": round((scale - 1) * 100, 3),
        "handheld_s": handheld_s,
        "verdict": "ok" if not reasons else "rejected",
        "reasons": reasons,
        "warnings": warnings,
    }


def _report_edge(path: Path) -> tuple[str, str, np.ndarray, bool] | None:
    """``(label, source world, T_fr3_base_world, fitted on balls)`` of a register_fr3_base.py candidate report."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        edge = report["edge"]
        T = np.asarray(edge["T_to_from"], dtype=np.float64)
    except (OSError, ValueError, KeyError):
        return None
    src, dst = str(edge.get("from_world_frame_id")), str(edge.get("to_world_frame_id"))
    from_balls = str(edge.get("method") or "").startswith(BALL_EDGE_METHOD)
    if dst == wp.FR3_BASE_WORLD_ID:
        return path.name, src, T, from_balls
    if src == wp.FR3_BASE_WORLD_ID:
        return path.name, dst, np.linalg.inv(T), from_balls
    return None


def check(
    rows: list[dict[str, Any]], graph: dict[str, Any], candidate_reports: list[Path], *, handheld_s: str = "shared"
) -> dict[str, Any]:
    worlds = sorted({r["world"] for r in rows if r["usable"]})
    shared = {}
    if handheld_s == "shared":
        shared = {w: sh for w in worlds if (sh := shared_handheld(rows, w)) is not None}
    for row in rows:
        row.pop("_poses", None)
    edges = []
    for world in worlds:
        live = wp.world_transform(graph, world, wp.FR3_BASE_WORLD_ID)
        if live is not None:
            hops = live[1]
            edges.append(
                {
                    "name": "live",
                    "world": world,
                    "method": hops[-1]["method"] if hops else "",
                    "created_utc": hops[-1]["created_utc"] if hops else "",
                    **edge_errors(np.asarray(live[0]), rows, world),
                }
            )
    for path in candidate_reports:
        found = _report_edge(path)
        if found is None:
            edges.append({"name": path.name, "error": "不是一个 world <-> fr3_base 的候选报告"})
            continue
        label, world, T, from_balls = found
        if world not in worlds:
            edges.append({"name": label, "world": world, "error": f"候选边的世界系 {world} 不是这些录制的世界系"})
            continue
        # a ball edge scored on balls it was fitted to reads in-sample small
        edges.append({"name": label, "world": world, "from_balls": from_balls, **edge_errors(T, rows, world)})
    return {
        "schema": SCHEMA,
        "generated_utc": _now_iso(),
        "handheld_s": handheld_s,
        "handheld_shared": shared,
        "positions": rows,
        "edges": edges,
        "ball_edges": {w: be for w in worlds if (be := ball_edge(rows, w, handheld_s=handheld_s)) is not None},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_report(
    check_report: dict[str, Any], check_path: str, world: str, *, target_world: str = wp.FR3_BASE_WORLD_ID
) -> dict[str, Any]:
    """A check's ball edge as a register_fr3_base.py candidate report (``apply`` / the calibration page take it).

    ``world_reference_sha256`` is the one the check ran against: the handheld centres are only valid with
    the camera poses their labels were made with, and ``apply`` refuses the report once that file changes.
    """
    be = (check_report.get("ball_edges") or {}).get(world)
    if not be or "error" in be:
        raise ValueError(f"{check_path} has no ball edge in {world}" + (f": {be['error']}" if be else ""))
    reasons = list(be.get("reasons") or [])
    sha = str(check_report.get("world_reference_sha256") or "")
    if not sha:
        reasons.append("检验报告没有记下 world_reference.json 的哈希（旧版检验），重新跑一次同球检验")
    elif check_report.get("reference_world") != world:
        reasons.append(f"检验时的基准世界系是 {check_report.get('reference_world')}，不是 {world}")
    T_world_base = np.asarray(be["T_world_base"], dtype=np.float64)
    positions = [check_report["positions"][i] for i in be["position_indices"]]
    shared = (check_report.get("handheld_shared") or {}).get(world) or {}
    return {
        "schema": EDGE_REPORT_SCHEMA,
        "generated_utc": _now_iso(),
        "verdict": "ok" if not reasons else "rejected",
        "reasons": reasons,
        "warnings": list(be.get("warnings") or []),
        "edge": {
            "from_world_frame_id": world,
            "to_world_frame_id": target_world,
            "T_to_from": np.linalg.inv(T_world_base).tolist(),
            "covariance_6x6": None,
            "method": (
                f"{BALL_EDGE_METHOD}: a ball fixed to the table, pivoted about by the FR3 (FK; centre in fr3_base) "
                f"and by the handheld UMI (production labels, cameras at their {world} poses; centre in the world, "
                f"handheld s {be.get('handheld_s')}) at {len(positions)} positions; rigid Kabsch, no scale"
            ),
            "source": {
                "ball_check": check_path,
                "positions": [
                    {"fr3_pivot": p["fr3_pivot"], "fr3_runs": p.get("fr3_runs"), "dataset": p["dataset"], "episode": p["episode"]}
                    for p in positions
                ],
                "world_reference_sha256": sha,
            },
            "quality": {
                "positions": be["positions"],
                "extent_mm": be["extent_mm"],
                "residual_mm": be["residual_mm"],
                "residual_rms_mm": be["residual_rms_mm"],
                "leave_one_out_mm": be["leave_one_out_mm"],
                "leave_one_out_rms_mm": be["leave_one_out_rms_mm"],
                "leave_one_out_max_mm": be["leave_one_out_max_mm"],
                "rotation_spread_deg": be["rotation_spread_deg"],
                "scale_pct": be["scale_pct"],
                "handheld_s": be.get("handheld_s"),
                "handheld_shared_s_mm": shared.get("s_mm"),
            },
            "valid_while": (
                f"the rig cameras stay at their {world} poses and the FR3 is not re-mounted; either change, or a "
                "fix of the camera world's scale, needs the ball check run again and the edge refitted"
            ),
        },
        "target_node": {
            "world_frame_id": target_world,
            "calibration_id": "",
            "parent_world_frame_id": None,
            "reason": "robot_base",
            "definition": "FR3 base frame (libfranka O frame) of the arm at its current mount",
        },
        # where the balls were: the workspace this edge was measured in ("how far does it move the live edge")
        "workspace_points_base_m": [np.round(np.asarray(p["fr3"]["centre_mm"]) / 1e3, 5).tolist() for p in positions],
    }


def _resolve_dataset(raw: str, datasets_root: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and datasets_root is not None and not path.exists():
        path = datasets_root / path
    return path.resolve()


def _edge_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="fr3_base_ball_check.py edge", description="write a check's ball edge as a register_fr3_base.py candidate report"
    )
    parser.add_argument("--check", type=Path, required=True, help="check_*.json written by this tool")
    parser.add_argument("--world", default="", help="default: the check's reference world")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    check_report = json.loads(args.check.read_text(encoding="utf-8"))
    world = args.world or str(check_report.get("reference_world") or "")
    try:
        report = edge_report(check_report, _repo_relative(args.check.resolve()), world)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    q = report["edge"]["quality"]
    print(
        f"ball edge {world} -> fr3_base: {q['positions']} positions, residual rms {q['residual_rms_mm']} mm, "
        f"leave-one-out rms {q['leave_one_out_rms_mm']} / max {q['leave_one_out_max_mm']} mm"
    )
    print(f"verdict: {report['verdict']}" + (f" ({'; '.join(report['reasons'])})" if report["reasons"] else ""))
    print(f"written: {args.out}")
    return 0 if report["verdict"] == "ok" else 2


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["edge"]:
        return _edge_main(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog="'edge --check CHECK --out REPORT' writes a check's ball edge as a candidate"
    )
    parser.add_argument(
        "--position",
        nargs=3,
        action="append",
        required=True,
        metavar=("FR3_PIVOT_FIT", "DATASET", "EPISODE"),
        help="fr3_tcp_pivot.py fit (or fit-joint, one shared centre) --json output, and the handheld episode "
        "pivoted about the same ball; repeat per ball position",
    )
    parser.add_argument("--cube", default="", help="handheld cube (default: the one with the most solved frames)")
    parser.add_argument(
        "--handheld-s",
        choices=HANDHELD_S_MODES,
        default="shared",
        help="one handheld lever arm for all positions (default; the insert does not change), or one each",
    )
    parser.add_argument("--candidate", action="append", type=Path, default=[], help="register_fr3_base.py report to check too")
    parser.add_argument("--datasets-root", type=Path, default=None)
    parser.add_argument("--world-dir", type=Path, default=None, help="default: tools/thor/gmsl2/world")
    parser.add_argument("--sidecar", default=SIDECAR_NAME)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    world_dir = (args.world_dir or _REPO_ROOT / wp.WORLD_SUBDIR).resolve()
    graph_path = world_dir / wp.WORLD_GRAPH_FILE
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else {"edges": []}
    rows = []
    for fit_raw, dataset_raw, episode_raw in args.position:
        fit_path = Path(fit_raw).resolve()
        fr3_fit = json.loads(fit_path.read_text(encoding="utf-8"))
        dataset = _resolve_dataset(dataset_raw, args.datasets_root)
        rows.append(position_row(fr3_fit, fit_path.name, dataset, int(episode_raw), cube=args.cube, sidecar_name=args.sidecar))
    report = check(rows, graph, args.candidate, handheld_s=args.handheld_s)
    reference_path = world_dir / wp.WORLD_REFERENCE_FILE
    if reference_path.is_file():
        report["reference_world"] = str(json.loads(reference_path.read_text(encoding="utf-8")).get("world_frame_id") or "")
        report["world_reference_sha256"] = _sha256(reference_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for i, row in enumerate(rows):
        head = f"position {i}: {row['fr3_pivot']} + {row['dataset']} ep{row['episode']} ({row['cube'] or '-'})"
        if not row["usable"]:
            print(f"{head}: unusable -- {'; '.join(row['reasons'])}")
            continue
        fr3, hh = row["fr3"], row["handheld"]
        print(
            f"{head}: ball in fr3_base {fr3['centre_mm']} mm, in {row['world']} {hh['centre_mm']} mm; "
            f"handheld s {hh['s_mm']} mm (residual p95 {hh['residual_p95_mm']} mm, span {hh['span_deg']} deg)"
        )
        if "handheld_shared" in row:
            print(f"  with the shared handheld s: ball in {row['world']} {row['handheld_shared']['centre_mm']} mm")
        for w in row["warnings"]:
            print(f"  warning: {w}")
    for world, sh in report["handheld_shared"].items():
        print(f"handheld s shared over {len(sh['positions'])} positions in {world}: {sh.get('s_mm', sh.get('error'))}")
    for edge in report["edges"]:
        if "error" in edge:
            print(f"edge {edge['name']}: {edge['error']}")
        elif edge["positions"]:
            print(f"edge {edge['name']}: mean {edge['mean_mm']} mm, rms {edge['rms_mm']} mm, max {edge['max_mm']} mm")
    for world, be in report["ball_edges"].items():
        if "error" in be:
            print(f"ball edge {world}: {be['error']}")
            continue
        print(
            f"ball edge {world}: {be['positions']} positions, residual rms {be['residual_rms_mm']} mm, leave-one-out "
            f"{be['leave_one_out_mm']} mm (rms {be['leave_one_out_rms_mm']}), scale {be['scale_pct']:+.2f} %, "
            f"verdict {be['verdict']}" + (f" ({'; '.join(be['reasons'])})" if be["reasons"] else "")
        )
    print(f"written: {args.out}")
    return 0 if any(r["usable"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
