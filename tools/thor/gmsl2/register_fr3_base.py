"""Register the FR3 base into a camera world as an edge of ``world_graph.json``.

The production world is the camera rig's own frame (``world_reference.json``);
the FR3 base is a different physical frame, and how the two relate is one rigid
transform that the world graph is designed to hold.  This tool measures it from
a P0 single-tag run (``tools/thor/p0_two_marker_calibration.py``): a tag36h11 on
the FR3 tool, seen by the rig at many arm poses, with the arm's own
``T_base_tcp`` recorded at each.

Unlike the P0 solve, the cameras are **held at the world's poses**.  The P0
solve frees them and so defines a new world (the FR3 base) -- that is what must
not happen behind ``world_reference.json``'s back.  Here the unknowns are only

    T_world_base, T_tcp_tag, and the printed tag's scale,

fitted by fisheye reprojection of the tag corners over every (capture, camera):

    x = proj_c( T_world_c^-1 @ T_world_base @ T_base_tcp(i) @ T_tcp_tag @ (s * X_k) )

Two subcommands, because the inputs live on Thor and the graph lives in git:

* ``solve`` (numpy/scipy/cv2; run on Thor in ``third_party/opencv_kalibr/.venv``)
  writes a report holding the proposed edge, its residuals and a verdict.
* ``apply`` (stdlib) appends that edge, and the target node, to the tracked
  ``world_graph.json``.  It refuses a report whose verdict is not ``ok``, an
  edge between two worlds that are already connected, and a source world that is
  not a node of the graph.

The verdict is out-of-sample: captures are split into folds (by source -- an
imported dataset vs a live session -- or in halves when there is one source),
each fold's edge predicts the other fold's tag in the world, and that prediction
is compared with the tag the fixed cameras measure.  In-sample numbers are
reported, but they do not decide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402

REPORT_SCHEMA = "fr3_base_registration_report/v1"
DEFAULT_TARGET_WORLD = wp.FR3_BASE_WORLD_ID
TAG_ID = 6

#: Verdict gates.  The out-of-sample tag-centre p95 is the number that matters:
#: it is what an exported pose would be off by inside the workspace.
GATE_MIN_CAPTURES = 20
GATE_OOS_P95_MM = 6.0
GATE_TAG_SCALE = (0.97, 1.03)
#: Per-observation corner RMSE above which an observation is dropped after the
#: first (robust) pass.
REJECT_PX = 4.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    """``outputs/...`` for a path inside any checkout, so the record is the same on every machine."""
    parts = path.parts
    return str(Path(*parts[parts.index("outputs") :])) if "outputs" in parts else str(path)


def _now_iso() -> str:
    # timezone.utc, not datetime.UTC: solve runs in opencv_kalibr's Python 3.10 venv on Thor.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: UP017


# ------------------------------------------------------------------- solve ---


def _solve(args: argparse.Namespace) -> int:
    run_dir = args.run.resolve()
    captures_path = run_dir / "captures.json"
    p0_summary_path = run_dir / "camera_calibration" / "summary.json"
    world_dir = (args.world_dir or _REPO_ROOT / wp.WORLD_SUBDIR).resolve()
    reference_path = world_dir / wp.WORLD_REFERENCE_FILE
    intrinsics_path = args.intrinsics.resolve()

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    source_world = str(reference["world_frame_id"])
    calibration_id = str(reference.get("calibration_id") or "")
    intrinsics_run = intrinsics_path.parent.name
    if calibration_id.removesuffix("_extrinsics") != intrinsics_run.removesuffix("_intrinsics"):
        raise SystemExit(
            f"--intrinsics {intrinsics_run} is not the run the world's cameras were solved with "
            f"({calibration_id}); the camera poses are only valid with their own intrinsics"
        )

    import cv2
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def se3(xi):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
        T[:3, 3] = xi[3:6]
        return T

    def xi_of(T):
        return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]

    def inv(T):
        Ti = np.eye(4)
        Ti[:3, :3] = T[:3, :3].T
        Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
        return Ti

    def tag_points(size):
        h = size / 2.0
        return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])

    def project(cam, T_c_x, X):
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T_c_x[:3, :3]))
        tvec = np.ascontiguousarray(T_c_x[:3, 3]).reshape(3, 1)
        uv, _ = cv2.fisheye.projectPoints(np.ascontiguousarray(X).reshape(1, -1, 3), rvec, tvec, cam["K"], cam["D"])
        return uv.reshape(-1, 2)

    cams: dict[str, dict[str, Any]] = {}
    for row in json.loads(intrinsics_path.read_text(encoding="utf-8"))["cameras"]:
        if str(row.get("status", "")).lower() != "ok":
            continue
        name = str(row["camera_name"])
        if name not in (reference.get("cameras") or {}):
            continue
        data = json.loads(Path(row["intrinsics_json"]).read_text(encoding="utf-8"))
        if str(data.get("model", "")).lower() not in {"opencv_fisheye", "fisheye", "equidistant"}:
            raise SystemExit(f"{name}: expected fisheye intrinsics, got {data.get('model')!r}")
        cams[name] = {
            "K": np.asarray(data["camera_matrix"], dtype=np.float64),
            "D": np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(4, 1),
            "size": (int(data["image_width"]), int(data["image_height"])),
            "T_world_c": np.asarray(reference["cameras"][name]["T_world_camera"], dtype=np.float64),
        }

    p0 = json.loads(p0_summary_path.read_text(encoding="utf-8"))
    marker_size = float(p0["marker"]["marker_size_m"])
    joint = p0["joint_solution"]
    T_tcp_tag0 = np.asarray(joint["tool_to_board"]["matrix_4x4"], dtype=np.float64)
    # Initial T_world_base from the P0 solve's own cameras: each camera seen in
    # both frames gives one T_world_c @ T_base_c^-1.
    candidates = [
        cams[name]["T_world_c"] @ inv(np.asarray(entry["base_to_camera"]["matrix_4x4"], dtype=np.float64))
        for name, entry in (joint.get("cameras") or {}).items()
        if name in cams
    ]
    if not candidates:
        raise SystemExit("the P0 run shares no camera with the world reference")
    T_world_base0 = np.eye(4)
    T_world_base0[:3, :3] = Rotation.from_matrix(np.array([c[:3, :3] for c in candidates])).mean().as_matrix()
    T_world_base0[:3, 3] = np.mean([c[:3, 3] for c in candidates], axis=0)

    observations: list[dict[str, Any]] = []
    for record in json.loads(captures_path.read_text(encoding="utf-8"))["records"]:
        source = record.get("source") or {}
        group = _repo_relative(Path(source["dataset_root"])) if source.get("dataset_root") else "live"
        for live_name, item in (record.get("cameras") or {}).items():
            name = str(item.get("calibration_camera", live_name))
            if name not in cams:
                continue
            detections = [d for d in item.get("detections") or [] if int(d.get("tag_id", -1)) == TAG_ID]
            if not detections:
                continue
            det = detections[0]
            if (int(det["image_width"]), int(det["image_height"])) != cams[name]["size"]:
                raise SystemExit(f"{name}: capture size differs from the intrinsics' {cams[name]['size']}")
            observations.append(
                {
                    "capture": int(record["capture_index"]),
                    "group": group,
                    "camera": name,
                    "uv": np.asarray(det["corners_px"], dtype=np.float64).reshape(4, 2),
                    "T_base_tcp": np.asarray(record["T_base_tcp"], dtype=np.float64),
                }
            )
    captures = sorted({o["capture"] for o in observations})
    groups = sorted({o["group"] for o in observations})
    if len(groups) >= 2:
        folds = {g: {o["capture"] for o in observations if o["group"] == g} for g in groups}
    else:
        half = len(captures) // 2
        folds = {"first_half": set(captures[:half]), "second_half": set(captures[half:])}

    def residuals(p, obs):
        T_world_base, T_tcp_tag = se3(p[:6]), se3(p[6:12])
        X = tag_points(marker_size * p[12])
        return np.concatenate(
            [
                (project(cams[o["camera"]], inv(cams[o["camera"]]["T_world_c"]) @ T_world_base @ o["T_base_tcp"] @ T_tcp_tag, X) - o["uv"]).ravel()
                for o in obs
            ]
        )

    def per_obs_rmse(p, obs):
        r = residuals(p, obs).reshape(len(obs), 8)
        return np.sqrt((r**2).mean(axis=1))

    def fit(obs):
        p = np.r_[xi_of(T_world_base0), xi_of(T_tcp_tag0), 1.0]
        p = least_squares(residuals, p, args=(obs,), loss="soft_l1", f_scale=1.5, x_scale="jac").x
        kept = [o for o, e in zip(obs, per_obs_rmse(p, obs), strict=True) if e < REJECT_PX]
        p = least_squares(residuals, p, args=(kept,), loss="soft_l1", f_scale=1.5, x_scale="jac").x
        rmse = per_obs_rmse(p, kept)
        return p, kept, {
            "observations_in": len(obs),
            "observations_kept": len(kept),
            "rmse_px": float(np.sqrt((rmse**2).mean())),
            "p95_px": float(np.percentile(rmse, 95)),
            "per_camera_rmse_px": {
                c: float(np.sqrt(np.mean([e**2 for o, e in zip(kept, rmse, strict=True) if o["camera"] == c])))
                for c in sorted({o["camera"] for o in kept})
            },
        }

    def measured_tag(obs_one_capture, scale, T_init):
        """The tag in the world from the fixed cameras alone (no arm)."""
        X = tag_points(marker_size * scale)

        def f(xi):
            T = se3(xi)
            return np.concatenate(
                [(project(cams[o["camera"]], inv(cams[o["camera"]]["T_world_c"]) @ T, X) - o["uv"]).ravel() for o in obs_one_capture]
            )

        r = least_squares(f, xi_of(T_init), loss="soft_l1", f_scale=1.5)
        return se3(r.x), float(np.sqrt((r.fun**2).mean()))

    def tag_check(p, obs):
        T_world_base, T_tcp_tag = se3(p[:6]), se3(p[6:12])
        by_capture: dict[int, list[dict[str, Any]]] = {}
        for o in obs:
            by_capture.setdefault(o["capture"], []).append(o)
        dist, ang = [], []
        for items in by_capture.values():
            if len(items) < 2:  # one camera cannot place the tag independently of its own pose error
                continue
            predicted = T_world_base @ items[0]["T_base_tcp"] @ T_tcp_tag
            measured, rmse = measured_tag(items, p[12], predicted)
            if rmse > REJECT_PX:
                continue
            delta = inv(predicted) @ measured
            dist.append(float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]) * 1e3))
            ang.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))))
        if not dist:
            return {"captures": 0}
        return {
            "captures": len(dist),
            "tag_centre_p50_mm": float(np.median(dist)),
            "tag_centre_p95_mm": float(np.percentile(dist, 95)),
            "tag_centre_max_mm": float(np.max(dist)),
            "rotation_p50_deg": float(np.median(ang)),
        }

    p_all, kept_all, fit_all = fit(observations)
    fold_reports: dict[str, Any] = {}
    oos_p95: list[float] = []
    for name, held_out in folds.items():
        train = [o for o in observations if o["capture"] not in held_out]
        test = [o for o in kept_all if o["capture"] in held_out]
        p_fold, _, fit_fold = fit(train)
        check = tag_check(p_fold, test)
        delta = inv(se3(p_all[:6])) @ se3(p_fold[:6])
        fold_reports[name] = {
            "held_out_captures": len(held_out),
            "fit_on_the_rest": fit_fold,
            "tag_scale": float(p_fold[12]),
            "T_world_base_vs_all": {
                "translation_mm_at_base_origin": float(np.linalg.norm(delta[:3, 3]) * 1e3),
                "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))),
            },
            "held_out_tag_check": check,
        }
        if check.get("captures"):
            oos_p95.append(check["tag_centre_p95_mm"])

    T_world_base = se3(p_all[:6])
    in_sample = tag_check(p_all, kept_all)
    reasons = []
    if len(captures) < GATE_MIN_CAPTURES:
        reasons.append(f"only {len(captures)} captures (< {GATE_MIN_CAPTURES})")
    if not oos_p95:
        reasons.append("no fold produced an out-of-sample tag check")
    elif max(oos_p95) > GATE_OOS_P95_MM:
        reasons.append(f"out-of-sample tag-centre p95 {max(oos_p95):.2f} mm > {GATE_OOS_P95_MM} mm")
    if not GATE_TAG_SCALE[0] <= p_all[12] <= GATE_TAG_SCALE[1]:
        reasons.append(f"tag scale {p_all[12]:.4f} outside {GATE_TAG_SCALE}")

    delta0 = inv(T_world_base0) @ T_world_base
    report = {
        "schema": REPORT_SCHEMA,
        "generated_utc": _now_iso(),
        "verdict": "ok" if not reasons else "rejected",
        "reasons": reasons,
        "edge": {
            "from_world_frame_id": source_world,
            "to_world_frame_id": args.target_world,
            "T_to_from": inv(T_world_base).tolist(),
            "covariance_6x6": None,
            "method": (
                "fr3_single_tag_hand_eye: FR3 T_base_tcp + tag36h11 id6 on the tool, seen by the "
                f"world's cameras held at their {source_world} poses; tag corners reprojected "
                "(fisheye), T_world_base / T_tcp_tag / tag scale free"
            ),
            "source": {
                "p0_run": _repo_relative(run_dir),
                "captures_sha256": _sha256(captures_path),
                "intrinsics_run": intrinsics_run,
                "world_reference_sha256": _sha256(reference_path),
            },
            "quality": {
                "captures": len(captures),
                "fit": fit_all,
                "tag_scale": float(p_all[12]),
                "in_sample_tag_check": in_sample,
                "folds": fold_reports,
                "out_of_sample_tag_centre_p95_mm": max(oos_p95) if oos_p95 else None,
                "vs_p0_camera_alignment": {
                    "translation_mm_at_base_origin": float(np.linalg.norm(delta0[:3, 3]) * 1e3),
                    "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta0[:3, :3]).as_rotvec()))),
                },
            },
            "valid_while": (
                f"the rig cameras stay at their {source_world} poses and the FR3 is not "
                "re-mounted; either change needs a new edge"
            ),
        },
        "target_node": {
            "world_frame_id": args.target_world,
            "calibration_id": "",
            "parent_world_frame_id": None,
            "reason": "robot_base",
            "definition": "FR3 base frame (libfranka O frame) of the arm at its current mount",
        },
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"world {source_world} -> {args.target_world}: {len(captures)} captures, {len(observations)} observations")
    print(
        f"fit: kept {fit_all['observations_kept']}/{fit_all['observations_in']}, "
        f"rmse {fit_all['rmse_px']:.2f} px (p95 {fit_all['p95_px']:.2f}), tag scale {p_all[12]:.4f}"
    )
    if in_sample.get("captures"):
        print(f"in-sample tag centre: p50 {in_sample['tag_centre_p50_mm']:.2f} / p95 {in_sample['tag_centre_p95_mm']:.2f} mm")
    for name, fold in fold_reports.items():
        check = fold["held_out_tag_check"]
        shift = fold["T_world_base_vs_all"]
        if check.get("captures"):
            print(
                f"held out {name[-40:]}: {check['captures']} captures, tag centre p50 "
                f"{check['tag_centre_p50_mm']:.2f} / p95 {check['tag_centre_p95_mm']:.2f} mm; "
                f"edge moves {shift['translation_mm_at_base_origin']:.2f} mm / {shift['rotation_deg']:.3f} deg"
            )
    print(f"verdict: {report['verdict']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(f"written: {out}")
    return 0 if not reasons else 2


# ------------------------------------------------------------------- apply ---


def apply_report(report: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    """Return ``graph`` with the report's edge and target node added (stdlib)."""
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"not a {REPORT_SCHEMA} report")
    if report.get("verdict") != "ok":
        raise ValueError(f"report verdict is {report.get('verdict')!r}: {'; '.join(report.get('reasons') or [])}")
    edge = dict(report["edge"])
    source = str(edge["from_world_frame_id"])
    target = str(edge["to_world_frame_id"])
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    known = {str(node.get("world_frame_id")) for node in nodes}
    if source not in known:
        raise ValueError(f"{source} is not a node of the world graph")
    if wp.world_transform({"edges": edges}, source, target) is not None:
        raise ValueError(f"{source} and {target} are already connected; remove the old edge first")
    if target not in known:
        node = dict(report["target_node"])
        node.setdefault("created_utc", _now_iso())
        nodes.append(node)
    edge["created_utc"] = _now_iso()
    edges.append(edge)
    return {**graph, "version": graph.get("version", 1), "nodes": nodes, "edges": edges}


def _apply(args: argparse.Namespace) -> int:
    world_dir = (args.world_dir or _REPO_ROOT / wp.WORLD_SUBDIR).resolve()
    graph_path = world_dir / wp.WORLD_GRAPH_FILE
    report = json.loads(args.report.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else {"version": 1}
    try:
        updated = apply_report(report, graph)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    graph_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    edge = report["edge"]
    print(f"edge {edge['from_world_frame_id']} -> {edge['to_world_frame_id']} written to {graph_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world-dir", type=Path, default=None, help="default: tools/thor/gmsl2/world")
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", help="measure the edge from a P0 single-tag run (needs scipy/cv2)")
    solve.add_argument("--run", required=True, type=Path, help="P0 run dir holding captures.json and camera_calibration/")
    solve.add_argument(
        "--intrinsics",
        required=True,
        type=Path,
        help="summary.json of the intrinsics run the world's cameras were solved with",
    )
    solve.add_argument("--target-world", default=DEFAULT_TARGET_WORLD)
    solve.add_argument("--out", required=True, type=Path, help="report json to write")
    apply = sub.add_parser("apply", help="append a solved edge to world_graph.json")
    apply.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    return _solve(args) if args.command == "solve" else _apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
