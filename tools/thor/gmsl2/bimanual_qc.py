"""Can a tracked session's two hands be replayed together? Bimanual QC of the left/right sidecars.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, A1 (main roadmap Phase 8, the bimanual part). The
per-hand QC asks whether each hand's track is usable; a two-arm replay also needs the two tracks to agree
*with each other*, frame by frame, because what it reproduces first is the relative pose
``T_LR = inv(T_L) T_R`` -- a hand-over, a box held between two hands. That is measured in no single hand's
numbers:

* **when both hands are seen** -- frames where both solved, and where both had at least MIN_CAMERAS cameras
  (the cube's own threshold for a trustworthy solve), stratified by how many cameras each hand had;
* **steps in T_LR** -- a frame where the relative position jumps (second difference above STEP_MM). The
  hands' own motion is smooth at 60 fps; a step is a camera-set change in one hand's solve (08-28: 19.6 mm),
  and the report says how many steps sat on such a change;
* **T_LR jitter** -- the frame-to-frame noise of the relative position where neither hand's camera set
  changed, from second differences (white noise sigma = std / sqrt(6)).

  Both are read on ``p_R - p_L`` in the dataset's frame, not on the translation of ``inv(T_L) T_R`` (the
  right TCP in the *left TCP's* frame). That one turns with the left hand, so the left hand's ordinary
  rotation times the ~0.8 m between the hands reads as steps: on stand-in pairs of recorded episodes
  (``pseudo_bimanual``, 10-26) its second difference had p50 2 mm and p95 8-11 mm, against p50 0.5 and
  p95 1.3-1.7 mm for the frame-fixed difference, and every pair failed. A label jump in either hand shows
  in ``p_R - p_L`` in full, and a rotation glitch of a cube moves its TCP point, so it shows there too;
* **one-hand losses** -- runs where one hand is tracked and the other is not (cross occlusion, a hand out of
  view), longest and how many exceed MAX_LOSS_S: a replay cannot interpolate the relative pose across them;
* **usable segments** -- contiguous runs where both hands have MIN_CAMERAS and T_LR has no step, at least
  MIN_SEGMENT_S long: what the A2 benchmark selection picks from.

``bimanual_pass`` is the conjunction of the gates below, each recorded with its value and threshold. It is a
statement about consistency, not accuracy: the left hand's label chain (marker -> TCP, d_box) has never been
measured against the laser, so a session can pass here and still carry the left hand mm off (A3).

Stdlib only, like ``label_provenance``: it runs in the gateway's system python.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SIDECAR = "april_cube_tracking_in_robot_base"
HANDS = ("left", "right")
SCHEMA = "bimanual_qc/v1"

MIN_CAMERAS = 4
#: A T_LR step: second difference of the relative position above this. Smooth motion at 60 fps with 5 m/s^2
#: relative acceleration is 1.4 mm; the camera-set steps the per-hand QC sees are 3-20 mm.
STEP_MM = 5.0
MAX_LOSS_S = 0.5
MIN_SEGMENT_S = 2.0
#: Gates of bimanual_pass.
MIN_BOTH_SEEN_FRACTION = 0.8
MAX_STEPS_PER_MIN = 2.0
DEFAULT_FPS = 60.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: Any, default: int = 0) -> int:
    number = _float(value)
    return int(number) if math.isfinite(number) else default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def read_hand(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    """``{episode: {frame: row}}`` of one hand's sidecar; ``ok`` is solve_ok with a finite pose.

    A v1 sidecar (no ``schema_version``) carries no solve quality and is refused rather than read as if every
    frame were seen by every camera.
    """
    episodes: dict[int, dict[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "schema_version" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: no schema_version column; a v1 sidecar has no per-frame solve quality")
        for row in reader:
            pose = [_float(row.get(f"state_{k}")) for k in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")]
            ok = _truthy(row.get("solve_ok")) and all(math.isfinite(v) for v in pose)
            episodes.setdefault(_int(row.get("episode_index")), {})[_int(row.get("frame_index"), -1)] = {
                "t": _float(row.get("timestamp_s")),
                "p": pose[:3],
                "q": pose[3:],
                "ok": ok,
                "cams": _int(row.get("num_cameras")) if ok else 0,
                "camera_set": str(row.get("camera_set") or ""),
                "set_changed": _truthy(row.get("camera_set_changed")),
            }
    return episodes


def _rotate_inverse(q: list[float], v: list[float]) -> list[float]:
    """``R(q)^T v`` for a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = -x / n, -y / n, -z / n, w / n
    # v' = v + 2w (u x v) + 2 u x (u x v), u = (x, y, z)
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    return [
        v[0] + 2 * (w * cx + y * cz - z * cy),
        v[1] + 2 * (w * cy + z * cx - x * cz),
        v[2] + 2 * (w * cz + x * cy - y * cx),
    ]


def relative_position(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[float]:
    """The right TCP in the left TCP's frame: the translation of ``inv(T_L) T_R``, metres."""
    return _rotate_inverse(list(left["q"]), [right["p"][k] - left["p"][k] for k in range(3)])


def _bucket(cams: int) -> str:
    return "0" if cams <= 0 else "1" if cams == 1 else "2-3" if cams <= 3 else f">={MIN_CAMERAS}"


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Half-open index runs where ``flags`` is True."""
    runs, start = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 3)


def _fps(left: Mapping[int, Mapping[str, Any]], frames: list[int]) -> float:
    times = [left[f]["t"] for f in frames if f in left and math.isfinite(left[f]["t"])]
    steps = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
    return 1.0 / statistics.median(steps) if steps else DEFAULT_FPS


def episode_qc(left: Mapping[int, Mapping[str, Any]], right: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    frames = sorted(set(left) | set(right))
    if not frames:
        return {"frames": 0, "bimanual_pass": False, "reasons": ["no frames"]}
    fps = _fps(left, frames) if left else DEFAULT_FPS
    L = [left.get(f) for f in frames]
    R = [right.get(f) for f in frames]
    l_ok = [bool(a and a["ok"]) for a in L]
    r_ok = [bool(b and b["ok"]) for b in R]
    both = [a and b for a, b in zip(l_ok, r_ok, strict=True)]
    seen = [
        both[i] and L[i]["cams"] >= MIN_CAMERAS and R[i]["cams"] >= MIN_CAMERAS for i in range(len(frames))
    ]
    strata: dict[str, int] = {}
    for i in range(len(frames)):
        if both[i]:
            key = f"L{_bucket(L[i]['cams'])} R{_bucket(R[i]['cams'])}"
            strata[key] = strata.get(key, 0) + 1

    # frame-fixed, not in the left TCP's frame: see the module docstring (steps in T_LR)
    rel = [[R[i]["p"][k] - L[i]["p"][k] for k in range(3)] if both[i] else None for i in range(len(frames))]
    changed = [bool((L[i] and L[i]["set_changed"]) or (R[i] and R[i]["set_changed"])) for i in range(len(frames))]
    step = [False] * len(frames)
    second: list[float] = []
    steps_on_change = 0
    for i in range(1, len(frames) - 1):
        a, b, c = rel[i - 1], rel[i], rel[i + 1]
        if a is None or b is None or c is None or frames[i + 1] - frames[i - 1] != 2:
            continue
        d2 = math.sqrt(sum((a[k] - 2 * b[k] + c[k]) ** 2 for k in range(3))) * 1e3
        if not (changed[i] or changed[i + 1]) and seen[i - 1] and seen[i] and seen[i + 1]:
            second.append(d2)
        if d2 > STEP_MM:
            step[i] = True
    # a step of size s shows as two neighbouring second differences; count it once
    step_runs = _runs(step)
    for start, end in step_runs:
        if any(changed[j] for j in range(max(0, start - 1), min(len(frames), end + 2))):
            steps_on_change += 1

    loss = [a != b for a, b in zip(l_ok, r_ok, strict=True)]
    loss_runs = [(s, e) for s, e in _runs(loss)]
    loss_s = [(e - s) / fps for s, e in loss_runs]
    usable = [seen[i] and not step[i] for i in range(len(frames))]
    segments = [
        {"start_frame": frames[s], "end_frame": frames[e - 1], "duration_s": round((e - s) / fps, 2)}
        for s, e in _runs(usable)
        if (e - s) / fps >= MIN_SEGMENT_S
    ]

    n = len(frames)
    duration_min = n / fps / 60.0
    seen_fraction = sum(seen) / n
    steps_per_min = len(step_runs) / duration_min if duration_min > 0 else 0.0
    longest_loss = max(loss_s, default=0.0)
    gates = {
        "both_seen_fraction": {"value": round(seen_fraction, 4), "min": MIN_BOTH_SEEN_FRACTION},
        "tlr_steps_per_min": {"value": round(steps_per_min, 2), "max": MAX_STEPS_PER_MIN},
        "longest_one_hand_loss_s": {"value": round(longest_loss, 3), "max": MAX_LOSS_S},
    }
    reasons = []
    if seen_fraction < MIN_BOTH_SEEN_FRACTION:
        reasons.append(f"both hands seen by >= {MIN_CAMERAS} cameras in {seen_fraction:.0%} of frames")
    if steps_per_min > MAX_STEPS_PER_MIN:
        reasons.append(f"{len(step_runs)} T_LR steps > {STEP_MM:g} mm ({steps_per_min:.1f}/min)")
    if longest_loss > MAX_LOSS_S:
        reasons.append(f"one hand lost for {longest_loss:.2f} s while the other was tracked")
    jitter = statistics.median(second) / 0.6745 / math.sqrt(6) if second else None
    return {
        "frames": n,
        "fps": round(fps, 3),
        "both_solved": sum(both),
        "both_seen": sum(seen),
        "left_only": sum(a and not b for a, b in zip(l_ok, r_ok, strict=True)),
        "right_only": sum(b and not a for a, b in zip(l_ok, r_ok, strict=True)),
        "neither": sum(not a and not b for a, b in zip(l_ok, r_ok, strict=True)),
        "camera_strata": dict(sorted(strata.items())),
        "tlr_jitter_mm": None if jitter is None else round(jitter, 3),
        "tlr_steps": len(step_runs),
        "tlr_steps_on_camera_set_change": steps_on_change,
        "tlr_step_frames": [frames[s] for s, _ in step_runs],
        "one_hand_losses": len(loss_runs),
        "one_hand_losses_over_max": sum(s > MAX_LOSS_S for s in loss_s),
        "one_hand_loss_p95_s": _pct(loss_s, 95),
        "usable_segments": segments,
        "usable_s": round(sum(seg["duration_s"] for seg in segments), 2),
        "gates": gates,
        "bimanual_pass": not reasons,
        "reasons": reasons,
    }


def session_qc(session_dir: Path, *, sidecar: str = SIDECAR, raw: bool = False) -> dict[str, Any]:
    """Every episode of a session that has both hands' sidecars."""
    prefix = "raw_state_action" if raw else "state_action"
    paths = {hand: session_dir / "derived" / sidecar / f"{prefix}.{hand}.csv" for hand in HANDS}
    missing = [hand for hand, path in paths.items() if not path.is_file()]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "session": session_dir.name,
        "sidecar": f"{sidecar}/{prefix}.<hand>.csv",
        "thresholds": {
            "min_cameras": MIN_CAMERAS,
            "step_mm": STEP_MM,
            "max_loss_s": MAX_LOSS_S,
            "min_segment_s": MIN_SEGMENT_S,
            "min_both_seen_fraction": MIN_BOTH_SEEN_FRACTION,
            "max_steps_per_min": MAX_STEPS_PER_MIN,
        },
        "note": (
            "consistency of the two hands' tracks, not their accuracy: the left hand's label chain is unvalidated "
            "(roadmap A3)"
        ),
    }
    if missing:
        report["error"] = f"no {prefix} sidecar for the {', '.join(missing)} hand"
        report["episodes"] = {}
        return report
    hands = {hand: read_hand(path) for hand, path in paths.items()}
    episodes = sorted(set(hands["left"]) | set(hands["right"]))
    report["episodes"] = {
        str(ep): episode_qc(hands["left"].get(ep, {}), hands["right"].get(ep, {})) for ep in episodes
    }
    passed = [ep for ep, row in report["episodes"].items() if row.get("bimanual_pass")]
    report["summary"] = {
        "episodes": len(episodes),
        "passed": len(passed),
        "usable_s": round(sum(row.get("usable_s", 0.0) for row in report["episodes"].values()), 2),
    }
    return report


def describe(report: Mapping[str, Any]) -> Iterable[str]:
    yield f"{report['session']}: {report.get('error') or ''}".rstrip(": ")
    for ep, row in (report.get("episodes") or {}).items():
        verdict = "PASS" if row.get("bimanual_pass") else "FAIL"
        yield (
            f"  ep {ep}: {verdict} {row['frames']} frames, both seen {row['both_seen']}"
            f" ({row['gates']['both_seen_fraction']['value']:.0%}), T_LR jitter {row['tlr_jitter_mm']} mm,"
            f" steps {row['tlr_steps']} ({row['tlr_steps_on_camera_set_change']} on a camera-set change),"
            f" one-hand losses {row['one_hand_losses']} (> {MAX_LOSS_S:g} s: {row['one_hand_losses_over_max']}),"
            f" usable {row['usable_s']} s"
            + (f" -- {'; '.join(row['reasons'])}" if row.get("reasons") else "")
        )
    summary = report.get("summary")
    if summary:
        yield f"  {summary['passed']}/{summary['episodes']} episodes pass, {summary['usable_s']} s usable"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", type=Path, help="recorded session dirs (with derived/<sidecar>/)")
    parser.add_argument("--sidecar", default=SIDECAR)
    parser.add_argument("--raw", action="store_true", help="read raw_state_action (before the offline smoother)")
    parser.add_argument("--out", type=Path, help="write all reports to this json")
    args = parser.parse_args(argv)
    reports = []
    for session in args.sessions:
        try:
            report = session_qc(session, sidecar=args.sidecar, raw=args.raw)
        except ValueError as exc:
            report = {"schema": SCHEMA, "session": session.name, "error": str(exc), "episodes": {}}
        reports.append(report)
        for line in describe(report):
            print(line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report: {args.out}")
    return 0 if all("error" not in r for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
