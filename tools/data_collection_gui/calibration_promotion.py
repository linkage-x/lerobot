"""Comparing a solved calibration against the live one, and promoting it.

Three things live here, and they exist because of one incident: on 2026-08-20 a
solve produced extrinsics whose own summary recorded "cam_09 has moved", the
panel showed that run as live, and production went on loading the previous run
for seven days -- the reprojection gate then discarding cam_09 from 1675 of 1680
frames, i.e. throwing away a whole camera. Nothing said so, because the only
place the new run name was ever written was the gateway process's memory.

The obvious fix -- have the solve write the pointers itself -- is the wrong one,
and that is measured rather than argued. Any automatic rule has to rank the two
candidates by some number, and the numbers available are known to rank them
backwards: the 0804 run self-scored 0.244 px against 0820's 0.273 px, so both
"newest wins" and "best reprojection wins" would have kept the run that was
missing a moved camera. A solve auto-repointing production would also turn every
experiment into a deployment, which is the default the "solve without exporting"
mode exists to remove.

So the split is: the *comparison* is automatic and mandatory, the *decision* is
human, and the *execution* is one call rather than a hand edit of a YAML file.

Deliberately pure Python. The gateway runs on whatever interpreter is on the rig
-- numpy lives in the solve venv, not there -- so every quantity below is chosen
to be computable without a linear-algebra library. That constraint turns out to
improve the report: the gauge-free quantities (how far apart two cameras are,
how they are rotated relative to each other) need no common-frame alignment at
all, so there is no gauge choice to argue about, and a camera that moved on its
own shows up without having to first decide which frame stayed put.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Matrix = tuple[tuple[float, ...], ...]

# The two keys in the tracking config that decide which calibration production
# loads. Kept as an ordered mapping so callers name a kind ("extrinsics") rather
# than a YAML key, and the writer stays the only place that knows the spelling.
POINTER_KEYS: dict[str, str] = {
    "intrinsics": "intrinsics_run_name",
    "extrinsics": "fixed_camera_run_name",
}

# A promotion is refused outright when the candidate would move the world frame,
# unless the operator says so explicitly. Everything downstream -- every label
# already recorded -- is expressed in that frame, and a silent change makes old
# and new data incomparable without either of them looking wrong.
_CONTINUOUS = "CONTINUOUS"


@dataclass
class RunPoses:
    """What a calibration run says about where the cameras are.

    ``error`` is non-empty instead of raising: a run that cannot be read is a
    normal state to report on screen (it may be half-written, or on a machine
    that has not synced yet), not an exception that should take out the panel.
    """

    name: str = ""
    cameras: dict[str, Matrix] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    rmse_px: float | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.cameras)


def _as_matrix(value: Any) -> Matrix | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        try:
            rows.append(tuple(float(v) for v in row))
        except (TypeError, ValueError):
            return None
    return tuple(rows)


def load_run(run_dir: Path, name: str = "") -> RunPoses:
    """Read one extrinsics run's per-camera poses and world declaration."""
    label = name or run_dir.name
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return RunPoses(name=label, error=f"读不到 {summary_path.name}：{run_dir}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunPoses(name=label, error=f"{summary_path.name} 解析失败：{exc}")
    if not isinstance(summary, dict):
        return RunPoses(name=label, error=f"{summary_path.name} 不是一个 JSON 对象")

    joint = summary.get("joint_solution")
    entries = joint.get("cameras") if isinstance(joint, dict) else None
    cameras: dict[str, Matrix] = {}
    if isinstance(entries, dict):
        for cam, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            pose = entry.get("base_to_camera")
            matrix = _as_matrix(pose.get("matrix_4x4")) if isinstance(pose, dict) else None
            if matrix is not None:
                cameras[str(cam)] = matrix
    if not cameras:
        return RunPoses(name=label, error=f"{summary_path.name} 里没有 joint_solution.cameras 位姿")

    world = summary.get("world") if isinstance(summary.get("world"), dict) else {}
    rmse = summary.get("bundle_rmse_px")
    return RunPoses(
        name=label,
        cameras=cameras,
        world=world,
        rmse_px=float(rmse) if isinstance(rmse, (int, float)) else None,
    )


def _translation(matrix: Matrix) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])


def _distance_m(a: Matrix, b: Matrix) -> float:
    pa, pb = _translation(a), _translation(b)
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def _relative_rotation(a: Matrix, b: Matrix) -> Matrix:
    """R_a^T R_b -- how b is oriented as seen from a. Independent of any frame."""
    return tuple(
        tuple(sum(a[k][i] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rotation_angle_deg(a: Matrix, b: Matrix) -> float:
    """Angle between two 3x3 rotations, via trace(a^T b)."""
    trace = sum(sum(a[k][i] * b[k][i] for k in range(3)) for i in range(3))
    # Clamp before acos: accumulated float error puts the argument a few ulps
    # outside [-1, 1] for identical rotations, which is exactly the case that
    # matters here (a camera that did not move must report 0, not a domain error).
    cos = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos))


def compare_runs(live: RunPoses, candidate: RunPoses) -> dict[str, Any]:
    """How the candidate differs from what production loads, gauge-free.

    Every number reported here is invariant under a rigid motion of the whole
    rig, so none of them depends on picking a reference camera or on aligning
    the two solutions first. Two families:

    * ``baselineShiftMm`` -- how the straight-line distance between two cameras
      changed. Cameras that did not move keep their distances.
    * ``relativeRotationDeg`` -- how the orientation of one camera as seen from
      another changed.

    The per-camera row takes the **median over that camera's partners**, and the
    median rather than the max is load-bearing: if one camera moves, every one of
    its own baselines changes while each other camera sees exactly one of its
    baselines change. The median therefore isolates the mover, where a max would
    smear it across the whole rig.

    There is deliberately no score, ranking, or verdict. Choosing between two
    calibrations by a single number is the thing that already picked wrong.
    """
    if not live.ok or not candidate.ok:
        return {
            "ok": False,
            "error": live.error or candidate.error or "两份标定里至少一份读不出位姿",
            "live": live.name,
            "candidate": candidate.name,
        }

    shared = sorted(set(live.cameras) & set(candidate.cameras))
    added = sorted(set(candidate.cameras) - set(live.cameras))
    removed = sorted(set(live.cameras) - set(candidate.cameras))

    pairs: list[dict[str, Any]] = []
    by_camera: dict[str, dict[str, list[float]]] = {
        cam: {"baseline": [], "rotation": []} for cam in shared
    }
    for i, cam_a in enumerate(shared):
        for cam_b in shared[i + 1 :]:
            d_live = _distance_m(live.cameras[cam_a], live.cameras[cam_b])
            d_cand = _distance_m(candidate.cameras[cam_a], candidate.cameras[cam_b])
            shift_mm = abs(d_cand - d_live) * 1000.0
            rot_deg = _rotation_angle_deg(
                _relative_rotation(live.cameras[cam_a], live.cameras[cam_b]),
                _relative_rotation(candidate.cameras[cam_a], candidate.cameras[cam_b]),
            )
            pairs.append(
                {
                    "a": cam_a,
                    "b": cam_b,
                    "liveMm": round(d_live * 1000.0, 2),
                    "candidateMm": round(d_cand * 1000.0, 2),
                    "shiftMm": round(shift_mm, 3),
                    "rotationDeg": round(rot_deg, 4),
                }
            )
            for cam in (cam_a, cam_b):
                by_camera[cam]["baseline"].append(shift_mm)
                by_camera[cam]["rotation"].append(rot_deg)

    cameras = [
        {
            "camera": cam,
            "medianBaselineShiftMm": round(statistics.median(values["baseline"]), 3),
            "maxBaselineShiftMm": round(max(values["baseline"]), 3),
            "medianRotationDeg": round(statistics.median(values["rotation"]), 4),
            "maxRotationDeg": round(max(values["rotation"]), 4),
        }
        for cam, values in by_camera.items()
        if values["baseline"]
    ]
    cameras.sort(key=lambda row: -row["medianBaselineShiftMm"])
    worst_pair = max(pairs, key=lambda row: row["shiftMm"]) if pairs else None

    return {
        "ok": True,
        "live": live.name,
        "candidate": candidate.name,
        "cameras": cameras,
        "addedCameras": added,
        "removedCameras": removed,
        "pairCount": len(pairs),
        "medianBaselineShiftMm": round(statistics.median([p["shiftMm"] for p in pairs]), 3) if pairs else 0.0,
        "medianRotationDeg": round(statistics.median([p["rotationDeg"] for p in pairs]), 4) if pairs else 0.0,
        "worstPair": worst_pair,
        "liveWorld": _world_view(live.world),
        "candidateWorld": _world_view(candidate.world),
        # Reported because the operator will look for it, and labelled so that
        # it is not mistaken for a criterion: this is the number that ranked the
        # two runs backwards in August.
        "liveRmsePx": live.rmse_px,
        "candidateRmsePx": candidate.rmse_px,
        "rmseIsNotACriterion": True,
    }


def normalize_model(name: str) -> str:
    """One spelling for a projection model, because the repo uses two.

    The tracking config says ``camera_model: fisheye``; the per-camera producer
    files written by the exporter say ``"model": "opencv_fisheye"``. Comparing
    them raw is worse than not comparing at all -- it makes every correct lens
    run look like a mismatch, so the check would either be disabled or, worse,
    silently drop every real candidate from the staleness gate.
    """
    text = str(name or "").strip().lower()
    return text[len("opencv_") :] if text.startswith("opencv_") else text


@dataclass
class IntrinsicsRun:
    """Which cameras a lens run ships, and under which projection model."""

    name: str = ""
    cameras: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.cameras)

    @property
    def model(self) -> str:
        """The single model this run uses, or "" when it is not single-valued."""
        distinct = set(self.models.values())
        return distinct.pop() if len(distinct) == 1 else ""

    @property
    def normalized_model(self) -> str:
        return normalize_model(self.model)


def load_intrinsics_run(run_dir: Path, name: str = "") -> IntrinsicsRun:
    """Read a lens run's camera list and per-camera projection model.

    Reads the per-camera producer files rather than the run summary, because
    those are the files the tracker actually loads -- a summary that disagrees
    with them would be a report about a calibration nobody uses.
    """
    label = name or run_dir.name
    if not run_dir.is_dir():
        return IntrinsicsRun(name=label, error=f"内参 run 目录不存在：{run_dir}")
    models: dict[str, str] = {}
    for producer in sorted(run_dir.glob("converted/*/intrinsics_producer.json")):
        try:
            payload = json.loads(producer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        camera = str(payload.get("camera_name") or "").strip()
        if not camera:
            # Fall back to the directory name, which is "<camera>_<serial>".
            parts = producer.parent.name.split("_")
            camera = "_".join(parts[:2]) if len(parts) >= 2 else ""
        if camera:
            models[camera] = str(payload.get("model") or "").strip()
    if not models:
        return IntrinsicsRun(name=label, error=f"内参 run 里没有 converted/*/intrinsics_producer.json：{run_dir}")
    return IntrinsicsRun(name=label, cameras=sorted(models), models=models)


def compare_intrinsics_runs(
    live: IntrinsicsRun, candidate: IntrinsicsRun, *, tracker_model: str = ""
) -> dict[str, Any]:
    """What promoting a lens run would change, and what would break.

    The blocking cases here are not hypothetical. ``load_intrinsics_map_from_run``
    refuses a run whose cameras do not all share one projection model, and the
    tracker separately checks the loaded model against ``cube_tracker.camera_model``
    -- so a run that is fine on its own can still be unloadable in this config.
    Both are cheap to check before the write and expensive to discover after it.
    """
    if not candidate.ok:
        return {"ok": False, "error": candidate.error, "live": live.name, "candidate": candidate.name}
    mixed = sorted(set(candidate.models.values()))
    return {
        "ok": True,
        "live": live.name,
        "candidate": candidate.name,
        "cameras": candidate.cameras,
        "model": candidate.model,
        "mixedModels": mixed if len(mixed) > 1 else [],
        "trackerModel": tracker_model,
        "addedCameras": sorted(set(candidate.cameras) - set(live.cameras)),
        "removedCameras": sorted(set(live.cameras) - set(candidate.cameras)) if live.ok else [],
        "liveError": live.error,
    }


def intrinsics_blockers(comparison: dict[str, Any]) -> list[dict[str, str]]:
    """Reasons promoting a lens run should stop and ask."""
    if not comparison.get("ok"):
        return [{"kind": "unreadable", "message": str(comparison.get("error") or "内参读不出来")}]
    blockers: list[dict[str, str]] = []
    if comparison.get("mixedModels"):
        blockers.append(
            {
                "kind": "mixed_models",
                "message": (
                    "这份内参里混着多种投影模型（"
                    + "、".join(comparison["mixedModels"])
                    + "），加载时会被 load_intrinsics_map_from_run 直接拒绝。"
                ),
            }
        )
    tracker_model = comparison.get("trackerModel") or ""
    model = comparison.get("model") or ""
    if tracker_model and model and normalize_model(tracker_model) != normalize_model(model):
        blockers.append(
            {
                "kind": "model_mismatch",
                "message": (
                    f"这份内参是 {model}，而配置里 cube_tracker.camera_model 是 {tracker_model}。"
                    "两者必须一致，否则追踪启动时会被 check_camera_model 拦住。"
                ),
            }
        )
    if comparison.get("removedCameras"):
        blockers.append(
            {
                "kind": "cameras_removed",
                "message": (
                    "这份内参比在产的少了：" + "、".join(comparison["removedCameras"])
                    + "。提升后这些相机在生产里将没有内参。"
                ),
            }
        )
    return blockers


def _world_view(world: dict[str, Any]) -> dict[str, Any]:
    stable = world.get("stable_cameras")
    return {
        "worldFrameId": str(world.get("world_frame_id", "") or ""),
        "referenceWorldFrameId": str(world.get("reference_world_frame_id", "") or ""),
        "continuityState": str(world.get("world_continuity_state", "") or ""),
        "reason": str(world.get("reason", "") or ""),
        "stableCameras": sorted(str(c) for c in stable) if isinstance(stable, list) else [],
    }


def promotion_blockers(comparison: dict[str, Any]) -> list[dict[str, str]]:
    """Reasons a promotion should stop and ask, rather than proceed.

    These are refusals an operator can override, not errors: each one names a
    real risk that a human at the rig may already know the answer to (they may
    have deliberately re-registered the world, or deliberately dropped a camera).
    What must not happen is the promotion going through without the question
    being put.
    """
    if not comparison.get("ok"):
        return [{"kind": "unreadable", "message": str(comparison.get("error") or "标定读不出来")}]

    blockers: list[dict[str, str]] = []
    world = comparison.get("candidateWorld") or {}
    live_world = comparison.get("liveWorld") or {}
    state = world.get("continuityState") or ""
    if state and state != _CONTINUOUS:
        blockers.append(
            {
                "kind": "world_continuity",
                "message": (
                    f"这份标定的世界连续性是 {state}（不是 {_CONTINUOUS}）。"
                    "提升后新旧数据将不在同一个世界系里，已录的标签无法与新标签直接比较。"
                ),
            }
        )
    candidate_id = world.get("worldFrameId") or ""
    live_id = live_world.get("worldFrameId") or ""
    if candidate_id and live_id and candidate_id != live_id:
        blockers.append(
            {
                "kind": "world_frame_changed",
                "message": (
                    f"世界系从 {live_id} 变成了 {candidate_id}。"
                    "这会让此前录制的数据与今后录的处在不同世界，导出时会被拒绝合池。"
                ),
            }
        )
    if comparison.get("removedCameras"):
        blockers.append(
            {
                "kind": "cameras_removed",
                "message": (
                    "这份标定比在产的少了：" + "、".join(comparison["removedCameras"])
                    + "。提升后这些相机在生产里将没有外参。"
                ),
            }
        )
    return blockers


# ---------------------------------------------------------------------------
# Writing the pointer


class PointerWriteError(RuntimeError):
    """The tracking config could not be rewritten safely."""


def rewrite_pointers(text: str, updates: dict[str, str]) -> tuple[str, list[dict[str, str]]]:
    """Replace pointer values in the config text, keeping everything else byte-identical.

    A YAML round-trip is not an option here. The lines above these two keys carry
    the reasoning for the current choice -- including an explicit "Do NOT repoint
    world_reference.json / world_graph.json at this run" -- and ``yaml.safe_dump``
    would delete every comment in the file. So this edits the two lines in place
    and leaves the rest of the bytes alone.

    Refuses when a key is absent or appears more than once, rather than guessing
    which occurrence production reads. Guessing wrong here writes a pointer that
    looks promoted and is not, which is the failure this whole module exists for.
    """
    lines = text.splitlines(keepends=True)
    changes: list[dict[str, str]] = []
    for kind, value in updates.items():
        key = POINTER_KEYS.get(kind)
        if key is None:
            raise PointerWriteError(f"未知的标定类型：{kind}")
        pattern = re.compile(rf"^(\s*){re.escape(key)}(\s*:\s*)(.*?)(\s*(?:#.*)?)$")
        hits = [(i, m) for i, line in enumerate(lines) if (m := pattern.match(line.rstrip("\n")))]
        if not hits:
            raise PointerWriteError(f"配置里找不到 {key}")
        if len(hits) > 1:
            raise PointerWriteError(
                f"配置里有 {len(hits)} 处 {key}，不确定生产读的是哪一处，拒绝改写"
            )
        index, match = hits[0]
        previous = match.group(3).strip().strip("'\"")
        newline = "\n" if lines[index].endswith("\n") else ""
        lines[index] = f"{match.group(1)}{key}{match.group(2)}{value}{match.group(4)}{newline}"
        changes.append({"kind": kind, "key": key, "from": previous, "to": value})
    return "".join(lines), changes


def write_config_atomically(path: Path, text: str) -> None:
    """Replace the config in one step, so a crash cannot leave it half-written.

    A truncated tracking config is worse than a stale one: the stale config loads
    the previous calibration, the truncated one stops every consumer.
    """
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temp_name)
        raise


def promotion_record(
    *,
    changes: Sequence[dict[str, str]],
    comparison: dict[str, Any],
    acknowledged: Sequence[str] = (),
    note: str = "",
    actor: str = "",
) -> dict[str, Any]:
    """The line appended to the promotion log.

    Kept deliberately small and flat: this is read months later to answer "which
    calibration produced this trajectory, and who decided that", so it stores the
    decision and the evidence it was made on, not the whole comparison.
    """
    return {
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor": actor or "gui",
        "changes": [dict(change) for change in changes],
        "note": note,
        "acknowledgedBlockers": sorted(str(k) for k in acknowledged),
        "evidence": {
            "medianBaselineShiftMm": comparison.get("medianBaselineShiftMm"),
            "medianRotationDeg": comparison.get("medianRotationDeg"),
            "worstPair": comparison.get("worstPair"),
            "candidateWorld": comparison.get("candidateWorld"),
            "addedCameras": comparison.get("addedCameras"),
            "removedCameras": comparison.get("removedCameras"),
        },
    }


def append_promotion_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Staleness: promotable runs that production is not loading

# Written into a run directory by a solve that deliberately did not export.
# Those runs are real bundle adjustments and their directories look exactly like
# a promotable one, so without this marker the staleness gate would nag about
# every experiment -- and a gate that cries wolf is a gate that gets ignored.
EXPERIMENT_MARKER = "experiment.json"


def promotable_runs(
    calibration_root: Path,
    *,
    suffix: str,
    live_run: str,
    require_model: str = "",
) -> list[dict[str, Any]]:
    """Run directories newer than the one production loads.

    Durable across gateway restarts on purpose. The in-memory "last solve" is
    what the August incident had, and it was erased by the first restart -- after
    which the panel agreed with the config and nothing was left to notice the
    seven days of the wrong calibration. The filesystem remembers.

    ``require_model`` drops lens runs the tracker could not load anyway. Without
    it the rig's own history defeats the gate: ``thor_gmsl2_selfcal_0804_fisheye``
    and ``..._rational`` were exported from one report seconds apart, so the
    rational twin reads as "newer" forever and every trajectory run would be
    warned about a calibration that ``check_camera_model`` would reject. A gate
    that cries wolf is a gate that gets ignored, which is the failure this whole
    module is trying not to repeat.
    """
    if not calibration_root.is_dir():
        return []
    live_dir = calibration_root / live_run if live_run else None
    try:
        live_mtime = live_dir.stat().st_mtime if live_dir and live_dir.is_dir() else None
    except OSError:
        live_mtime = None

    found: list[dict[str, Any]] = []
    for entry in calibration_root.iterdir():
        if not entry.is_dir() or not entry.name.endswith(suffix) or entry.name == live_run:
            continue
        if (entry / EXPERIMENT_MARKER).is_file():
            continue
        if not (entry / "summary.json").is_file():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if live_mtime is not None and mtime <= live_mtime:
            continue
        if require_model and load_intrinsics_run(entry).normalized_model != normalize_model(require_model):
            continue
        found.append(
            {
                "run": entry.name,
                "updatedAt": datetime.fromtimestamp(mtime, UTC)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M"),
                "mtime": mtime,
            }
        )
    found.sort(key=lambda row: -row["mtime"])
    for row in found:
        row.pop("mtime", None)
    return found


def write_experiment_marker(run_dir: Path, *, reason: str = "") -> None:
    """Mark a run as not-for-promotion, at the moment it is produced."""
    if not run_dir.is_dir():
        return
    payload = {
        "experiment": True,
        "reason": reason or "solved without exporting production calibration",
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (run_dir / EXPERIMENT_MARKER).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def stale_pointer_refusal(stale: Iterable[dict[str, Any]], *, kind_label: str, live_run: str) -> str:
    """The message the trajectory gate shows, or "" when there is nothing newer.

    Phrased as a question about which calibration to use rather than as an error,
    because using the older one is sometimes right -- reproducing an earlier
    result, for instance. What is never right is not being told.
    """
    rows = list(stale)
    if not rows:
        return ""
    names = "、".join(row["run"] for row in rows[:3])
    more = f" 等 {len(rows)} 份" if len(rows) > 3 else ""
    return (
        f"生产加载的{kind_label}是 {live_run}，而磁盘上有更新的{names}{more}。"
        f"这条轨迹会用旧标定生成。先提升，或确认就是要用旧的。"
    )
