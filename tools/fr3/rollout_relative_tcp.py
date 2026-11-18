#!/usr/bin/env python
# ruff: noqa: N803, N806  -- T, R, A, B: the usual names of transforms, rotations and capsule ends
"""Rollout of a relative-TCP policy, in stages; this file delivers the first two and the execution plan check.

    offline   predict on a QC-gated export through the online code path, against what the hand did
    preview   predict on the recorder's live cameras and BOX with the arm still; record every input
    replay    feed a preview record back through the offline path: the predictions must be identical
    execute   drive the arm with the policy, by stages (rollout_execute): a person at the e-stop starts it,
              and nothing moves without the maintainer's unlock for that arm (real_execution_unlock)

What is the same online and offline (user guide #infer-inputs), and where it lives:

* images: the full-resolution frame of each camera, cropped and resized by relative_tcp.crop_resize.
  Offline that frame is decoded from the export's 1080p video; online it is the recorder's frame bus
  (``/dev/shm/lerobot_online_sync_preview``: synchronised NV12 clusters at full resolution; the policy
  never opens a camera). Offline also predicts from the training view's frames, which is what training
  saw, so the gap between "what training saw" and "what inference sees" is measured, not assumed.
* gripper width and external wrench: offline the export's state columns; online the recorder's BOX_LIVE
  snapshot of the arm's gripper BOX, its width through thor_lerobot_v3's own state layout and its
  external wrench from box_six_d.live_external (the arm BOX's K in force, bias = this Connect's last
  zero; an export takes the bias from each episode's still ends -- recorded in every preview record).
* TCP history: offline the tracked TCP; online the arm's FK at ARM_FK_FRAME. In preview the arm does not
  move, so the history is "not moving" whatever the FK (exact, not an approximation, while it stands
  still); FK of the arm's joints is still read once for the execution plan check.

The execution plan check (:meth:`PlanContext.check`) is the execution layer's first half, run on every preview
prediction and never followed by motion: targets = FK(q_obs) · ΔT in the arm's base; IK reachability
inside the joint walls (fr3_kinematics, as IK QC); the training workspace re-expressed in the base through
the registration edge of the day (refused when the world-graph guard blocks or the edge is retired); the
capsule model's clearance to the carts, the table and the cameras (dual_fr3_collision, as move-to-start).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.fr3 import relative_tcp as rt  # noqa: E402, I001
from tools.fr3 import relative_tcp_data as rd  # noqa: E402

RESULT_SCHEMA = "fr3_rollout_result/1"
RECORD_SCHEMA = "fr3_rollout_record/1"
PROGRESS_PREFIX = "FR3_ROLLOUT "
PREVIEW_FRAME_BUS = Path("/dev/shm/lerobot_online_sync_preview")
STAGES = (
    {"id": "offline", "label": "Offline prediction on recordings", "status": "available"},
    {"id": "preview", "label": "Online preview (live cameras + BOX, arm still)", "status": "available"},
    {"id": "plan", "label": "Execution plan check (IK, workspace, collision) on every prediction", "status": "available"},
    {"id": "execute", "label": "Execution on the arm (stages 1-3, attended, maintainer unlock)", "status": "available"},
)
ENV_MARGIN_M = 0.010  # dual_fr3_retarget.ENV_MARGIN_MM
WORKSPACE_MARGIN_M = 0.03


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def progress(**fields: Any) -> None:
    print(PROGRESS_PREFIX + json.dumps(fields, default=float), flush=True)


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def chunk_errors(pred: np.ndarray, gt: np.ndarray, pad: np.ndarray) -> dict[str, np.ndarray]:
    """Per step of a chunk: position (mm), rotation (deg) and width (mm) error; NaN where padded."""
    Tp, Tg = rt.decode(pred[:, : rt.POSE_DIM]), rt.decode(gt[:, : rt.POSE_DIM])
    pos = np.linalg.norm(Tp[:, :3, 3] - Tg[:, :3, 3], axis=1) * 1e3
    rot = _rot_angle_deg(Tp[:, :3, :3], Tg[:, :3, :3])
    width = np.abs(pred[:, rt.POSE_DIM] - gt[:, rt.POSE_DIM]) * 1e3
    for arr in (pos, rot, width):
        arr[pad] = np.nan
    return {"pos_mm": pos, "rot_deg": rot, "width_mm": width}


def _summ(values: list[np.ndarray]) -> dict[str, Any]:
    stack = np.stack(values)  # (frames, horizon)
    flat = stack[np.isfinite(stack)]
    if flat.size == 0:
        return {}
    return {
        "mean": float(flat.mean()),
        "p95": float(np.quantile(flat, 0.95)),
        "by_step": [None if not np.isfinite(v) else round(float(v), 3) for v in np.nanmean(stack, axis=0)],
    }


# ------------------------------------------------------------------ offline ---


def offline(
    checkpoint: Path,
    export_root: Path,
    *,
    episodes: list[int] | None = None,
    stride: int = 15,
    views_root: Path = Path("outputs/training/views"),
    device: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Predict along an export's episodes; compare with the hand's motion, a "not moving" baseline, and the
    same prediction from the training view's frames."""
    from tools.fr3 import relative_tcp_policy as rtp

    loaded = rtp.load_checkpoint(checkpoint, device=device)
    c = loaded.contract
    check = rd.check_exports([export_root], c.hand)
    check.raise_if_bad()
    arrays_all = rd.load_episode_arrays(export_root, c)
    full = rd.ViewVideos(export_root, c.cameras)
    # The view this checkpoint trained on, when it is here: the image gap measured is then this checkpoint's.
    source = str(Path(export_root).resolve())
    view_root = next(
        (Path(v["root"]) for v in loaded.provenance.get("views") or []
         if v.get("source") == source and (Path(v["root"]) / rd.VIEW_FILE).is_file()),
        None,
    ) or rd.build_view(export_root, c.cameras, views_root, log=log)
    view = rd.ViewVideos(view_root, c.cameras)
    trained_on = {e.get("name") for e in loaded.provenance.get("exports") or []}
    wanted = set(episodes) if episodes else None
    errs: dict[str, list[np.ndarray]] = {"pos_mm": [], "rot_deg": [], "width_mm": []}
    still: list[np.ndarray] = []
    view_diff: list[float] = []
    frames_out: list[dict[str, Any]] = []
    for arrays in arrays_all:
        if wanted is not None and arrays.episode not in wanted:
            continue
        for t in range(0, len(arrays.poses), max(1, stride)):
            sample = rd.sample_vectors(c, arrays, t)
            if sample is None:
                continue
            state, gt, pad = sample
            if pad.all():
                continue
            hist_i, _ = rd._clamped(len(arrays.poses), t, c.history_offsets)
            T_hist = rt.pose7_to_matrix(arrays.poses[hist_i])
            wrench = arrays.wrench[t] if c.wrench_names else None
            images = {cam.name: full.frame(arrays.episode, t, cam.name) for cam in c.cameras}
            obs = rtp.observation_sample(c, T_hist, float(arrays.widths[t]), images, wrench)
            if not np.allclose(obs["observation.state"].numpy(), state, atol=1e-6):
                raise RuntimeError("the online state path and the training state path disagree")
            pred = loaded.predict(obs)
            cropped = {cam.name: view.frame(arrays.episode, t, cam.name) for cam in c.cameras}
            pred_view = loaded.predict(rtp.observation_sample(c, T_hist, float(arrays.widths[t]), cropped, wrench,
                                                              already_cropped=True))
            e = chunk_errors(pred, gt, pad)
            for k in errs:
                errs[k].append(e[k])
            zero = np.tile(np.concatenate([rt.encode(np.eye(4)), [arrays.widths[t]]]), (len(gt), 1))
            still.append(chunk_errors(zero, gt, pad)["pos_mm"])
            pv = rt.decode(pred_view[:, : rt.POSE_DIM])[:, :3, 3]
            view_diff.append(float(np.abs(pv - rt.decode(pred[:, : rt.POSE_DIM])[:, :3, 3]).max() * 1e3))
            T_obs = T_hist[-1]
            pred_world, _ = rt.targets_from_actions(T_obs, pred)
            gt_world, _ = rt.targets_from_actions(T_obs, gt)
            frames_out.append({
                "episode": arrays.episode,
                "frame": t,
                "tcp": T_obs[:3, 3].round(5).tolist(),
                "pred": pred_world[:, :3, 3].round(5).tolist(),
                "truth": np.where(pad[:, None], np.nan, gt_world[:, :3, 3]).round(5).tolist(),
                "pred_width": pred[:, rt.POSE_DIM].round(5).tolist(),
                "truth_width": gt[:, rt.POSE_DIM].round(5).tolist(),
                "pos_err_mm_end": None if pad[-1] else round(float(e["pos_mm"][-1]), 2),
            })
        progress(event="offline", episode=arrays.episode, frames=len(frames_out))
    if not frames_out:
        raise rd.TrainingDataError("no frame of these episodes could be predicted")
    return {
        "schema": RESULT_SCHEMA,
        "kind": "offline",
        "created": _now(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "export": str(Path(export_root).resolve()),
        "trained_on_this_export": Path(export_root).name in trained_on,
        "contract": c.to_dict(),
        "episodes": sorted({f["episode"] for f in frames_out}),
        "stride": stride,
        "frames": len(frames_out),
        "error": {k: _summ(v) for k, v in errs.items()},
        "not_moving_baseline_pos_mm": _summ(still),
        "view_vs_online_path_mm": {
            "max": max(view_diff),
            "mean": float(np.mean(view_diff)),
            "note": "predicted target positions from the training view's frames (what training saw) against "
            "the full-resolution frame cropped online; the codec chain of the view is the only difference",
        },
        "series": frames_out,
    }


# ---------------------------------------------------------- execution plan ---


class PlanContext:
    """What the execution plan check needs about one arm, read once: tool frame, edge, workspace, scene."""

    def __init__(self, arm: str, workspace: dict[str, Any] | None, repo_root: Path = _REPO_ROOT):
        from tools.fr3 import (
            dual_fr3_collision as col,
            dual_fr3_scene as scene,
            fr3_arms,
            fr3_kinematics as kin,
        )
        from tools.thor.gmsl2 import world_graph_guard as guard, world_provenance as wp

        self.arm = arm
        self.kin = kin
        self.col = col
        self.problems: list[str] = []
        registry = fr3_arms.load(repo_root=repo_root)
        entry = fr3_arms.resolve(registry, arm)
        tool = fr3_arms.tool_model(entry, repo_root=repo_root)
        urdf = repo_root / str(tool.get("urdf_path") or kin.FR3_URDF)
        frame = str(tool.get("target_frame_name") or "")
        if frame != rt.ARM_FK_FRAME:
            raise ValueError(f"the {arm} arm drives {frame!r}; the policy's TCP is {rt.ARM_FK_FRAME!r} (relative_tcp)")
        self.T_tool = kin.urdf_fixed_transform(rt.ARM_FK_FRAME, urdf=urdf)
        T_label = kin.urdf_fixed_transform(rt.TCP_LINK, urdf=urdf)
        if not np.allclose(self.T_tool[:3, :3], T_label[:3, :3], atol=1e-9):
            raise ValueError(f"{rt.ARM_FK_FRAME} and {rt.TCP_LINK} do not share their axes in {urdf}")
        self.base = fr3_arms.base_world_id(arm)
        if arm != "right":
            self.problems.append(f"the plan check models the right arm's scene; the {arm} arm is not modelled yet")
        graph = wp.read_world_graph(repo_root)
        status = guard.check(repo_root, graph)
        self.T_base_world = None
        if guard.blocks(status):
            self.problems.append("world-graph guard: " + guard.note(status))
        elif not workspace:
            self.problems.append("the checkpoint carries no training workspace (trained before 11-16 evening)")
        else:
            found = wp.world_transform(graph, workspace["frame"], self.base)
            if found is None:
                note = wp.retired_note(graph, workspace["frame"], self.base)
                self.problems.append(f"no live edge from {workspace['frame']} to {self.base}" + (f": {note}" if note else ""))
            else:
                self.T_base_world = np.asarray(found[0], dtype=np.float64)
        self.workspace = workspace
        self.model = col.load()
        self.obstacles = scene.environment_obstacles()
        cams, _ = scene.camera_obstacles(repo_root, graph)
        self.obstacles += cams

    def evaluate(
        self, T_obs: np.ndarray, actions: np.ndarray, seed: np.ndarray, *, early_exit: bool = False
    ) -> dict[str, Any]:
        """Per step: targets = T_obs · ΔT, their IK from ``seed``, workspace, clearance; ok/why per step.

        ``T_obs`` is the TCP (ARM_FK_FRAME) in the arm's base at the observation; ``seed`` one posture (7,) or
        one per step (n, 7). ``early_exit`` is the
        execution loop's IK (kin.ik: stop when every target has converged, the posture nearest the seed).
        """
        kin = self.kin
        targets, widths = rt.targets_from_actions(T_obs, actions)
        flange = targets @ np.linalg.inv(self.T_tool)
        q_sol, ik_ok = kin.ik_retry(kin.NOMINAL, flange, np.asarray(seed, dtype=np.float64), early_exit=early_exit)
        n = len(targets)
        inside = np.ones(n, dtype=bool)
        if self.T_base_world is not None:
            T_world_base = np.linalg.inv(self.T_base_world)
            pts = (T_world_base[:3, :3] @ targets[:, :3, 3].T).T + T_world_base[:3, 3]
            lo = np.asarray(self.workspace["p005"]) - WORKSPACE_MARGIN_M
            hi = np.asarray(self.workspace["p995"]) + WORKSPACE_MARGIN_M
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        A, B = self.model.place(kin.NOMINAL, q_sol)
        env = self.col.obstacle_gaps(self.obstacles, A, B, self.model.r, self.model.table_mask())
        gap = env.reshape(n, -1).min(axis=1) if env.size else np.full(n, math.inf)
        why = []
        for k in range(n):
            if self.problems:
                why.append("; ".join(self.problems))
            elif not ik_ok[k]:
                why.append(f"step {k + 1}: IK not reachable inside the joint walls")
            elif not inside[k]:
                why.append(f"step {k + 1}: leaves the training workspace (+{WORKSPACE_MARGIN_M * 100:.0f} cm)")
            elif gap[k] < ENV_MARGIN_M:
                why.append(f"step {k + 1}: {gap[k] * 1e3:.0f} mm from the carts/table/cameras (needs {ENV_MARGIN_M * 1e3:.0f})")
            else:
                why.append("")
        return {
            "T": targets,
            "q": q_sol,
            "widths": widths,
            "ok": np.array([not w for w in why]),
            "why": why,
            "ik_ok": ik_ok,
            "inside_workspace": None if self.T_base_world is None else inside,
            "env_gap_m": gap,
        }

    def check(self, q_obs: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
        """The preview's check: every step of the chunk, the arm standing at ``q_obs``."""
        q_obs = np.asarray(q_obs, dtype=np.float64)
        T_obs = self.kin.NOMINAL.fk(q_obs[None])[0] @ self.T_tool
        ev = self.evaluate(T_obs, actions, q_obs)
        ok, inside, gap = ev["ik_ok"], ev["inside_workspace"], ev["env_gap_m"]
        reasons = list(self.problems)
        if not ok.all():
            reasons.append(f"IK: {int((~ok).sum())} of {len(ok)} targets not reachable inside the joint walls")
        if inside is not None and not inside.all():
            reasons.append(f"workspace: {int((~inside).sum())} targets leave the training workspace (+{WORKSPACE_MARGIN_M * 100:.0f} cm)")
        min_env = float(gap.min()) if gap.size else math.inf
        if min_env < ENV_MARGIN_M:
            reasons.append(f"collision: {min_env * 1e3:.0f} mm from the carts/table/cameras (needs {ENV_MARGIN_M * 1e3:.0f})")
        return {
            "ok": not reasons,
            "reasons": reasons,
            "tcp_base": T_obs[:3, 3].round(4).tolist(),
            "targets_base": ev["T"][:, :3, 3].round(4).tolist(),
            "widths": ev["widths"].round(4).tolist(),
            "ik_ok": ok.tolist(),
            "inside_workspace": None if inside is None else inside.tolist(),
            "min_env_gap_mm": None if not math.isfinite(min_env) else round(min_env * 1e3, 1),
        }

    def tcp(self, q: np.ndarray) -> np.ndarray:
        """FK to ARM_FK_FRAME in the arm's base."""
        return self.kin.NOMINAL.fk(np.asarray(q, dtype=np.float64)[None])[0] @ self.T_tool


# ------------------------------------------------------------------ preview ---


class LiveInputs:
    """The recorder's frame bus and BOX_LIVE snapshot (through the gateway), for one policy contract."""

    def __init__(self, contract: rt.PolicyContract, *, gateway: str, box_id: str, frame_bus: Path = PREVIEW_FRAME_BUS):
        from tools.thor.gmsl2.online_sync_frame_client import ThorOnlineSyncFrameClient

        self.contract = contract
        self.gateway = gateway.rstrip("/")
        self.box_id = box_id
        self.client = ThorOnlineSyncFrameClient(frame_bus, cameras=[c.name for c in contract.cameras])
        self.last_seq: int | None = None

    def keep_previews_on(self) -> None:
        """The recorder publishes the preview frame bus only while someone asks for previews (an idle TTL);
        polling a camera preview is how the gateway passes that demand on (as the Device Manager grid does)."""
        if time.monotonic() - getattr(self, "_demanded", 0.0) < 1.0:
            return
        self._demanded = time.monotonic()
        name = self.contract.cameras[0].name
        with contextlib.suppress(OSError, ValueError):
            urllib.request.urlopen(f"{self.gateway}/api/device-preview/camera.jpg?key={name}", timeout=2).close()

    def frames(self, timeout_s: float = 2.0) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.keep_previews_on()
        cluster = self.client.get_latest(timeout_s=timeout_s, min_publish_seq=self.last_seq)
        if cluster is None:
            raise RuntimeError("no frame from the recorder (is it connected, with previews on?)")
        self.last_seq = cluster.publish_seq + 1
        images = {}
        for cam in self.contract.cameras:
            frame = cluster.frames.get(cam.name)
            if frame is None:
                raise RuntimeError(f"the recorder publishes no {cam.name}")
            images[cam.name] = frame.as_rgb()
        return images, {"publish_seq": cluster.publish_seq, "sync_ns": cluster.sync_timestamp_ns,
                        "max_delta_ns": cluster.max_delta_ns, "recording": cluster.recording}

    def frames_since(self, min_t_s: float, timeout_s: float = 1.5) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
        """The newest cluster whose capture instant is at or after ``min_t_s`` (CLOCK_MONOTONIC seconds):
        (images, instant, meta). The instant is the median SOF ``sensor_timestamp_ns`` of the policy's
        cameras -- CLOCK_MONOTONIC on Thor (measured 11-18: 60-263 ms before time.monotonic at read)."""
        deadline = time.monotonic() + timeout_s
        while True:
            self.keep_previews_on()
            cluster = self.client.get_latest(timeout_s=0.05)
            if cluster is not None:
                stamps = [cluster.frames[c.name].sensor_timestamp_ns for c in self.contract.cameras if c.name in cluster.frames]
                if len(stamps) == len(self.contract.cameras):
                    t_img = float(np.median(stamps)) / 1e9
                    if t_img >= min_t_s:
                        images = {c.name: cluster.frames[c.name].as_rgb() for c in self.contract.cameras}
                        return images, t_img, {"publish_seq": cluster.publish_seq, "max_delta_ns": cluster.max_delta_ns,
                                               "recording": cluster.recording}
            if time.monotonic() >= deadline:
                raise RuntimeError(f"no frame of {', '.join(c.name for c in self.contract.cameras)} newer than the "
                                   f"last observation within {timeout_s:g} s (recorder connected, previews on?)")
            time.sleep(0.01)

    def box(self) -> tuple[float, np.ndarray | None, dict[str, Any]]:
        with urllib.request.urlopen(f"{self.gateway}/api/device-preview/box?device=x", timeout=2) as answer:
            payload = json.loads(answer.read())
        if not payload.get("active"):
            raise RuntimeError("no live BOX data (recorder not connected, or the BOX stream is stale)")
        snapshot = {"sensors": payload.get("sensors") or {}, "status": payload.get("status") or {}}
        return self.box_from_snapshot(snapshot, payload)

    def box_from_snapshot(self, snapshot: dict[str, Any], payload: dict[str, Any] | None = None):
        """(width, wrench, meta) of the arm's gripper BOX in a BOX_LIVE snapshot, in the recorder's own layout."""
        from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

        ids = [i for i in lr3.box_ids_from_snapshot(snapshot) if i]
        if ids:
            mine = [i for i in ids if i.removeprefix("box") == self.box_id]
            if not mine:
                raise RuntimeError(f"the arm's gripper BOX {self.box_id} is not in the live data (BOXes: {', '.join(ids)})")
            box = mine[0]
            names = list(lr3.box_state_names((box,)))
            values = lr3.box_snapshot_to_state_for_boxes(snapshot, (box,))
            prefix = f"{box}/"
        else:  # one BOX, unprefixed: it must be the arm's (the status says which)
            device = str((snapshot.get("status") or {}).get("device_id") or "")
            if device and device != self.box_id:
                raise RuntimeError(f"the live BOX is {device}, not the arm's gripper BOX {self.box_id}")
            names = list(lr3.box_state_names())
            values = lr3.box_snapshot_to_state(snapshot)
            prefix = ""
        width_i = next((i for i, n in enumerate(names) if n.endswith(rd.WIDTH_SUFFIX)), None)
        if width_i is None:
            raise RuntimeError(f"the live BOX state has no {rd.WIDTH_SUFFIX}")
        wrench = None
        if self.contract.wrench_names:
            six = snapshot["sensors"].get(prefix + "box_six_d_force") or {}
            ext = six.get("fxyz_mxyz_external")
            if not ext:
                raise RuntimeError(f"no live external wrench for BOX {self.box_id}: {six.get('external_note') or 'no 6D reading'}")
            wrench = np.asarray(ext[:6], dtype=np.float64)
        payload = payload or {}
        return float(values[width_i]), wrench, {"updatedAt": payload.get("updatedAt"), "staleS": payload.get("staleS")}


def read_arm_joints(arm: str, repo_root: Path = _REPO_ROOT) -> np.ndarray:
    """The arm's joints, read once by fr3_print_joints_runtime (connect, read, disconnect: no motion), in the
    host python that has panda_py, with its library path (fr3_record.build_host_env)."""
    import subprocess

    from tools.fr3 import fr3_arms, fr3_record

    entry = fr3_arms.resolve(fr3_arms.load(repo_root=repo_root), arm)
    env = fr3_record.build_host_env(repo_root)
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}:" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [str(fr3_record._host_python_executable(repo_root)), "tools/fr3/fr3_print_joints_runtime.py",
         "--robot-ip", str(entry["robot_ip"])],
        cwd=repo_root, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL, env=env,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Joint angles (rad):"):
            return np.asarray(json.loads(line.split(":", 1)[1].strip()), dtype=np.float64)
    raise RuntimeError(f"could not read the {arm} arm's joints: {(result.stderr or result.stdout).strip()[-300:]}")


_STOP = False


def _stop(*_a) -> None:
    global _STOP
    _STOP = True


def preview(
    checkpoint: Path,
    record_dir: Path,
    *,
    arm: str,
    gateway: str,
    rate_hz: float = 2.0,
    max_steps: int = 0,
    q_arm: np.ndarray | None = None,
    device: str | None = None,
    frame_bus: Path = PREVIEW_FRAME_BUS,
) -> dict[str, Any]:
    """Predict on live inputs with the arm still; record each step's inputs and outputs under ``record_dir``."""
    import cv2

    from tools.fr3 import fr3_arms, relative_tcp_policy as rtp

    loaded = rtp.load_checkpoint(checkpoint, device=device)
    c = loaded.contract
    entry = fr3_arms.resolve(fr3_arms.load(), arm)
    box_id = str(entry.get("corenetic_device_id") or "")
    if not box_id:
        raise RuntimeError(f"the {arm} arm has no gripper BOX ({entry.get('no_gripper_why') or 'not wired'})")
    plan = None
    plan_problem = ""
    try:
        plan = PlanContext(arm, loaded.provenance.get("workspace"))
    except Exception as exc:  # noqa: BLE001 -- reported; preview still runs
        plan_problem = f"{type(exc).__name__}: {exc}"
    live = LiveInputs(c, gateway=gateway, box_id=box_id, frame_bus=frame_bus)
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "steps").mkdir(exist_ok=True)
    meta = {
        "schema": RECORD_SCHEMA,
        "kind": "preview",
        "created": _now(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "contract": c.to_dict(),
        "arm": arm,
        "box_id": box_id,
        "q_arm": None if q_arm is None else q_arm.tolist(),
        "tcp_history": "arm still: every history pose equals the current one (exact while it does not move)",
        "wrench_bias": "live: this Connect's last BOX zero (an export takes each episode's still ends)" if c.wrench_names else None,
        "frame_source": str(frame_bus),
        "plan_problem": plan_problem,
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    period = 1.0 / max(0.1, rate_hz)
    step = 0
    history = np.repeat(np.eye(4)[None], c.history + 1, axis=0)
    while not _STOP and (max_steps <= 0 or step < max_steps):
        t0 = time.monotonic()
        try:
            images, cluster = live.frames()
            width, wrench, box_meta = live.box()
        except Exception as exc:  # noqa: BLE001 -- the page shows it and the loop keeps trying
            progress(event="wait", error=str(exc))
            time.sleep(1.0)
            continue
        cropped = {cam.name: rt.crop_resize(images[cam.name], cam) for cam in c.cameras}
        sample = rtp.observation_sample(c, history, width, cropped, wrench, already_cropped=True)
        t_pred = time.monotonic()
        actions = loaded.predict(sample)
        latency_ms = (time.monotonic() - t_pred) * 1e3
        checked = plan.check(q_arm, actions) if (plan is not None and q_arm is not None) else None
        np.savez_compressed(
            record_dir / "steps" / f"{step:06d}.npz",
            width=np.float64(width),
            wrench=np.zeros(0) if wrench is None else wrench,
            actions=actions,
            **{f"image_{name}": img for name, img in cropped.items()},
        )
        strip = np.concatenate([cv2.resize(img, (int(img.shape[1] * 240 / img.shape[0]), 240)) for img in cropped.values()], axis=1)
        cv2.imwrite(str(record_dir / "latest.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rel = rt.decode(actions[:, : rt.POSE_DIM])
        latest = {
            "step": step,
            "time": _now(),
            "cluster": cluster,
            "box": box_meta,
            "width": width,
            "wrench": None if wrench is None else wrench.round(3).tolist(),
            "latency_ms": round(latency_ms, 1),
            "targets_tcp_mm": (rel[:, :3, 3] * 1e3).round(1).tolist(),
            "widths": actions[:, rt.POSE_DIM].round(4).tolist(),
            "plan": checked,
        }
        (record_dir / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
        with (record_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(latest) + "\n")
        progress(event="preview", step=step, latency_ms=latency_ms, plan_ok=None if checked is None else checked["ok"])
        step += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    meta["steps"] = step
    meta["finished"] = _now()
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def replay(record_dir: Path, *, device: str | None = None) -> dict[str, Any]:
    """Every recorded preview step, re-predicted from its recorded inputs through the offline path."""
    from tools.fr3 import relative_tcp_policy as rtp

    meta = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    loaded = rtp.load_checkpoint(Path(meta["checkpoint"]), device=device)
    c = loaded.contract
    if c.to_dict() != meta["contract"]:
        raise RuntimeError("the checkpoint's contract changed since this record was made")
    still = np.repeat(np.eye(4)[None], c.history + 1, axis=0)
    diffs = []
    for path in sorted((record_dir / "steps").glob("*.npz")):
        data = np.load(path)
        images = {cam.name: data[f"image_{cam.name}"] for cam in c.cameras}
        wrench = data["wrench"] if c.wrench_names else None
        history = data["history"] if "history" in data.files else still  # an execute record keeps its FK history
        sample = rtp.observation_sample(c, history, float(data["width"]), images, wrench, already_cropped=True)
        diffs.append(float(np.abs(loaded.predict(sample) - data["actions"]).max()))
    if not diffs:
        raise RuntimeError("the record has no steps")
    result = {"steps": len(diffs), "max_abs_diff": max(diffs), "ok": max(diffs) < 1e-4, "checked": _now()}
    meta["replay_check"] = result
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


# ------------------------------------------------------------------ execute ---


EXECUTE_LOCKED = (
    "executing a policy on the arm is locked: a maintainer who is present unlocks one arm for a bounded time "
    "(tools/fr3/real_execution_unlock.py); until then the Rollout page runs the preview only"
)


def execute(
    checkpoint: Path,
    record_dir: Path,
    *,
    arm: str,
    stage: int,
    gateway: str,
    frame_bus: Path = PREVIEW_FRAME_BUS,
    device: str | None = None,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, Any]:
    """One staged run on the arm (rollout_execute). The unlock is read before anything is loaded."""
    import cv2

    from tools.fr3 import (
        fr3_arms,
        fr3_kinematics as kin,
        real_execution_unlock,
        relative_tcp_policy as rtp,
        rollout_execute as rx,
    )
    from tools.thor.gmsl2 import cart_state

    unlock = real_execution_unlock.read(arm, repo_root=repo_root, locked=EXECUTE_LOCKED)
    if stage not in rx.STAGES:
        raise ValueError(f"stage {stage}: one of {sorted(rx.STAGES)}")
    st = rx.STAGES[stage]
    limits = rx.Limits()
    loaded = rtp.load_checkpoint(checkpoint, device=device)
    c = loaded.contract
    n_exec = int(getattr(rtp._config(loaded.policy), "n_action_steps", 0) or 0)  # what training's eval executed
    if not 1 <= n_exec <= c.horizon:
        raise RuntimeError(f"the checkpoint's n_action_steps ({n_exec}) is not within 1..{c.horizon}")
    entry = fr3_arms.resolve(fr3_arms.load(repo_root=repo_root), arm)
    box_id = str(entry.get("corenetic_device_id") or "")
    if not box_id:
        raise RuntimeError(f"the {arm} arm has no gripper BOX ({entry.get('no_gripper_why') or 'not wired'})")
    plan = PlanContext(arm, loaded.provenance.get("workspace"), repo_root=repo_root)
    if plan.problems:
        raise RuntimeError("the plan check cannot vouch for this arm: " + "; ".join(plan.problems))
    cart = (cart_state.read(repo_root).get("bases") or {}).get(plan.base) or {}
    live = LiveInputs(c, gateway=gateway, box_id=box_id, frame_bus=frame_bus)

    def observe(min_t: float) -> rx.Observation:
        images, t_img, meta = live.frames_since(min_t)
        width, wrench, box_meta = live.box()
        updated = box_meta.get("updatedAt")
        meta = {**meta, "box": box_meta,
                "box_stale_s": None if updated is None else max(0.0, time.time() - float(updated))}
        return rx.Observation(t_img, {cam.name: rt.crop_resize(images[cam.name], cam) for cam in c.cameras},
                              width, wrench, meta)

    def sample(history: np.ndarray, obs: rx.Observation) -> dict[str, Any]:
        return rtp.observation_sample(c, history, obs.width, obs.images, obs.wrench, already_cropped=True)

    def plan_fn(T_obs: np.ndarray, actions: np.ndarray, seed: np.ndarray) -> dict[str, Any]:
        return plan.evaluate(T_obs, actions, seed, early_exit=True)

    # warm-up on a live observation, arm untouched: the first CUDA call is slower than the 100 ms allowed
    first = observe(time.monotonic() - 1.0)
    for _ in range(2):
        loaded.predict(sample(np.repeat(np.eye(4)[None], c.history + 1, axis=0), first))

    record_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "schema": rx.RECORD_SCHEMA,
        "kind": "execute",
        "created": _now(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "contract": c.to_dict(),
        "arm": arm,
        "box_id": box_id,
        "n_action_steps": n_exec,
        "unlock": unlock,
        "cart_check": cart or None,
        **rx.stage_dict(st, limits),
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    recorder = rx.Recorder(record_dir)
    history = rx.PoseHistory()
    predictor = rx.Predictor(contract=c, history=history, observe=observe, predict=loaded.predict, plan=plan_fn,
                             sample=sample, limits=limits)
    stop = {"why": ""}

    def _signal(*_a) -> None:
        stop["why"] = stop["why"] or "stopped by the operator"

    def _gripper_error(message: str) -> None:
        stop["why"] = stop["why"] or message

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    gripper = rx.GatewayGripper(gateway, arm, _gripper_error) if st.gripper else None
    progress(event="execute", stage=stage, label=st.label, n_action_steps=n_exec, max_s=st.max_s,
             cart=cart.get("verdict"), unlock_by=unlock.get("by"))
    if stop["why"]:
        raise RuntimeError(f"{stop['why']} before the arm was connected; nothing moved")
    panda = rx.PandaArm(str(entry["robot_ip"]))
    predictor.start()
    sys.setswitchinterval(0.001)
    resume_gc = rx.hold_gc()
    try:
        result = rx.run_loop(
            arm=panda, predictor=predictor, stage=st, n_exec=n_exec, limits=limits, fk_tcp=plan.tcp,
            wall_lower=kin._WALL_LOWER, wall_upper=kin._WALL_UPPER, history=history,
            stop_flag=lambda: stop["why"], log=progress, gripper=gripper, width_now=first.width,
            record=recorder,
        )
    finally:
        resume_gc()
        recorder.close()
    with contextlib.suppress(Exception):
        strip = np.concatenate([cv2.resize(img, (int(img.shape[1] * 240 / img.shape[0]), 240)) for img in first.images.values()], axis=1)
        cv2.imwrite(str(record_dir / "latest.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta["result"] = {"completed": result.completed, "stopped": result.stopped, "chunks": result.chunks_accepted,
                      "duration_s": round(result.duration_s, 3), "ticks": result.ticks,
                      "end_cursor": round(result.end_cursor, 2)}
    meta["finished"] = _now()
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


# --------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    off = sub.add_parser("offline")
    off.add_argument("--checkpoint", type=Path, required=True)
    off.add_argument("--export", type=Path, required=True)
    off.add_argument("--episode", type=int, action="append", default=[])
    off.add_argument("--stride", type=int, default=15)
    off.add_argument("--out", type=Path, required=True)
    pre = sub.add_parser("preview")
    pre.add_argument("--checkpoint", type=Path, required=True)
    pre.add_argument("--record", type=Path, required=True)
    pre.add_argument("--arm", choices=("right", "left"), required=True)
    pre.add_argument("--gateway", default="http://127.0.0.1:8765")
    pre.add_argument("--rate", type=float, default=2.0)
    pre.add_argument("--frame-bus", type=Path, default=PREVIEW_FRAME_BUS)
    pre.add_argument("--steps", type=int, default=0)
    pre.add_argument("--q", default="", help="the arm's joints (JSON list, rad) for the plan check")
    pre.add_argument("--read-joints", action="store_true", help="read the arm's joints once at the start (no motion)")
    rep = sub.add_parser("replay")
    rep.add_argument("--record", type=Path, required=True)
    exe = sub.add_parser("execute", help="moves the arm: a person at the e-stop, the maintainer's unlock")
    exe.add_argument("--checkpoint", type=Path, required=True)
    exe.add_argument("--record", type=Path, required=True)
    exe.add_argument("--arm", choices=("right", "left"), required=True)
    exe.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    exe.add_argument("--gateway", default="http://127.0.0.1:8765")
    exe.add_argument("--frame-bus", type=Path, default=PREVIEW_FRAME_BUS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "offline":
            result = offline(args.checkpoint, args.export, episodes=args.episode or None, stride=args.stride)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result), encoding="utf-8")
            progress(event="done", out=str(args.out), frames=result["frames"],
                     pos_mm=result["error"]["pos_mm"].get("mean"))
        elif args.cmd == "preview":
            q = np.asarray(json.loads(args.q), dtype=np.float64) if args.q else None
            if q is None and args.read_joints:
                q = read_arm_joints(args.arm)
                progress(event="joints", q=q.round(5).tolist())
            meta = preview(args.checkpoint, args.record, arm=args.arm, gateway=args.gateway, rate_hz=args.rate,
                           max_steps=args.steps, q_arm=q, frame_bus=args.frame_bus)
            progress(event="done", steps=meta["steps"])
        elif args.cmd == "execute":
            meta = execute(args.checkpoint, args.record, arm=args.arm, stage=args.stage, gateway=args.gateway,
                           frame_bus=args.frame_bus)
            r = meta["result"]
            progress(event="done", completed=r["completed"], stopped=r["stopped"], chunks=r["chunks"],
                     duration_s=r["duration_s"])
        else:
            progress(event="done", **replay(args.record))
    except Exception as exc:  # noqa: BLE001 -- one line the page can show
        with contextlib.suppress(Exception):
            progress(event="error", error=f"{type(exc).__name__}: {exc}")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
