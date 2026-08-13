#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Headless FR3 SpaceMouse recorder driven by the data-collection gateway.

The terminal recorder (:mod:`tools.fr3.fr3_record_runtime`) blocks on a keyboard listener and
``input()`` prompts, so it cannot be operated from the web UI. This module keeps the exact same
recording pipeline -- same ee2ee processors, same ``record_loop``, same dataset feature
contract -- and only swaps the operator interface for the gateway's line protocol:

    stdin (gateway -> recorder)     stdout (recorder -> gateway)
    ---------------------------     ----------------------------
    ""      start the armed episode  Dataset root: <path>
    save    keep the current episode  Cameras: <ids>
    n       drop the current episode  Episode <n> ready
    q       stop the session          Recorded <n> frames
    exit    shut down                 Episode saved / Episode discarded
                                      Total saved episodes: <n>
                                      SYNC <one-line timestamp audit>

``--backend real`` records the hardware FR3; ``--backend sim`` records the MuJoCo twin through
the same ``Robot`` interface, from the *same* YAML, so the two datasets are schema-identical
and a sim episode can be replayed against hardware tooling without conversion.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import queue
import re
import sys
import threading
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lerobot.configs import parser  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import DEFAULT_TASKS_PATH, LEGACY_TASKS_PATH  # noqa: E402
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
import lerobot.robots.franka_research3  # noqa: E402,F401  # registers both FR3 robot choices
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.robots.franka_research3 import FrankaResearch3MujocoConfig  # noqa: E402
from lerobot.scripts.lerobot_record import RecordConfig, record_loop  # noqa: E402
import lerobot.teleoperators.spacemouse  # noqa: E402,F401
from lerobot.teleoperators import make_teleoperator_from_config  # noqa: E402
from lerobot.utils.control_utils import (  # noqa: E402
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from tools.fr3.fr3_record_runtime import (  # noqa: E402
    build_fr3_ee2ee_dataset_features,
    make_fr3_action_processors,
)
from tools.fr3.fr3_sync_audit import (  # noqa: E402
    DEFAULT_BIAS_TOLERANCE_MS,
    DEFAULT_GLOBAL_LAG_TOLERANCE_MS,
    DEFAULT_TOLERANCE_MS,
    format_episode_sync_line,
    format_sync_summary_line,
    residual_tolerance_for_camera_fps,
    summarize_episode_capture_timestamps,
    write_fr3_sync_report,
)

_RUNTIME_ARGS: argparse.Namespace | None = None
_PROGRESS_INTERVAL_S = 0.25
_CAPTURE_TIMESTAMP_FEATURE = "observation.device_capture_timestamp"
# Wrist-ish camera names map onto the MuJoCo end-effector camera; everything else looks at the
# scene from outside. Overridable with --sim-camera-map when a rig uses different names.
_WRIST_CAMERA_HINTS = ("ee", "wrist", "hand", "eih", "gripper")
_SIM_WRIST_CAMERA = "ee_cam"
_SIM_EXTERNAL_CAMERA = "external_cam"


def emit(line: str) -> None:
    """Write one protocol line. Never routed through logging, which the gateway treats as noise."""
    print(line, flush=True)


class _CommandChannel:
    """Non-blocking view of the gateway's stdin command stream.

    ``record_loop`` owns the control thread for the duration of an episode, so commands that
    arrive mid-episode have to be observed through the shared ``events`` dict it already polls.
    A single reader thread parses lines and both flips those flags and queues the command, so
    nothing is lost between episodes either.
    """

    def __init__(self, events: dict[str, bool]) -> None:
        self._events = events
        self._queue: queue.Queue[str] = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="fr3-gui-record-stdin", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        for raw_line in sys.stdin:
            command = raw_line.strip().lower()
            # A bare newline is the gateway's "start episode" keypress; everything else is a word.
            if command in ("save", "y"):
                self._events["exit_early"] = True
            elif command in ("n", "discard"):
                self._events["exit_early"] = True
            elif command in ("q", "quit", "exit", "stop"):
                self._events["exit_early"] = True
                self._events["stop_recording"] = True
            self._queue.put(command)
        # stdin closed: the gateway process is gone, so wind the session down cleanly.
        self._events["exit_early"] = True
        self._events["stop_recording"] = True
        self._closed.set()
        self._queue.put("exit")

    def wait_for_command(self, timeout: float | None = None) -> str | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_latest(self) -> str | None:
        """Return the most recent queued command, discarding anything older."""
        latest: str | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest


class _ProgressReporter:
    """Emit ``Recorded <n> frames`` while ``record_loop`` runs, so the UI's bar moves."""

    def __init__(self, dataset: LeRobotDataset) -> None:
        self._dataset = dataset
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_ProgressReporter":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fr3-gui-record-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        emit(f"Recorded {self._frame_count()} frames")

    def _frame_count(self) -> int:
        buffer = getattr(self._dataset, "episode_buffer", None)
        if not isinstance(buffer, dict):
            return 0
        return int(buffer.get("size", 0))

    def _run(self) -> None:
        last_reported = -1
        while not self._stop.wait(_PROGRESS_INTERVAL_S):
            count = self._frame_count()
            if count != last_reported:
                last_reported = count
                emit(f"Recorded {count} frames")


def parse_runtime_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    arg_parser = argparse.ArgumentParser(
        description="Gateway-driven FR3 SpaceMouse recorder (hardware or MuJoCo).",
        add_help=False,
    )
    arg_parser.add_argument(
        "--backend",
        choices=("real", "sim"),
        default="real",
        help="Record the hardware FR3 ('real') or its MuJoCo twin ('sim').",
    )
    arg_parser.add_argument(
        "--sim-camera-map",
        default="",
        help=(
            "Override the dataset-camera -> MuJoCo-camera mapping used by --backend sim, "
            "e.g. 'ee=ee_cam,side=external_cam'."
        ),
    )
    arg_parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help="Budget for skew within one modality group (the cameras against each other).",
    )
    arg_parser.add_argument(
        "--sync-global-lag-tolerance-ms", type=float, default=DEFAULT_GLOBAL_LAG_TOLERANCE_MS
    )
    arg_parser.add_argument(
        "--sync-residual-tolerance-ms",
        type=float,
        default=None,
        help=(
            "Budget for skew once each device's constant offset is removed. Derived from the "
            "config's slowest camera rate when unset, because one sensor period is its floor."
        ),
    )
    arg_parser.add_argument(
        "--sync-bias-tolerance-ms", type=float, default=DEFAULT_BIAS_TOLERANCE_MS
    )
    arg_parser.add_argument(
        "--no-sync-audit",
        action="store_true",
        help="Skip the per-episode timestamp-synchronisation audit.",
    )
    # No -h here on purpose: it falls through to draccus so `--help` documents the full
    # RecordConfig surface rather than just these few runtime switches.
    return arg_parser.parse_known_args(argv)


def _resolve_workspace_path(value: str) -> str:
    """Map a container-style ``/lerobot/...`` config path onto this checkout."""
    path = Path(value)
    if str(path).startswith("/lerobot/"):
        return str(_REPO_ROOT / path.relative_to("/lerobot"))
    return value


def _parse_sim_camera_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"--sim-camera-map entries must look like name=mujoco_camera, got {entry!r}")
        dataset_name, model_name = entry.split("=", 1)
        mapping[dataset_name.strip()] = model_name.strip()
    return mapping


def _default_sim_camera_name(dataset_camera_name: str) -> str:
    lowered = dataset_camera_name.lower()
    if any(hint in lowered for hint in _WRIST_CAMERA_HINTS):
        return _SIM_WRIST_CAMERA
    return _SIM_EXTERNAL_CAMERA


def build_sim_robot_config(
    cfg: RecordConfig,
    *,
    camera_map_override: dict[str, str] | None = None,
) -> FrankaResearch3MujocoConfig:
    """Derive the MuJoCo robot config from the same YAML the hardware run uses.

    Camera *keys*, workspace bounds and per-step delta clamps are copied verbatim so the sim
    dataset carries the same feature names and the same safety envelope as the hardware one.
    Only the scene assets and the physics settings are sim-specific.
    """
    robot_cfg = cfg.robot
    hardware_cameras = dict(getattr(robot_cfg, "cameras", {}) or {})
    camera_names = tuple(hardware_cameras) or ("external", "wrist")

    widths = [int(c.width) for c in hardware_cameras.values() if getattr(c, "width", None)]
    heights = [int(c.height) for c in hardware_cameras.values() if getattr(c, "height", None)]
    if len({*widths}) > 1 or len({*heights}) > 1:
        # A single MuJoCo renderer serves every camera, so mixed hardware resolutions cannot be
        # reproduced. Say so instead of silently recording one of them at the wrong size.
        raise ValueError(
            "--backend sim requires all cameras in the config to share one resolution; "
            f"got widths={sorted(set(widths))} heights={sorted(set(heights))}."
        )

    override = camera_map_override or {}
    camera_name_mapping = {
        name: override.get(name, _default_sim_camera_name(name)) for name in camera_names
    }

    urdf_path = _resolve_workspace_path(str(getattr(robot_cfg, "urdf_path", "") or ""))
    sim_xml_path = ""
    if urdf_path:
        candidate = Path(urdf_path).with_suffix(".xml")
        scene_candidate = candidate.with_name(f"{candidate.stem}_scene.xml")
        if scene_candidate.is_file():
            sim_xml_path = str(scene_candidate)
        elif candidate.is_file():
            sim_xml_path = str(candidate)

    sim_kwargs: dict[str, Any] = {
        "id": getattr(robot_cfg, "id", None),
        "urdf_path": urdf_path,
        "sim_xml_path": sim_xml_path,
        "target_frame_name": str(getattr(robot_cfg, "target_frame_name", "pika_task_tcp")),
        "camera_names": camera_names,
        "camera_name_mapping": camera_name_mapping,
        "camera_width": widths[0] if widths else 640,
        "camera_height": heights[0] if heights else 480,
        "workspace_min": tuple(getattr(robot_cfg, "workspace_min", (0.2, -0.6, 0.05))),
        "workspace_max": tuple(getattr(robot_cfg, "workspace_max", (0.9, 0.6, 0.8))),
        "max_target_delta_pos": getattr(robot_cfg, "max_target_delta_pos", None),
        "max_target_delta_rot": getattr(robot_cfg, "max_target_delta_rot", None),
        "camera_max_skew_ms": float(getattr(robot_cfg, "camera_max_skew_ms", 20.0)),
        "teleop_control_frequency": float(cfg.control_fps or cfg.dataset.fps),
    }
    joint_names = list(getattr(robot_cfg, "joint_names", []) or [])
    if joint_names:
        sim_kwargs["joint_names"] = joint_names
    return FrankaResearch3MujocoConfig(**sim_kwargs)


def _build_robot(cfg: RecordConfig, runtime_args: argparse.Namespace):
    if runtime_args.backend == "sim":
        sim_cfg = build_sim_robot_config(
            cfg, camera_map_override=_parse_sim_camera_map(runtime_args.sim_camera_map)
        )
        return make_robot_from_config(sim_cfg), sim_cfg
    cfg.robot.urdf_path = _resolve_workspace_path(cfg.robot.urdf_path)
    return make_robot_from_config(cfg.robot), cfg.robot


def _dataset_root_for_backend(cfg: RecordConfig, backend: str) -> str:
    """Keep sim and hardware captures in separate dataset roots.

    Mixing them under one repo id would produce a dataset whose episodes silently come from two
    different worlds; the schema is identical by design, so nothing else would flag it.
    """
    root = _resolve_workspace_path(str(cfg.dataset.root or ""))
    if backend != "sim" or not root:
        return root
    path = Path(root)
    return str(path) if path.name.endswith("_sim") else str(path.parent / f"{path.name}_sim")


# What the gateway strips off a directory name to attribute a session to its dataset. Matching
# it here is what keeps a session countable: `_dataset_name_prefixes` recognises exactly this.
_SESSION_STAMP_RE = re.compile(r"_\d{8}_\d{6}(?:_\d{2})?$")


def _session_dataset_root(cfg: RecordConfig, backend: str) -> str:
    """This session's own root: the configured name plus the instant the session started.

    The config names a *series* of recordings, not one dataset. Writing every session into one
    fixed root meant a run's identity was the moment you happened to look at the directory --
    change a camera rate, a gripper, the lighting, and the episodes landed in the same pile with
    nothing to separate them. Stamping the root makes each session a dataset, which is what the
    docker recorder (`fr3_record.py`) and the Thor recorder have always done, and what the
    gateway's episode counter already expects when it strips this suffix back off.

    Two cases keep the configured root verbatim: ``cfg.resume``, which asks for one specific
    dataset by definition, and a root that is already stamped, so pointing the recorder at an
    existing session extends it instead of nesting a second stamp inside the first.
    """
    root = _dataset_root_for_backend(cfg, backend)
    if not root or cfg.resume:
        return root
    path = Path(root)
    if _SESSION_STAMP_RE.search(path.name):
        return str(path)
    return str(path.parent / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _assert_resumable_or_absent(dataset_root: str) -> bool:
    """Is there a dataset at ``dataset_root`` that can actually be resumed?

    ``meta/info.json`` alone does not make one. A session that created the dataset and then
    discarded every episode leaves ``info.json`` behind with no task metadata, and
    ``LeRobotDatasetMetadata`` answers a missing tasks file by falling back to the Hub:
    ``get_safe_version`` -> ``list_repo_refs`` -> ``httpx`` -> ``socket.create_connection``.
    With no route to huggingface.co that connect has no timeout, so the recorder hangs there
    forever, before its first output line and before it starts reading its own stdin -- the GUI
    is left on the gateway's spawn message with no way to cancel. Decide resumability from what
    the metadata loader actually requires, and refuse the in-between state explicitly.
    """
    root = Path(dataset_root)
    if not (root / "meta" / "info.json").is_file():
        return False
    if (root / DEFAULT_TASKS_PATH).is_file() or (root / LEGACY_TASKS_PATH).is_file():
        return True
    raise RuntimeError(
        f"{root} has meta/info.json but no task metadata ({DEFAULT_TASKS_PATH}), so it is "
        "neither a fresh dataset nor a resumable one -- a previous session created it and then "
        "discarded every episode without finalizing. Loading it would fall back to the "
        "HuggingFace Hub and hang with no timeout when the Hub is unreachable. Delete "
        f"{root} and press Connect again."
    )


def _slowest_camera_fps(cfg: RecordConfig) -> float | None:
    """Frame rate of the slowest configured camera, which sets the residual-skew floor.

    Read off the config rather than the dataset: ``dataset.fps`` is deliberately lower than the
    sensor rate here (30 vs 60), and it is the *sensor* period a free-running camera's phase
    wanders over.
    """
    cameras = getattr(getattr(cfg, "robot", None), "cameras", None) or {}
    rates: list[float] = []
    for camera in cameras.values():
        fps = camera.get("fps") if isinstance(camera, dict) else getattr(camera, "fps", None)
        try:
            rate = float(fps)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            rates.append(rate)
    return min(rates) if rates else None


def _reset_gripper_to_open(robot: Any, teleop: Any | None = None) -> None:
    send_action = getattr(robot, "send_action", None)
    if not callable(send_action):
        robot_name = getattr(robot, "name", type(robot).__name__)
        raise RuntimeError(f"Robot '{robot_name}' does not support send_action().")

    send_action(
        {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": 1.0,
        }
    )
    if teleop is not None:
        set_gripper = getattr(teleop, "set_gripper", None)
        if callable(set_gripper):
            set_gripper(1.0)


def _audit_episode_buffer(
    dataset: LeRobotDataset,
    *,
    device_names: tuple[str, ...],
    clock_semantics: str,
    episode: int,
    runtime_args: argparse.Namespace,
    residual_tolerance_ms: float | None,
) -> None:
    """Report per-modality alignment for the episode still sitting in the frame buffer."""
    if runtime_args.no_sync_audit or not device_names:
        return
    buffer = getattr(dataset, "episode_buffer", None)
    if not isinstance(buffer, dict):
        return
    rows = buffer.get(_CAPTURE_TIMESTAMP_FEATURE)
    frame_timestamps = buffer.get("timestamp")
    if not rows or not frame_timestamps:
        return
    try:
        summary = summarize_episode_capture_timestamps(
            capture_timestamps=np.asarray(rows, dtype=np.float64),
            frame_timestamps=np.asarray(frame_timestamps, dtype=np.float64),
            device_names=list(device_names),
            clock_semantics=clock_semantics,
            tolerance_ms=float(runtime_args.sync_tolerance_ms),
            global_lag_tolerance_ms=float(runtime_args.sync_global_lag_tolerance_ms),
            residual_tolerance_ms=residual_tolerance_ms,
            bias_tolerance_ms=float(runtime_args.sync_bias_tolerance_ms),
        )
    except Exception as exc:  # noqa: BLE001 - an audit failure must never lose the episode
        emit(f"SYNC audit unavailable: {exc}")
        return
    emit(f"SYNC {format_episode_sync_line(summary, episode=episode)}")
    for name, bias_ms in sorted(summary["cross_modality_bias_ms"].items()):
        if abs(bias_ms) >= 0.005:
            emit(f"SYNC bias_vs_arm_ms[{name}]={bias_ms:.2f}")
    if summary["status"] != "pass":
        # Warning, never "ERROR:", so a skew violation does not make the gateway mark the
        # whole recorder session failed and tear the process down mid-session.
        emit(
            f"SYNC WARN: episode {episode} alignment out of budget "
            f"({summary['within_group_skew_over_budget_frames']} group-skew / "
            f"{summary['residual_skew_over_budget_frames'] or 0} residual / "
            f"{summary['global_lag_over_tolerance_frames']} grid-lag frames "
            f"of {summary['frames']})"
        )
        # Each budgeted failure in its own words: the counts above say how many frames, not
        # which measurement went out, and the operator has to fix the latter.
        for failure in summary["failures"]:
            emit(f"SYNC WARN: episode {episode}: {failure}")
        measured = float(summary["measured_frame_interval_ms"])
        nominal = float(summary["nominal_frame_interval_ms"])
        if summary["global_lag_over_tolerance_frames"] and nominal > 0 and measured > nominal * 1.05:
            # The dataset labels frames as evenly spaced at 1/fps; if the loop actually ran
            # slower, the recorded cadence is a fiction and needs fixing at capture time.
            emit(
                f"SYNC WARN: control loop delivered {measured:.1f} ms between frames but the "
                f"dataset labels them {nominal:.1f} ms apart -- lower dataset.fps or reduce "
                "per-frame work; do not train on this cadence as-is"
            )


def _write_dataset_sync_report(
    dataset_root: Path,
    runtime_args: argparse.Namespace,
    *,
    residual_tolerance_ms: float | None,
) -> None:
    """Persist the file-based audit once the parquet files are closed by ``finalize()``."""
    if runtime_args.no_sync_audit:
        return
    try:
        report, destination = write_fr3_sync_report(
            dataset_root,
            tolerance_ms=float(runtime_args.sync_tolerance_ms),
            global_lag_tolerance_ms=float(runtime_args.sync_global_lag_tolerance_ms),
            residual_tolerance_ms=residual_tolerance_ms,
            bias_tolerance_ms=float(runtime_args.sync_bias_tolerance_ms),
        )
    except Exception as exc:  # noqa: BLE001 - a failed audit must not fail the session
        emit(f"SYNC audit unavailable: {exc}")
        return
    emit(f"SYNC {format_sync_summary_line(report)}")
    emit(f"SYNC report={destination}")
    for failure in report["failures"]:
        emit(f"SYNC WARN: {failure}")


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    runtime_args = _RUNTIME_ARGS or parse_runtime_args([])[0]
    if cfg.teleop is None:
        raise ValueError("FR3 GUI recording requires a teleoperator configuration.")
    if cfg.policy is not None:
        raise ValueError("FR3 GUI recording supports teleop recording only, not policy evaluation.")

    init_logging()
    logging.getLogger().setLevel(logging.WARNING)

    backend = str(runtime_args.backend)
    robot, robot_cfg = _build_robot(cfg, runtime_args)
    teleop = make_teleoperator_from_config(cfg.teleop)
    # The ee2ee processors read the workspace envelope off cfg.robot. Point cfg at whichever
    # backend config is actually in use so a sim run is clamped exactly like the hardware run.
    # (Rebinding in place rather than dataclasses.replace: RecordConfig.__post_init__ re-reads
    # draccus CLI state, which is not valid to do a second time outside the parser context.)
    cfg.robot = robot_cfg
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_fr3_action_processors(cfg)
    )

    dataset_features = build_fr3_ee2ee_dataset_features(
        robot=robot,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
        use_videos=cfg.dataset.video,
    )

    dataset_root = _session_dataset_root(cfg, backend)
    repo_id = f"{cfg.dataset.repo_id}_sim" if backend == "sim" else cfg.dataset.repo_id
    camera_count = max(len(getattr(robot, "cameras", {}) or getattr(robot_cfg, "camera_names", ())), 1)

    # A stamped root is normally new, so this normally creates. It still has to handle an
    # existing one: `cfg.resume` and an already-stamped root both name a specific dataset, and
    # then a second session must extend it rather than die on LeRobotDataset.create()'s
    # exist_ok=False mkdir -- a raw FileExistsError traceback in the operator's face.
    existing_dataset = _assert_resumable_or_absent(dataset_root)
    if existing_dataset:
        dataset = LeRobotDataset(
            repo_id,
            root=dataset_root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        # Appending frames whose schema disagrees with the existing episodes would produce a
        # dataset that only fails at training time. Refuse now, with the mismatch named.
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        dataset.start_image_writer(
            num_processes=cfg.dataset.num_image_writer_processes,
            num_threads=cfg.dataset.num_image_writer_threads_per_camera * camera_count,
        )
        emit(f"Resuming dataset with {dataset.num_episodes} existing episode(s)")
    elif cfg.resume:
        raise RuntimeError(f"resume was requested but {dataset_root} does not contain meta/info.json.")
    else:
        sanity_check_dataset_name(repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            repo_id,
            cfg.dataset.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * camera_count,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    commands = _CommandChannel(events)
    control_fps = int(cfg.control_fps or cfg.dataset.fps)
    saved_episodes = 0
    capture_timestamp_names = tuple(getattr(robot, "capture_timestamp_feature_names", ()))
    # Declared, not guessed: the two backends put different meanings behind these timestamps
    # (hardware sensor/host reads vs. one shared physics instant), and the audit must say which.
    clock_semantics = "sim_extraction_wallclock" if backend == "sim" else "hardware_mixed"
    # Derived once per session: the budget depends on the rig's sensor rate, and both the live
    # per-episode verdict and the persisted report have to be judged against the same one.
    residual_tolerance_ms = runtime_args.sync_residual_tolerance_ms
    if residual_tolerance_ms is None:
        residual_tolerance_ms = residual_tolerance_for_camera_fps(
            _slowest_camera_fps(cfg),
            within_group_tolerance_ms=float(runtime_args.sync_tolerance_ms),
        )

    try:
        # Progress lines around each connect, because these are the calls that can block for a
        # long time on hardware: the arm waits on libfranka, and each RealSense opens a USB
        # pipeline. Every other emit happens after the whole connect sequence, so without these
        # a stall here reaches the operator as the gateway's own spawn message and nothing else
        # -- indistinguishable from a recorder that died before its first line.
        emit(f"Connecting {robot.name} ({backend})")
        robot.connect()
        emit(f"Connected {robot.name}")
        emit(f"Connecting teleoperator {cfg.teleop.type}")
        teleop.connect()
        emit(f"Connected teleoperator {cfg.teleop.type}")
        sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
        if callable(sync_gripper):
            observation = robot.get_observation(include_cameras=False)
            sync_gripper(float(observation["gripper.pos"]))
        _reset_gripper_to_open(robot, teleop)
        emit("Gripper opened")

        commands.start()
        emit(f"Backend: {backend}")
        emit(f"Dataset root: {dataset.root}")
        emit(f"Robot model: {robot.name}")
        # Device roster lines use the gateway's device-row ids so the Device Manager marks
        # exactly what came up. A sim session opens no camera, arm or gripper hardware, so it
        # claims none of them -- only the SpaceMouse is real in both backends.
        emit(f"Teleoperators: {cfg.teleop.type}")
        if backend != "sim":
            # Ids match the gateway's seeded device rows (`fr3`, `pika`, and the config's
            # camera keys). Sim stays silent here so the gateway can leave those rows idle
            # rather than flagging absent hardware as failed.
            emit(f"Cameras: {', '.join(getattr(robot, 'cameras', {}) or ())}")
            emit("Robots: fr3")
            emit("Grippers: pika")

        with VideoEncodingManager(dataset):
            while not events["stop_recording"] and saved_episodes < cfg.dataset.num_episodes:
                if cfg.auto_move_to_start_after_episode:
                    move_to_start = getattr(robot, "move_to_start", None)
                    if callable(move_to_start):
                        move_to_start()
                    _reset_gripper_to_open(robot, teleop)
                    emit("Gripper opened")

                emit(f"Episode {dataset.num_episodes} ready")
                command = commands.wait_for_command()
                if command is None:
                    continue
                if command in ("q", "quit", "exit", "stop"):
                    break
                if command not in ("", "start"):
                    # Save/discard with no episode in flight: nothing to act on, re-arm.
                    continue

                events["exit_early"] = False
                events["rerecord_episode"] = False
                reset_origin = getattr(robot, "reset_capture_timestamp_origin", None)
                if callable(reset_origin):
                    reset_origin()

                with _ProgressReporter(dataset):
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=control_fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        dataset=dataset,
                        policy=None,
                        preprocessor=None,
                        postprocessor=None,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=False,
                        display_compressed_images=False,
                    )

                # Whatever ended the episode (timer, save, discard, quit) is the last command
                # seen; the timer case leaves nothing queued and defaults to keeping the take.
                decision = commands.drain_latest()
                keep_episode = decision not in ("n", "discard", "q", "quit", "exit", "stop")

                # An episode stopped before any frame landed has nothing to save; save_episode()
                # would raise and take the whole session down with it.
                buffer = getattr(dataset, "episode_buffer", None)
                buffered_frames = int(buffer.get("size", 0)) if isinstance(buffer, dict) else 0
                if keep_episode and buffered_frames == 0:
                    keep_episode = False
                    emit("WARN: episode ended with 0 frames; nothing to save")

                if keep_episode:
                    # Audit before save_episode(): saving clears the buffer these numbers
                    # come from, and the parquet is not readable until finalize().
                    _audit_episode_buffer(
                        dataset,
                        device_names=capture_timestamp_names,
                        clock_semantics=clock_semantics,
                        episode=dataset.num_episodes,
                        runtime_args=runtime_args,
                        residual_tolerance_ms=residual_tolerance_ms,
                    )
                    # parallel_encoding=False on purpose: the multi-camera path forks a
                    # ProcessPoolExecutor, and this process already holds the MuJoCo/EGL
                    # context, camera driver threads and the stdin reader. Forking that
                    # deadlocks (reproduced: save_episode never returns with 2 cameras and
                    # streaming_encoding off). Sequential encoding costs a few hundred ms
                    # between episodes; a wedged recorder costs the session.
                    dataset.save_episode(parallel_encoding=False)
                    saved_episodes += 1
                    emit("Episode saved")
                    emit(f"Total saved episodes: {dataset.num_episodes}")
                else:
                    dataset.clear_episode_buffer()
                    emit("Episode discarded")

                teleop_action_processor.reset()
                robot_action_processor.reset()
                robot_observation_processor.reset()
    finally:
        emit("Recording stopped")
        finalized = False
        try:
            dataset.finalize()
            finalized = True
        except Exception as exc:  # noqa: BLE001 - report, but still release the hardware below
            emit(f"ERROR: dataset finalize failed: {exc}")
        if finalized and dataset.num_episodes:
            _write_dataset_sync_report(
                Path(dataset.root), runtime_args, residual_tolerance_ms=residual_tolerance_ms
            )
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        emit(f"Total saved episodes: {dataset.num_episodes}")
        emit("Recorder exited")

    return dataset


def main(argv: list[str] | None = None) -> None:
    runtime_args, remaining = parse_runtime_args(argv)
    global _RUNTIME_ARGS
    _RUNTIME_ARGS = runtime_args
    sys.argv = [sys.argv[0], *remaining]
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
