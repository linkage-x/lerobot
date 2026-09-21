#!/usr/bin/env python3
"""Interactive single-tag camera calibration in the current FR3 base frame.

This process is intentionally independent of the data-collection GUI.  It is
started from the host through ``run_p0_two_marker_calibration.sh`` but runs on
Thor, where both the Argus cameras and FR3 are connected.  The live viewer
reads the recorder-owned synchronized frame bus, so it never opens a second
camera session.

Controls in the operator window:
    Enter  capture tag-6 corners + the measured FR3 pose
    q      finish and solve (only after the minimum observation gate is ready)
    f      force a solve attempt before the readiness gate
    Esc    abort without replacing the active calibration
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

# Keep native math/image libraries from silently creating one worker per Thor
# CPU.  The operator can still override these before launching the program.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.robots import franka_research3, make_robot_from_config  # noqa: E402
from lerobot.robots.franka_research3 import FrankaResearch3Config  # noqa: E402
from third_party.opencv_kalibr.fr3_calibration.execute_pose_and_capture import (  # noqa: E402
    JOINT_VECTOR_NAMES,
)
from third_party.opencv_kalibr.fr3_calibration.execute_pose_and_capture_thor_gmsl2 import (  # noqa: E402
    ThorRecorderClient,
)
from tools.thor.gmsl2.online_sync_frame_client import (  # noqa: E402
    OnlineSyncCluster,
    ThorOnlineSyncFrameClient,
)
from tools.thor.p1_eye_hand_core import matrix_payload  # noqa: E402
from tools.thor.p1_simple_eye_hand_calibration import (  # noqa: E402
    _sha256,
    _write_json,
)
from tools.thor.single_tag_camera_calibration import (  # noqa: E402
    calibrate_and_write,
    load_existing_fisheye_intrinsics,
)

DEFAULT_RECORDER_CONFIG = (
    REPO_ROOT / "third_party/opencv_kalibr/fr3_calibration/host/thor_gmsl2_calibration.yaml"
)
DEFAULT_URDF = (
    REPO_ROOT
    / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
)
DEFAULT_ROOT = REPO_ROOT / "outputs/calibration/p0_single_tag_camera_calibration"
DEFAULT_INTRINSICS = Path(
    "/home/nvidia/lerobot/outputs/calibration/"
    "thor_gmsl2_selfcal_0804_fisheye_intrinsics/summary.json"
)
WINDOW_NAME = "FR3-base camera calibration: tag36h11 ID 6, 160 mm"
TAG_IDS = (6,)
TAG_SIZE_M = 0.16
MIN_EXTRINSIC_FRAMES = 20
MIN_CAMERAS = 2
CAPTURES_SCHEMA = "fr3_base_single_tag_camera_captures/v1"
RUN_MANIFEST_SCHEMA = "fr3_base_single_tag_camera_run/v1"
# cam_02 is the UMI camera and is intentionally outside this fixed-camera
# calibration. Keep this policy local to the standalone workflow;
# normal Thor recording still discovers and uses it.
DEFAULT_EXCLUDED_SENSOR_IDS = frozenset({2})


@dataclass
class CameraResult:
    camera: str
    calibration_camera: str
    image_bgr: np.ndarray
    annotated_bgr: np.ndarray
    detections: list[dict[str, Any]]
    error: str = ""


@dataclass(frozen=True)
class CpuIsolationPlan:
    enabled: bool
    vision_cpus: tuple[int, ...]
    robot_control_cpu: int | None
    robot_irq_cpu: int | None
    robot_interface: str | None


def _operator_key(keysym: str, char: str) -> int | None:
    if keysym in {"Return", "KP_Enter"}:
        return 13
    if keysym == "Escape":
        return 27
    value = (char or "").lower()
    if value in {"q", "f"}:
        return ord(value)
    return None


def _route_interface(robot_ip: str) -> str | None:
    try:
        completed = subprocess.run(
            ["ip", "route", "get", robot_ip],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    tokens = completed.stdout.split()
    try:
        return tokens[tokens.index("dev") + 1]
    except (ValueError, IndexError):
        return None


def _irq_cpu_from_interrupts(
    interrupts_text: str,
    interface: str,
    allowed_cpus: set[int],
) -> int | None:
    lines = interrupts_text.splitlines()
    if not lines:
        return None
    cpu_columns = [int(token.removeprefix("CPU")) for token in lines[0].split() if token.startswith("CPU")]
    totals = {cpu: 0 for cpu in cpu_columns if cpu in allowed_cpus}
    for line in lines[1:]:
        if interface not in line or ":" not in line:
            continue
        fields = line.split(":", 1)[1].split()
        for index, cpu in enumerate(cpu_columns):
            if cpu not in totals or index >= len(fields):
                continue
            try:
                totals[cpu] += int(fields[index])
            except ValueError:
                break
    if not totals or max(totals.values(), default=0) <= 0:
        return None
    return max(totals, key=totals.get)


def _build_cpu_isolation_plan(
    allowed_cpus: set[int],
    *,
    interface: str | None,
    irq_cpu: int | None,
    requested_control_cpu: int | None,
    disabled: bool,
) -> CpuIsolationPlan:
    if not allowed_cpus:
        raise RuntimeError("process has no allowed CPUs")
    ordered = sorted(allowed_cpus)
    if disabled or len(ordered) < 2:
        return CpuIsolationPlan(False, tuple(ordered), None, irq_cpu, interface)
    if requested_control_cpu is not None:
        if requested_control_cpu not in allowed_cpus:
            raise ValueError(
                f"--robot-control-cpu {requested_control_cpu} is outside allowed CPUs {ordered}"
            )
        control_cpu = requested_control_cpu
    elif irq_cpu in allowed_cpus and len(ordered) >= 3:
        irq_index = ordered.index(int(irq_cpu))
        control_cpu = ordered[(irq_index + 1) % len(ordered)]
    else:
        control_cpu = ordered[-1]
    reserved = {control_cpu}
    if irq_cpu in allowed_cpus and irq_cpu != control_cpu and len(ordered) >= 3:
        reserved.add(int(irq_cpu))
    vision_cpus = tuple(cpu for cpu in ordered if cpu not in reserved)
    if not vision_cpus:
        vision_cpus = tuple(cpu for cpu in ordered if cpu != control_cpu)
    return CpuIsolationPlan(True, vision_cpus, control_cpu, irq_cpu, interface)


def _configure_cpu_isolation(
    robot_ip: str,
    requested_control_cpu: int | None,
    disabled: bool,
) -> CpuIsolationPlan:
    allowed = set(os.sched_getaffinity(0))
    interface = _route_interface(robot_ip)
    irq_cpu = None
    if interface is not None:
        try:
            irq_cpu = _irq_cpu_from_interrupts(
                Path("/proc/interrupts").read_text(encoding="utf-8"),
                interface,
                allowed,
            )
        except OSError:
            pass
    plan = _build_cpu_isolation_plan(
        allowed,
        interface=interface,
        irq_cpu=irq_cpu,
        requested_control_cpu=requested_control_cpu,
        disabled=disabled,
    )
    if plan.enabled:
        os.sched_setaffinity(0, set(plan.vision_cpus))
        print(
            "[CPU] isolated runtime: "
            f"vision/Argus={list(plan.vision_cpus)} "
            f"robot_control={plan.robot_control_cpu} "
            f"{plan.robot_interface or 'robot_nic'}_irq={plan.robot_irq_cpu}",
            flush=True,
        )
    else:
        print(f"[CPU] isolation disabled; allowed={list(plan.vision_cpus)}", flush=True)
    return plan


class OperatorWindow:
    """Tk viewer used because Thor's FR3 venv ships headless OpenCV."""

    def __init__(self, title: str, width: int = 1440, height: int = 900):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError("operator window requires tkinter and Pillow") from exc
        try:
            self._root = tk.Tk()
        except Exception as exc:
            raise RuntimeError(
                "failed to create operator window; verify host X server and ssh -Y forwarding"
            ) from exc
        self._tk = tk
        self._image_module = Image
        self._image_tk_module = ImageTk
        self._keys: deque[int] = deque()
        self._photo: Any | None = None
        self._closed = False
        self._root.title(title)
        self._root.geometry(f"{width}x{height}")
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill=tk.BOTH, expand=True)
        self._root.bind("<KeyPress>", self._on_key)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._pump()

    def _on_key(self, event: Any) -> None:
        key = _operator_key(str(event.keysym), str(event.char))
        if key is not None:
            self._keys.append(key)

    def _on_close(self) -> None:
        self._keys.append(27)

    def _pump(self) -> None:
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
            self._keys.append(27)

    def show(self, image_bgr: np.ndarray) -> None:
        if self._closed:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._photo = self._image_tk_module.PhotoImage(
            image=self._image_module.fromarray(image_rgb)
        )
        self._label.configure(image=self._photo)
        self._pump()

    def poll_key(self) -> int:
        self._pump()
        return self._keys.popleft() if self._keys else -1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _matrix_from_payload(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _resolve_resume_run(root: Path, requested: str | None) -> Path | None:
    if requested is None:
        return None
    if requested.strip().lower() == "latest":
        candidates = sorted(
            (
                path.parent
                for path in root.glob("manual_run_*/captures.json")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no resumable manual_run_*/captures.json exists under {root}")
        return candidates[0].resolve()
    path = Path(requested).expanduser()
    if not path.is_absolute():
        direct = path.resolve()
        under_root = (root / path).resolve()
        path = direct if direct.is_dir() else under_root
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"resume run directory does not exist: {path}")
    return path


def _load_resume_records(run_dir: Path) -> list[dict[str, Any]]:
    captures_path = run_dir / "captures.json"
    if not captures_path.is_file():
        raise FileNotFoundError(f"resume run has no captures.json: {captures_path}")
    payload = json.loads(captures_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CAPTURES_SCHEMA:
        raise ValueError(f"unsupported captures schema in {captures_path}: {payload.get('schema')!r}")
    marker = payload.get("marker", {})
    if (
        str(marker.get("family")) != "tag36h11"
        or int(marker.get("id", -1)) != 6
        or abs(float(marker.get("marker_size_m", 0.0)) - TAG_SIZE_M) > 1e-9
    ):
        raise ValueError(f"resume marker does not match tag36h11 ID 6, {TAG_SIZE_M} m")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"resume run contains no committed captures: {captures_path}")
    seen_indices: set[int] = set()
    run_root = run_dir.resolve()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"capture record {position} is not an object")
        capture_index = int(record.get("capture_index", -1))
        if capture_index < 0 or capture_index in seen_indices:
            raise ValueError(f"invalid or duplicate capture_index={capture_index}")
        seen_indices.add(capture_index)
        _matrix_from_payload(record.get("T_base_tcp"), f"record {capture_index} T_base_tcp")
        joints = np.asarray(record.get("joint_values_rad", []), dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"record {capture_index} must contain seven finite joint values")
        cameras = record.get("cameras")
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError(f"record {capture_index} contains no cameras")
        for camera, item in cameras.items():
            if not isinstance(item, dict):
                raise ValueError(f"record {capture_index} camera {camera} is invalid")
            image_path = Path(str(item.get("image", ""))).expanduser()
            if not image_path.is_absolute():
                image_path = (REPO_ROOT / image_path).resolve()
            else:
                image_path = image_path.resolve()
            if not image_path.is_relative_to(run_root):
                raise ValueError(f"record {capture_index} image escapes run directory: {image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"record {capture_index} image is missing: {image_path}")
    return records


def _calibrated_camera_identities(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("calibration_camera", live_camera))
        for record in records
        for live_camera, item in record.get("cameras", {}).items()
    }


def _resume_counts(
    records: list[dict[str, Any]],
    resolved_cameras: dict[str, str],
) -> dict[str, int]:
    historical = _calibrated_camera_identities(records)
    current = set(resolved_cameras.values())
    if historical != current:
        raise ValueError(
            "resume camera identities differ from current cameras: "
            f"saved_only={sorted(historical - current)}, current_only={sorted(current - historical)}"
        )
    counts = {live_camera: 0 for live_camera in resolved_cameras}
    for record in records:
        valid_identities = {
            str(item.get("calibration_camera", live_camera))
            for live_camera, item in record.get("cameras", {}).items()
            if bool(item.get("valid"))
            and any(int(det.get("tag_id", -1)) == 6 for det in item.get("detections", []))
        }
        for live_camera, calibrated_camera in resolved_cameras.items():
            if calibrated_camera in valid_identities:
                counts[live_camera] += 1
    return counts


def _write_or_validate_run_manifest(
    run_dir: Path,
    intrinsics_summary: Path,
    resolved_cameras: dict[str, str],
) -> None:
    path = run_dir / "run_manifest.json"
    intrinsics_sha256 = _sha256(intrinsics_summary)
    identities = sorted(set(resolved_cameras.values()))
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != RUN_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported run manifest schema: {payload.get('schema')!r}")
        if payload.get("intrinsics_summary_sha256") != intrinsics_sha256:
            raise ValueError("resume intrinsics summary differs from the original run")
        if sorted(payload.get("calibrated_camera_identities", [])) != identities:
            raise ValueError("resume camera identities differ from run_manifest.json")
    else:
        payload = {
            "schema": RUN_MANIFEST_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "world_frame_id": "fr3_base",
            "marker": {"family": "tag36h11", "id": 6, "marker_size_m": TAG_SIZE_M},
            "intrinsics_summary": str(intrinsics_summary),
            "intrinsics_summary_sha256": intrinsics_sha256,
            "calibrated_camera_identities": identities,
        }
    sessions = payload.setdefault("sessions", [])
    sessions.append(
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "live_to_calibrated": dict(sorted(resolved_cameras.items())),
        }
    )
    _write_json(path, payload)


def _load_active(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    active_path = root / "active.json"
    if not active_path.is_file():
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    calibration_path = Path(str(active.get("calibration_path", "")))
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Active calibration references missing files: {active_path}")
    if active.get("calibration_sha256") and _sha256(calibration_path) != active["calibration_sha256"]:
        raise ValueError(f"Active calibration SHA-256 mismatch: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise ValueError(f"Active calibration is not passing: {calibration_path}")
    return calibration_path, calibration, active


def _choose_existing(mode: str, active: tuple[Path, dict[str, Any], dict[str, Any]] | None) -> bool:
    """Return True when the existing calibration should be reused and execution should end."""
    if active is None:
        if mode == "reuse":
            raise FileNotFoundError("--existing reuse requested, but no active calibration exists")
        return False
    path, calibration, _ = active
    cameras = calibration.get("joint_solution", {}).get("cameras", {})
    print(
        f"[EXISTING] {path}\n"
        f"  created={calibration.get('created_utc', 'unknown')} "
        f"frame={calibration.get('world', {}).get('world_frame_id', '?')} cameras={len(cameras)}",
        flush=True,
    )
    if mode == "reuse":
        return True
    if mode == "recalibrate":
        return False
    if not sys.stdin.isatty():
        raise RuntimeError("An active calibration exists; use --existing reuse or --existing recalibrate")
    while True:
        answer = input("Use existing calibration and exit? [Y=reuse / r=recalibrate]: ").strip().lower()
        if answer in {"", "y", "yes", "u", "use", "reuse"}:
            return True
        if answer in {"r", "redo", "recalibrate"}:
            return False


def _activate_teaching_mode(robot: Any) -> None:
    arm = getattr(robot, "_arm", None)
    if arm is None:
        raise RuntimeError("FR3 arm backend is unavailable")
    stop_otg = getattr(robot, "_stop_otg_loop", None)
    if callable(stop_otg):
        stop_otg()
    enter_teaching_mode = getattr(arm, "enter_teaching_mode", None)
    if not callable(enter_teaching_mode):
        raise RuntimeError("FR3 arm backend does not provide native teaching mode")
    enter_teaching_mode([0.0] * 7)
    print("[ROBOT] panda_py native teaching mode active: joint damping = 0", flush=True)


def _run_robot_only_control_test(robot: Any, duration_s: float) -> None:
    """Exercise the same teaching controller without cameras, detection, or UI."""
    arm = getattr(robot, "_arm", None)
    panda = getattr(arm, "_robot", None)
    if panda is None:
        raise RuntimeError("FR3 panda backend is unavailable for control test")
    rates: list[float] = []
    deadline = time.monotonic() + duration_s
    next_report = time.monotonic()
    while time.monotonic() < deadline:
        # panda_py controllers run asynchronously. The installed high-level
        # Panda API exposes async failures through raise_error(); unlike the
        # lower-level native wrapper, it has no control_thread_active().
        panda.raise_error()
        now = time.monotonic()
        if now >= next_report:
            state = panda.get_state()
            rate = float(state.control_command_success_rate)
            if np.isfinite(rate):
                rates.append(rate)
            mode = getattr(getattr(state, "robot_mode", None), "name", "unknown")
            print(f"[ROBOT-ONLY TEST] mode={mode} success_rate={rate:.6f}", flush=True)
            next_report = now + 1.0
        time.sleep(0.02)
    panda.raise_error()
    if not rates:
        raise RuntimeError("FR3 control test returned no finite command-success samples")
    print(
        "[ROBOT-ONLY TEST] passed "
        f"duration={duration_s:.1f}s min={min(rates):.6f} "
        f"mean={float(np.mean(rates)):.6f} final={rates[-1]:.6f}",
        flush=True,
    )


def _make_robot(robot_ip: str, urdf: Path) -> Any:
    if not urdf.is_file():
        raise FileNotFoundError(f"FR3 URDF not found: {urdf}")
    _ = franka_research3
    cfg = FrankaResearch3Config(
        id="p0_two_marker_calibration",
        robot_ip=robot_ip,
        gripper_backend="mock",
        allow_mock_gripper=True,
        urdf_path=str(urdf),
        target_frame_name="fr3_ee",
        cameras={},
        use_otg=False,
        stiffness=[0.0] * 7,
        damping=[0.0] * 7,
        arm_start_controller_on_connect=False,
        arm_state_poll_frequency_hz=0.0,
    )
    return make_robot_from_config(cfg)


def _robot_teaching_worker_main(
    connection: Any,
    robot_ip: str,
    urdf_text: str,
    control_cpu: int | None,
) -> None:
    """Own panda_py in a CPU-isolated process and serve state reads over a pipe."""
    robot: Any | None = None
    try:
        if control_cpu is not None:
            os.sched_setaffinity(0, {control_cpu})
        robot = _make_robot(robot_ip, Path(urdf_text))
        robot.connect()
        _activate_teaching_mode(robot)
        arm = getattr(robot, "_arm", None)
        panda = getattr(arm, "_robot", None)
        if panda is None:
            raise RuntimeError("FR3 panda backend is unavailable in teaching worker")
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "affinity": sorted(os.sched_getaffinity(0)),
            }
        )
        running = True
        while running:
            panda.raise_error()
            if not connection.poll(0.02):
                continue
            message = connection.recv()
            command = message.get("command")
            if command == "observation":
                observation = robot.get_observation(include_cameras=False)
                connection.send({"kind": "observation", "value": dict(observation)})
            elif command == "status":
                state = panda.get_state()
                connection.send(
                    {
                        "kind": "status",
                        "mode": getattr(getattr(state, "robot_mode", None), "name", "unknown"),
                        "success_rate": float(state.control_command_success_rate),
                    }
                )
            elif command == "stop":
                running = False
            else:
                raise RuntimeError(f"unknown robot worker command: {command!r}")
    except BaseException as exc:
        try:
            connection.send(
                {
                    "kind": "fatal",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
            except Exception:
                pass
        connection.close()


class RobotTeachingWorker:
    """Parent-side proxy for the isolated FR3 teaching process."""

    def __init__(self, robot_ip: str, urdf: Path, control_cpu: int | None) -> None:
        self.robot_ip = robot_ip
        self.urdf = urdf
        self.control_cpu = control_cpu
        self._process: Any | None = None
        self._connection: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self, timeout_s: float = 30.0) -> None:
        if self._process is not None:
            raise RuntimeError("robot teaching worker is already started")
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_robot_teaching_worker_main,
            args=(child_connection, self.robot_ip, str(self.urdf), self.control_cpu),
            name="FR3TeachingWorker",
        )
        process.start()
        child_connection.close()
        self._process = process
        self._connection = parent_connection
        message = self._receive(timeout_s)
        if message.get("kind") != "ready":
            self.stop()
            self._raise_message(message)
        print(
            f"[ROBOT] isolated teaching worker pid={message['pid']} "
            f"affinity={message['affinity']}",
            flush=True,
        )

    def _receive(self, timeout_s: float) -> dict[str, Any]:
        if self._connection is None or self._process is None:
            raise RuntimeError("robot teaching worker is not started")
        if not self._connection.poll(timeout_s):
            exitcode = self._process.exitcode
            raise TimeoutError(
                f"robot teaching worker did not respond within {timeout_s:.1f}s "
                f"(exitcode={exitcode})"
            )
        try:
            message = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"robot teaching worker closed unexpectedly (exitcode={self._process.exitcode})"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"invalid robot teaching worker response: {message!r}")
        return message

    @staticmethod
    def _raise_message(message: dict[str, Any]) -> None:
        if message.get("kind") == "fatal":
            detail = str(message.get("traceback") or message.get("error") or "unknown error")
            raise RuntimeError(f"FR3 teaching worker failed:\n{detail}")
        raise RuntimeError(f"unexpected FR3 teaching worker response: {message!r}")

    def _request(self, command: str, expected_kind: str, timeout_s: float = 5.0) -> dict[str, Any]:
        self.raise_if_failed()
        assert self._connection is not None
        self._connection.send({"command": command})
        message = self._receive(timeout_s)
        if message.get("kind") != expected_kind:
            self._raise_message(message)
        return message

    def raise_if_failed(self) -> None:
        if self._process is None or self._connection is None:
            raise RuntimeError("robot teaching worker is not started")
        if self._connection.poll(0.0):
            self._raise_message(self._receive(0.0))
        if not self._process.is_alive():
            raise RuntimeError(
                f"FR3 teaching worker exited unexpectedly (exitcode={self._process.exitcode})"
            )

    def get_observation(self, *, include_cameras: bool = False) -> dict[str, Any]:
        if include_cameras:
            raise ValueError("isolated teaching worker does not own cameras")
        return dict(self._request("observation", "observation")["value"])

    def get_status(self) -> tuple[str, float]:
        message = self._request("status", "status")
        return str(message["mode"]), float(message["success_rate"])

    def stop(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if process is None:
            return
        if process.is_alive() and connection is not None:
            try:
                connection.send({"command": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        if connection is not None:
            connection.close()


def _run_isolated_robot_only_control_test(
    worker: RobotTeachingWorker,
    duration_s: float,
) -> None:
    rates: list[float] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        mode, rate = worker.get_status()
        if np.isfinite(rate):
            rates.append(rate)
        print(f"[ROBOT-ONLY TEST] mode={mode} success_rate={rate:.6f}", flush=True)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    worker.raise_if_failed()
    if not rates:
        raise RuntimeError("FR3 control test returned no finite command-success samples")
    print(
        "[ROBOT-ONLY TEST] passed "
        f"duration={duration_s:.1f}s min={min(rates):.6f} "
        f"mean={float(np.mean(rates)):.6f} final={rates[-1]:.6f}",
        flush=True,
    )


def _make_recorder_config(
    template: Path,
    run_dir: Path,
    frame_bus_dir: Path,
    every_n: int,
    sensor_ids: list[int] | None = None,
) -> Path:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Recorder config root must be a mapping: {template}")
    cameras = payload.setdefault("sensors", {}).setdefault("cameras", {})
    # GMSL2 logical ids can change after links are unplugged/replugged.  Always
    # discover the currently locked links; calibration identity is handled
    # separately through --camera-alias rather than by freezing device ids.
    cameras["detect_all"] = sensor_ids is None
    cameras["sensor_ids"] = [] if sensor_ids is None else sensor_ids
    cameras.setdefault("defaults", {})["recorder_backend"] = "argus_online_sync"
    online = cameras.setdefault("online_sync", {})
    online["enabled"] = True
    online["frame_bus_dir"] = str(frame_bus_dir)
    online["frame_bus_every_n"] = max(1, int(every_n))
    payload.setdefault("box_collection", {})["enabled"] = False
    dataset = payload.setdefault("dataset", {})
    dataset["root"] = str(run_dir / "unused_recorder_dataset")
    dataset["num_episodes"] = 0
    dataset["single_task"] = "current-FR3-base camera extrinsics from tag36h11 ID 6"
    config_dir = Path(tempfile.mkdtemp(prefix="p0_two_marker_cfg_", dir="/tmp"))
    path = config_dir / "recorder.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _start_recorder(config_path: Path, *, skip_hardware_sync: bool) -> ThorRecorderClient:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/thor/gmsl2/thor_record.py"),
        "--config-path",
        str(config_path),
        "--repo-root",
        str(REPO_ROOT),
        "--no-box",
        "--skip-argus-probe",
    ]
    if skip_hardware_sync:
        cmd.append("--skip-hardware-sync")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    recorder = ThorRecorderClient(cmd, cwd=REPO_ROOT, env=env)
    print(f"[THOR] command: {' '.join(cmd)}", flush=True)
    recorder.start(wait_ready_timeout_s=240.0)
    return recorder


def _new_detector() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return (dictionary, parameters)


def _normalize_camera_name(raw: str) -> str:
    value = str(raw).strip()
    if value.isdigit():
        return f"cam_{int(value):02d}"
    if value.lower().startswith("cam_") and value[4:].isdigit():
        return f"cam_{int(value[4:]):02d}"
    if not value:
        raise ValueError("camera name must not be empty")
    return value


def _camera_id(raw: str) -> int:
    camera = _normalize_camera_name(raw)
    if not camera.startswith("cam_") or not camera[4:].isdigit():
        raise ValueError(f"camera must be a numeric id such as cam_02, got {raw!r}")
    return int(camera[4:])


def _excluded_sensor_ids(extra_cameras: list[str]) -> list[int]:
    return sorted(DEFAULT_EXCLUDED_SENSOR_IDS | {_camera_id(value) for value in extra_cameras})


def _select_sensor_ids(
    locked_sensor_ids: list[int],
    excluded_sensor_ids: list[int],
) -> tuple[list[int], dict[str, str]]:
    excluded = set(excluded_sensor_ids)
    selected: list[int] = []
    ignored: dict[str, str] = {}
    for sensor_id in locked_sensor_ids:
        current = f"cam_{sensor_id:02d}"
        if sensor_id in excluded:
            ignored[current] = "excluded by P0 policy"
            continue
        selected.append(sensor_id)
    return selected, ignored


def _locked_sensor_ids(
    repo_root: Path = REPO_ROOT,
    *,
    _runner: Any = subprocess.run,
) -> list[int]:
    script = repo_root / "tools/thor/gmsl2/check_max96726_locks.sh"
    result = _runner([str(script)], capture_output=True, text=True, timeout=30)
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"MAX96726 lock check failed rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("LOCKED_VIDEO_IDS="):
            value = line.split("=", 1)[1].strip()
            return [int(item) for item in value.split(",") if item.strip()]
    raise RuntimeError("MAX96726 lock check did not emit LOCKED_VIDEO_IDS=")


def _parse_camera_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"camera alias must be CURRENT=CALIBRATED, got {raw!r}")
        current_raw, calibrated_raw = raw.split("=", 1)
        current = _normalize_camera_name(current_raw)
        calibrated = _normalize_camera_name(calibrated_raw)
        if current in aliases and aliases[current] != calibrated:
            raise ValueError(f"camera {current} has conflicting aliases")
        aliases[current] = calibrated
    targets = list(aliases.values())
    if len(set(targets)) != len(targets):
        raise ValueError("two current cameras cannot alias the same calibrated camera")
    return aliases


def _detect_camera(
    camera: str,
    calibration_camera: str,
    image_bgr: np.ndarray,
    detection_scale: float,
) -> CameraResult:
    if 0.0 < detection_scale < 1.0:
        work = cv2.resize(image_bgr, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    else:
        work = image_bgr.copy()
    annotated = work.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    detector = _new_detector()
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    detections: list[dict[str, Any]] = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for marker_corners, marker_id_raw in zip(corners, ids.reshape(-1), strict=True):
            marker_id = int(marker_id_raw)
            if marker_id not in TAG_IDS:
                continue
            corners_px = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            if 0.0 < detection_scale < 1.0:
                corners_px = corners_px / detection_scale
            full_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            refined = corners_px.astype(np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(
                full_gray,
                refined,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
            )
            corners_px = refined.reshape(4, 2).astype(np.float64)
            detections.append(
                {
                    "tag_id": marker_id,
                    "corners_px": corners_px.tolist(),
                    "image_width": int(image_bgr.shape[1]),
                    "image_height": int(image_bgr.shape[0]),
                }
            )
    return CameraResult(camera, calibration_camera, image_bgr, annotated, detections)


def _process_cluster(
    cluster: OnlineSyncCluster,
    pool: ThreadPoolExecutor,
    camera_aliases: dict[str, str],
    detection_scale: float,
) -> dict[str, CameraResult]:
    cameras = sorted(cluster.frames)

    def task(camera: str) -> CameraResult:
        try:
            image_bgr = cv2.cvtColor(cluster.frames[camera].as_rgb(), cv2.COLOR_RGB2BGR)
            calibration_camera = camera_aliases.get(camera, camera)
            return _detect_camera(
                camera,
                calibration_camera,
                image_bgr,
                detection_scale,
            )
        except Exception as exc:  # keep one camera failure visible without killing the viewer
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            return CameraResult(
                camera,
                camera_aliases.get(camera, camera),
                blank,
                blank.copy(),
                [],
                str(exc),
            )

    return {result.camera: result for result in pool.map(task, cameras)}


def _draw_mosaic(
    results: dict[str, CameraResult],
    counts: dict[str, int],
    target: int,
    status_message: str,
) -> np.ndarray:
    tile_w, tile_h = 480, 300
    cameras = sorted(results)
    cols = min(3, max(1, len(cameras)))
    rows = (len(cameras) + cols - 1) // cols
    header_h = 88
    canvas = np.zeros((rows * tile_h + header_h, cols * tile_w, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "ENTER capture | q solve | f force solve | ESC abort",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GREEN=tag 6 detected  RED=no valid tag 6",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_message[:120],
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (100, 230, 255),
        1,
        cv2.LINE_AA,
    )
    for index, camera in enumerate(cameras):
        result = results[camera]
        tile = cv2.resize(result.annotated_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tag_ids = {int(item["tag_id"]) for item in result.detections}
        color = (0, 200, 0) if tag_ids == set(TAG_IDS) else (0, 0, 220)
        y0 = header_h + (index // cols) * tile_h
        x0 = (index % cols) * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + tile_w - 3, y0 + tile_h - 3), color, 5)
        identity = (
            camera
            if result.calibration_camera == camera
            else f"{camera}->{result.calibration_camera}"
        )
        label = f"{identity} tags={sorted(tag_ids)} valid={counts.get(camera, 0)}/{target}"
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + tile_w - 4, y0 + 34), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + 10, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        if result.error:
            cv2.putText(canvas, result.error[:55], (x0 + 10, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 1)
    return canvas


def _robot_sample(robot: Any, settle_s: float, max_joint_delta_rad: float) -> tuple[np.ndarray, list[float]]:
    first = robot.get_observation(include_cameras=False)
    time.sleep(max(0.0, settle_s))
    second = robot.get_observation(include_cameras=False)
    q1 = np.asarray([float(first[name]) for name in JOINT_VECTOR_NAMES])
    q2 = np.asarray([float(second[name]) for name in JOINT_VECTOR_NAMES])
    delta = float(np.max(np.abs(q2 - q1)))
    if delta > max_joint_delta_rad:
        raise RuntimeError(
            f"robot still moving: max joint change {delta:.5f} rad > {max_joint_delta_rad:.5f} rad"
        )
    rotvec = np.asarray([float(second[key]) for key in ("ee.wx", "ee.wy", "ee.wz")])
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T_base_tcp[:3, 3] = [float(second[key]) for key in ("ee.x", "ee.y", "ee.z")]
    return T_base_tcp, q2.tolist()


def _capture_readiness(
    counts: dict[str, int], expected_cameras: list[str], target_per_camera: int
) -> tuple[bool, str]:
    missing = {
        camera: max(0, int(target_per_camera) - int(counts.get(camera, 0)))
        for camera in expected_cameras
        if counts.get(camera, 0) < target_per_camera
    }
    ready = len(expected_cameras) >= MIN_CAMERAS and not missing
    return ready, f"valid per camera={counts}; remaining={missing}"


def _save_capture(
    run_dir: Path,
    records: list[dict[str, Any]],
    results: dict[str, CameraResult],
    cluster: OnlineSyncCluster,
    robot: Any,
    settle_s: float,
    max_joint_delta_rad: float,
) -> dict[str, Any]:
    if not any(result.detections for result in results.values()):
        raise RuntimeError("no valid tag36h11 ID 6 detection in the displayed cluster")
    T_base_tcp, joints = _robot_sample(robot, settle_s, max_joint_delta_rad)
    capture_index = max((int(record["capture_index"]) for record in records), default=-1) + 1
    image_dir = run_dir / "captures" / f"capture_{capture_index:03d}"
    temp_image_dir = image_dir.with_name(image_dir.name + ".tmp")
    shutil.rmtree(temp_image_dir, ignore_errors=True)
    temp_image_dir.mkdir(parents=True, exist_ok=False)
    camera_payload: dict[str, Any] = {}
    try:
        for camera, result in sorted(results.items()):
            image_path = image_dir / f"{camera}.jpg"
            temp_image_path = temp_image_dir / image_path.name
            if not cv2.imwrite(
                str(temp_image_path), result.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"failed to save {image_path}")
            camera_payload[camera] = {
                "calibration_camera": result.calibration_camera,
                "image": str(image_path),
                "valid": bool(result.detections),
                "detections": result.detections,
                "error": result.error,
            }
        temp_image_dir.replace(image_dir)
    except Exception:
        shutil.rmtree(temp_image_dir, ignore_errors=True)
        raise
    record = {
        "capture_index": capture_index,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sync": {
            "publish_seq": cluster.publish_seq,
            "logical_frame_index": cluster.logical_frame_index,
            "sync_timestamp_ns": cluster.sync_timestamp_ns,
            "max_delta_ns": cluster.max_delta_ns,
        },
        "T_base_tcp": matrix_payload(T_base_tcp),
        "joint_values_rad": joints,
        "cameras": camera_payload,
    }
    records.append(record)
    _write_json(
        run_dir / "captures.json",
        {
            "schema": CAPTURES_SCHEMA,
            "marker": {"family": "tag36h11", "id": 6, "marker_size_m": TAG_SIZE_M},
            "records": records,
        },
    )
    return record


def _solve_and_activate(
    run_dir: Path,
    root: Path,
    records: list[dict[str, Any]],
    intrinsics_summary: Path,
) -> Path:
    previous_summary = run_dir / "camera_calibration/summary.json"
    if previous_summary.is_file():
        attempts_dir = run_dir / "camera_calibration/solve_attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        archived = attempts_dir / f"summary_before_{_utc_stamp()}.json"
        shutil.copy2(previous_summary, archived)
        print(f"[SOLVE] archived previous summary: {archived}", flush=True)
    print(
        f"[SOLVE] using existing fisheye intrinsics: {intrinsics_summary} "
        "(cam_03 temporarily uses cam_13 intrinsics)",
        flush=True,
    )
    calibration_path, calibration = calibrate_and_write(
        records,
        run_dir / "camera_calibration",
        intrinsics_summary,
        marker_size_m=TAG_SIZE_M,
        min_frames=MIN_EXTRINSIC_FRAMES,
    )
    _write_json(
        root / "active.json",
        {
            "schema": "fr3_base_single_tag_camera_calibration_active/v1",
            "producer": "p0_single_tag_fixed_camera_calibration",
            "activated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "calibration_path": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "intrinsics_summary": str(intrinsics_summary.resolve()),
            "intrinsics_summary_sha256": _sha256(intrinsics_summary),
            "world_frame_id": calibration["world"]["world_frame_id"],
        },
    )
    print(f"[DONE] current-FR3-base camera calibration activated: {calibration_path}", flush=True)
    return calibration_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", choices=("ask", "reuse", "recalibrate"), default="ask")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize connecting FR3 in zero-stiffness teaching mode",
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--robot-ip", default="192.168.11.102")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--recorder-config", type=Path, default=DEFAULT_RECORDER_CONFIG)
    parser.add_argument("--intrinsics-summary", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--resume-run",
        default=None,
        metavar="PATH|latest",
        help="continue appending to a committed interrupted run instead of creating a new one",
    )
    parser.add_argument(
        "--solve-run",
        default=None,
        metavar="PATH|latest",
        help="offline solve/activate a committed run without connecting cameras or FR3",
    )
    parser.add_argument("--target-per-camera", type=int, default=30)
    parser.add_argument("--detection-workers", type=int, default=2)
    parser.add_argument("--frame-bus-every-n", type=int, default=15)
    parser.add_argument("--detection-scale", type=float, default=0.5)
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV native worker count; keep at 1 while the FR3 teaching controller is active",
    )
    parser.add_argument(
        "--robot-control-cpu",
        type=int,
        default=None,
        help="CPU for the isolated FR3 teaching process (default: auto from robot NIC IRQ)",
    )
    parser.add_argument(
        "--no-cpu-isolation",
        action="store_true",
        help="disable FR3/vision CPU affinity isolation (diagnostic only)",
    )
    parser.add_argument(
        "--robot-only-test-seconds",
        type=float,
        default=0.0,
        help=(
            "run the same zero-stiffness FR3 controller for this duration without cameras, "
            "AprilTag detection, or a window, then exit"
        ),
    )
    parser.add_argument("--settle-time-s", type=float, default=0.12)
    parser.add_argument("--max-settle-joint-delta-rad", type=float, default=0.003)
    parser.add_argument(
        "--camera-alias",
        action="append",
        default=[],
        metavar="CURRENT=CALIBRATED",
        help="map a runtime camera id to its calibrated identity; accepts cam_05=cam_06 or 5=6",
    )
    parser.add_argument(
        "--exclude-camera",
        action="append",
        default=[],
        metavar="CAMERA",
        help="exclude an additional runtime camera id; cam_02 (UMI) is always excluded",
    )
    parser.add_argument("--enable-hardware-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume_run is not None and args.solve_run is not None:
        parser.error("--resume-run and --solve-run are mutually exclusive")
    try:
        camera_aliases = _parse_camera_aliases(args.camera_alias)
        excluded_camera_ids = _excluded_sensor_ids(args.exclude_camera)
    except ValueError as exc:
        parser.error(str(exc))

    root = args.output_root.expanduser().resolve()
    active = _load_active(root)
    if _choose_existing(args.existing, active):
        print("[DONE] existing calibration retained; cameras and robot were not connected", flush=True)
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "would_recalibrate": True,
                    "output_root": str(root),
                    "recorder_config": str(args.recorder_config.expanduser().resolve()),
                    "intrinsics_summary": str(args.intrinsics_summary.expanduser().resolve()),
                    "intrinsics_policy": {"default": "same camera", "cam_03": "cam_13"},
                    "robot_only_test_seconds": args.robot_only_test_seconds,
                    "opencv_threads": args.opencv_threads,
                    "robot_control_cpu": args.robot_control_cpu,
                    "cpu_isolation": not args.no_cpu_isolation,
                    "resume_run": args.resume_run,
                    "solve_run": args.solve_run,
                    "marker": "tag36h11 id=6 size=0.16m",
                    "camera_model": "opencv_fisheye",
                    "output_frame": "current fr3_base",
                    "camera_aliases": camera_aliases,
                    "excluded_camera_ids": excluded_camera_ids,
                },
                indent=2,
            )
        )
        return 0
    if args.solve_run is not None:
        intrinsics_summary = args.intrinsics_summary.expanduser().resolve()
        if not intrinsics_summary.is_file():
            parser.error(f"existing fisheye intrinsics summary not found: {intrinsics_summary}")
        try:
            load_existing_fisheye_intrinsics(intrinsics_summary)
            solve_run = _resolve_resume_run(root, args.solve_run)
            if solve_run is None:
                raise ValueError("--solve-run requires a run path or latest")
            records = _load_resume_records(solve_run)
            manifest_path = solve_run / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_sha256 = manifest.get("intrinsics_summary_sha256")
                if expected_sha256 and expected_sha256 != _sha256(intrinsics_summary):
                    raise ValueError("intrinsics summary differs from the captured run manifest")
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            parser.error(f"cannot solve calibration run: {exc}")
        print(
            f"[OFFLINE SOLVE] run={solve_run} committed_captures={len(records)}; "
            "cameras and FR3 will not be connected",
            flush=True,
        )
        _solve_and_activate(solve_run, root, records, intrinsics_summary)
        return 0
    if not args.execute or args.confirmation != "P0_TWO_MARKER_TEACHING":
        parser.error("recalibration requires --execute --confirmation P0_TWO_MARKER_TEACHING")
    if (
        args.target_per_camera < MIN_EXTRINSIC_FRAMES
        or args.detection_workers <= 0
        or args.opencv_threads <= 0
    ):
        parser.error(
            f"--target-per-camera must be at least {MIN_EXTRINSIC_FRAMES}; "
            "--detection-workers and --opencv-threads must be positive"
        )
    if args.robot_only_test_seconds < 0.0:
        parser.error("--robot-only-test-seconds must be non-negative")
    cv2.setNumThreads(args.opencv_threads)
    try:
        cpu_plan = _configure_cpu_isolation(
            args.robot_ip,
            args.robot_control_cpu,
            args.no_cpu_isolation,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"cannot configure CPU isolation: {exc}")
    if args.robot_only_test_seconds > 0.0:
        print(
            "[SAFETY] ROBOT-ONLY TEST: no cameras/UI will start, but FR3 will enter the same "
            "zero-stiffness teaching controller. Support the arm and keep the E-stop ready.",
            flush=True,
        )
        robot_worker = RobotTeachingWorker(
            args.robot_ip,
            args.urdf.expanduser().resolve(),
            cpu_plan.robot_control_cpu,
        )
        try:
            robot_worker.start()
            _run_isolated_robot_only_control_test(robot_worker, args.robot_only_test_seconds)
        finally:
            robot_worker.stop()
        return 0
    intrinsics_summary = args.intrinsics_summary.expanduser().resolve()
    if not intrinsics_summary.is_file():
        parser.error(f"existing fisheye intrinsics summary not found: {intrinsics_summary}")
    try:
        existing_intrinsics = load_existing_fisheye_intrinsics(intrinsics_summary)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(f"invalid existing fisheye intrinsics: {exc}")
    try:
        resume_run = _resolve_resume_run(root, args.resume_run)
        records = [] if resume_run is None else _load_resume_records(resume_run)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"cannot resume calibration run: {exc}")
    if resume_run is not None:
        solved_path = resume_run / "camera_calibration/summary.json"
        if solved_path.is_file():
            try:
                solved = json.loads(solved_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"cannot inspect resumed calibration summary: {exc}")
            if solved.get("status") == "passed":
                parser.error(
                    f"resume run already has a passing calibration: {solved_path}; "
                    "start a new run instead"
                )

    print(
        "[SAFETY] FR3 will enter zero-stiffness teaching mode. Move it by hand, stop completely, "
        "then press Enter in the camera window. Keep the physical E-stop ready.",
        flush=True,
    )
    locked_sensor_ids = _locked_sensor_ids()
    selected_sensor_ids, ignored_cameras = _select_sensor_ids(
        locked_sensor_ids,
        excluded_camera_ids,
    )
    if len(selected_sensor_ids) < MIN_CAMERAS:
        parser.error(
            f"only {len(selected_sensor_ids)} locked non-UMI camera(s) remain; "
            f"locked={locked_sensor_ids}, ignored={ignored_cameras}"
        )
    print(
        f"[CAMERAS] locked={locked_sensor_ids}; selected={selected_sensor_ids}; "
        f"ignored={ignored_cameras}",
        flush=True,
    )
    run_dir = resume_run or (root / f"manual_run_{_utc_stamp()}")
    if resume_run is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    frame_bus_dir = Path(f"/dev/shm/fr3_base_single_tag_{os.getpid()}")
    recorder_config = _make_recorder_config(
        args.recorder_config.expanduser().resolve(),
        run_dir,
        frame_bus_dir,
        args.frame_bus_every_n,
        selected_sensor_ids,
    )
    robot_worker = RobotTeachingWorker(
        args.robot_ip,
        args.urdf.expanduser().resolve(),
        cpu_plan.robot_control_cpu,
    )
    recorder: ThorRecorderClient | None = None
    counts: dict[str, int] = {}
    should_solve = False
    window: OperatorWindow | None = None
    try:
        recorder = _start_recorder(recorder_config, skip_hardware_sync=not args.enable_hardware_sync)
        frame_client = ThorOnlineSyncFrameClient(frame_bus_dir)
        cluster = frame_client.get_latest(timeout_s=30.0)
        if cluster is None:
            raise RuntimeError(f"no synchronized frame cluster appeared under {frame_bus_dir}")
        fresh_cluster = frame_client.get_latest(
            timeout_s=2.5,
            min_publish_seq=cluster.publish_seq + 1,
        )
        if fresh_cluster is None:
            raise RuntimeError(
                "camera stream stalled before robot connection; no second synchronized cluster "
                "arrived within 2.5s. Check the timed-out camera in the [THOR] log, recover it, "
                "or rerun with --exclude-camera cam_XX"
            )
        cluster = fresh_cluster
        resolved_cameras = {
            camera: camera_aliases.get(camera, camera) for camera in sorted(cluster.frames)
        }
        duplicated_targets = sorted(
            {
                calibrated
                for calibrated in resolved_cameras.values()
                if list(resolved_cameras.values()).count(calibrated) > 1
            }
        )
        if duplicated_targets:
            raise RuntimeError(
                "multiple live cameras resolve to the same calibrated identity: "
                + ", ".join(duplicated_targets)
            )
        missing_intrinsics = sorted(
            camera
            for camera in resolved_cameras.values()
            if ("cam_13" if camera == "cam_03" else camera) not in existing_intrinsics
        )
        if missing_intrinsics:
            raise RuntimeError(
                "no existing fisheye intrinsics for calibrated camera identities: "
                + ", ".join(missing_intrinsics)
            )
        expected_live_cameras = sorted(resolved_cameras)
        print(f"[CAMERAS] live -> calibrated identities: {resolved_cameras}", flush=True)
        try:
            if records:
                counts = _resume_counts(records, resolved_cameras)
            else:
                counts = {camera: 0 for camera in expected_live_cameras}
            _write_or_validate_run_manifest(
                run_dir,
                intrinsics_summary,
                resolved_cameras,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"resume validation failed: {exc}") from exc
        if records:
            next_capture = max(int(record["capture_index"]) for record in records) + 1
            print(
                f"[RESUME] run={run_dir} committed_captures={len(records)} "
                f"next_capture={next_capture:03d} counts={counts}",
                flush=True,
            )
            print(
                "[RESUME] valid only if FR3 base, fixed cameras, and EE tag mount have not moved",
                flush=True,
            )
        window = OperatorWindow(WINDOW_NAME)
        robot_worker.start()
        last_seq = cluster.publish_seq
        last_fresh_frame_at = time.monotonic()
        last_capture_seq = -1
        results: dict[str, CameraResult] = {}
        ui_message = "Move FR3 by hand, stop completely, then press Enter"
        with ThreadPoolExecutor(
            max_workers=args.detection_workers, thread_name_prefix="apriltag-camera"
        ) as pool:
            while True:
                robot_worker.raise_if_failed()
                next_cluster = frame_client.get_latest(timeout_s=0.2, min_publish_seq=last_seq + 1)
                if next_cluster is not None:
                    cluster = next_cluster
                    last_seq = cluster.publish_seq
                    last_fresh_frame_at = time.monotonic()
                    results = _process_cluster(
                        cluster,
                        pool,
                        camera_aliases,
                        args.detection_scale,
                    )
                    for camera in results:
                        counts.setdefault(camera, 0)
                if results:
                    if time.monotonic() - last_fresh_frame_at > 1.0:
                        ui_message = "STREAM STALLED: capture disabled; check [THOR] camera timeout"
                    window.show(
                        _draw_mosaic(results, counts, args.target_per_camera, ui_message)
                    )
                key = window.poll_key()
                if key in (10, 13):
                    if time.monotonic() - last_fresh_frame_at > 1.0:
                        ui_message = "REJECTED: synchronized camera stream is stalled"
                        print(f"[REJECTED] {ui_message}", flush=True)
                        continue
                    if cluster.publish_seq == last_capture_seq:
                        ui_message = "REJECTED: no new synchronized frame since last capture"
                        print(f"[REJECTED] {ui_message}", flush=True)
                        continue
                    try:
                        record = _save_capture(
                            run_dir,
                            records,
                            results,
                            cluster,
                            robot_worker,
                            args.settle_time_s,
                            args.max_settle_joint_delta_rad,
                        )
                        valid = [camera for camera, item in record["cameras"].items() if item["valid"]]
                        for camera in valid:
                            counts[camera] += 1
                        last_capture_seq = cluster.publish_seq
                        _, readiness = _capture_readiness(
                            counts, expected_live_cameras, args.target_per_camera
                        )
                        ui_message = f"Captured #{len(records)}: {readiness}"
                        print(
                            f"[CAPTURE {len(records):03d}] valid cameras={valid}; counts={counts}; {readiness}",
                            flush=True,
                        )
                    except Exception as exc:
                        ui_message = f"REJECTED: {exc}"
                        print(f"[REJECTED] {exc}", flush=True)
                elif key == ord("q"):
                    ready, readiness = _capture_readiness(
                        counts, expected_live_cameras, args.target_per_camera
                    )
                    if ready:
                        print(f"[READY] {readiness}", flush=True)
                        should_solve = True
                        break
                    ui_message = f"NOT READY: {readiness}"
                    print(f"[NOT READY] {readiness}; continue capture or press f to force", flush=True)
                elif key == ord("f"):
                    print("[FORCE] attempting solve with current accepted observations", flush=True)
                    should_solve = True
                    break
                elif key == 27:
                    print("[ABORT] active calibration was not changed", flush=True)
                    break
    finally:
        try:
            if window is not None:
                window.close()
        except Exception:
            pass
        try:
            robot_worker.stop()
        finally:
            try:
                if recorder is not None:
                    recorder.stop()
            finally:
                shutil.rmtree(frame_bus_dir, ignore_errors=True)
                shutil.rmtree(recorder_config.parent, ignore_errors=True)

    if not should_solve:
        return 130
    _solve_and_activate(
        run_dir,
        root,
        records,
        intrinsics_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
