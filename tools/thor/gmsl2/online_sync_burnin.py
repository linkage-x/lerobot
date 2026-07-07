#!/usr/bin/env python3
"""Run or analyze argus_online_sync burn-in recordings.

This utility is intentionally outside the GUI path.  It drives the same
``thor_record.py`` stdin/stdout protocol that the UI uses, then summarizes the
saved ``online_sync_manifest.json`` and sidecar files.  It never materializes or
re-encodes video; optional ffprobe checks are read-only QC.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml"
CAMERA_SID_RE = re.compile(r"cam_(\d+)")


@dataclass
class DriverResult:
    rc: int | None
    elapsed_s: float
    dataset_root: Path | None
    log_path: Path
    timed_out: bool
    protocol_failure: str | None
    ready_count: int
    saved_count: int


def _json_read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sidecar_row_count(path: Path) -> int | None:
    try:
        count = 0
        with path.open() as f:
            for i, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                if i == 0 and text.startswith("camera,"):
                    continue
                count += 1
        return count
    except OSError:
        return None


def _camera_sort_key(camera: str) -> tuple[int, str]:
    match = CAMERA_SID_RE.fullmatch(camera)
    return (int(match.group(1)), camera) if match else (9999, camera)


def _camera_names(ep_dir: Path, manifest: dict[str, Any] | None) -> list[str]:
    names: set[str] = set()
    if manifest:
        for camera in manifest.get("active_cameras") or []:
            if isinstance(camera, str):
                names.add(camera)
        for camera in (manifest.get("frame_count_by_camera") or {}).keys():
            names.add(str(camera))
    for path in ep_dir.glob("cam_*.argus_frame_metadata.csv"):
        names.add(path.name.removesuffix(".argus_frame_metadata.csv"))
    for path in list(ep_dir.glob("cam_*.mkv")) + list(ep_dir.glob("cam_*.mp4")):
        names.add(path.stem)
    return sorted(names, key=_camera_sort_key)


def _video_path(ep_dir: Path, camera: str) -> Path | None:
    for suffix in (".mkv", ".mp4"):
        path = ep_dir / f"{camera}{suffix}"
        if path.exists():
            return path
    return None


def _ffprobe_frame_count(path: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return int(text[0])
    except ValueError:
        return None


def analyze_episode(
    ep_dir: Path,
    *,
    expected_frames: int | None = None,
    tolerance_ns: int | None = None,
    run_ffprobe: bool = False,
) -> dict[str, Any]:
    manifest_path = ep_dir / "online_sync_manifest.json"
    manifest = _json_read(manifest_path)
    meta = _json_read(ep_dir / "meta.json") or {}
    failures: list[str] = []

    if manifest is None:
        failures.append("missing or invalid online_sync_manifest.json")

    cameras = _camera_names(ep_dir, manifest)
    actual_frames = None
    target_frames = None
    manifest_counts: dict[str, int] = {}
    max_delta: dict[str, int] = {}
    manifest_ok = False
    if manifest is not None:
        manifest_ok = bool(manifest.get("ok"))
        if not manifest_ok:
            failures.append(str(manifest.get("failure") or "manifest ok=false"))
        try:
            actual_frames = int(manifest.get("actual_frames"))
        except (TypeError, ValueError):
            failures.append("manifest actual_frames is missing or invalid")
        try:
            target_frames = int(manifest.get("target_frames"))
        except (TypeError, ValueError):
            target_frames = None
        for camera, value in (manifest.get("frame_count_by_camera") or {}).items():
            try:
                manifest_counts[str(camera)] = int(value)
            except (TypeError, ValueError):
                failures.append(f"{camera} manifest frame count is invalid")
        for camera, value in (manifest.get("max_abs_delta_ns_by_camera") or {}).items():
            try:
                max_delta[str(camera)] = int(value)
            except (TypeError, ValueError):
                failures.append(f"{camera} max SOF delta is invalid")

    sidecar_counts: dict[str, int | None] = {}
    ffprobe_counts: dict[str, int | None] = {}
    for camera in cameras:
        sidecar = ep_dir / f"{camera}.argus_frame_metadata.csv"
        sidecar_counts[camera] = _sidecar_row_count(sidecar)
        if run_ffprobe:
            video = _video_path(ep_dir, camera)
            ffprobe_counts[camera] = _ffprobe_frame_count(video) if video else None

    if expected_frames is not None and actual_frames != expected_frames:
        failures.append(f"actual_frames {actual_frames} != expected {expected_frames}")
    if actual_frames is not None:
        for camera in cameras:
            count = manifest_counts.get(camera)
            if count != actual_frames:
                failures.append(f"{camera} manifest frame count {count} != {actual_frames}")
            sidecar_count = sidecar_counts.get(camera)
            if sidecar_count != actual_frames:
                failures.append(f"{camera} sidecar rows {sidecar_count} != {actual_frames}")
            if run_ffprobe:
                video_count = ffprobe_counts.get(camera)
                if video_count != actual_frames:
                    failures.append(f"{camera} video frames {video_count} != {actual_frames}")

    effective_tolerance_ns = tolerance_ns
    if effective_tolerance_ns is None and manifest is not None:
        try:
            effective_tolerance_ns = int(manifest.get("tolerance_ns"))
        except (TypeError, ValueError):
            effective_tolerance_ns = None
    if effective_tolerance_ns is not None:
        for camera, delta in max_delta.items():
            if delta > effective_tolerance_ns:
                failures.append(
                    f"{camera} max SOF delta {delta} ns > tolerance {effective_tolerance_ns} ns"
                )

    return {
        "episode": ep_dir.name,
        "path": str(ep_dir),
        "ok": not failures,
        "failures": failures,
        "manifest_ok": manifest_ok,
        "target_frames": target_frames,
        "actual_frames": actual_frames,
        "active_cameras": cameras,
        "manifest_frame_count_by_camera": manifest_counts,
        "sidecar_rows_by_camera": sidecar_counts,
        "ffprobe_video_frames_by_camera": ffprobe_counts if run_ffprobe else None,
        "max_abs_delta_ns_by_camera": max_delta,
        "cleanup_duration_s": meta.get("cleanup_duration_s"),
        "split_emit_ms": meta.get("split_emit_ms"),
        "duration_s": meta.get("duration_s"),
        "recording_stop_reason": meta.get("recording_stop_reason"),
    }


def analyze_dataset(
    dataset_root: Path,
    *,
    expected_episodes: int | None = None,
    expected_frames: int | None = None,
    expected_cameras: list[str] | None = None,
    tolerance_ns: int | None = None,
    run_ffprobe: bool = False,
) -> dict[str, Any]:
    episodes_root = dataset_root / "episodes"
    episodes = sorted(episodes_root.glob("episode_*")) if episodes_root.exists() else []
    episode_reports = [
        analyze_episode(
            ep_dir,
            expected_frames=expected_frames,
            tolerance_ns=tolerance_ns,
            run_ffprobe=run_ffprobe,
        )
        for ep_dir in episodes
        if ep_dir.is_dir()
    ]
    failures: list[str] = []
    if expected_episodes is not None and len(episode_reports) != expected_episodes:
        failures.append(f"episodes found {len(episode_reports)} != expected {expected_episodes}")
    for report in episode_reports:
        if expected_cameras is not None:
            active = sorted(str(camera) for camera in report.get("active_cameras") or [])
            expected = sorted(expected_cameras)
            if active != expected:
                failures.append(
                    f"{report['episode']}: active cameras {active} != expected {expected}"
                )
        failures.extend(f"{report['episode']}: {failure}" for failure in report["failures"])
    max_delta_ns = 0
    for report in episode_reports:
        deltas = report.get("max_abs_delta_ns_by_camera") or {}
        if deltas:
            max_delta_ns = max(max_delta_ns, max(int(v) for v in deltas.values()))
    return {
        "dataset_root": str(dataset_root),
        "postprocessing": {
            "ffmpeg_materialization": False,
            "ffprobe_enabled": run_ffprobe,
        },
        "summary": {
            "ok": not failures and bool(episode_reports),
            "episodes_found": len(episode_reports),
            "expected_episodes": expected_episodes,
            "expected_frames": expected_frames,
            "expected_cameras": expected_cameras,
            "max_delta_ns": max_delta_ns,
            "failures": failures,
        },
        "episodes": episode_reports,
    }


def _patch_config(
    config_path: Path,
    output_path: Path,
    *,
    dataset_root: Path,
    episodes: int,
    episode_time_s: float,
    fps: int,
    sensor_ids: list[int] | None,
    detect_all: bool,
    preview: bool,
    box_enabled: bool,
) -> None:
    raw = yaml.safe_load(config_path.read_text()) or {}
    raw.setdefault("dataset", {})
    raw["dataset"]["root"] = str(dataset_root)
    raw["dataset"]["num_episodes"] = int(episodes)
    raw["dataset"]["episode_time_s"] = float(episode_time_s)
    raw["dataset"]["fps"] = int(fps)

    cameras = raw.setdefault("sensors", {}).setdefault("cameras", {})
    cameras["detect_all"] = bool(detect_all)
    if sensor_ids is not None:
        cameras["detect_all"] = False
        cameras["sensor_ids"] = sensor_ids
    cameras["recording_preview_enabled"] = bool(preview)
    defaults = cameras.setdefault("defaults", {})
    defaults["recorder_backend"] = "argus_online_sync"
    defaults["fps"] = int(fps)

    raw.setdefault("box_collection", {})
    raw["box_collection"]["enabled"] = bool(box_enabled)

    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def run_ui_burnin(
    *,
    config_path: Path,
    repo_root: Path,
    dataset_root: Path,
    log_dir: Path,
    episodes: int,
    episode_time_s: float,
    fps: int,
    sensor_ids: list[int] | None,
    detect_all: bool,
    preview: bool,
    box_enabled: bool,
    no_auto_recover: bool,
    skip_argus_probe: bool,
    run_timeout_s: float,
    continue_on_failure: bool,
    debug: bool,
) -> DriverResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    driver_log = log_dir / "thor_record_driver.log"
    tmp_config = log_dir / "online_sync_burnin_config.yaml"
    _patch_config(
        config_path,
        tmp_config,
        dataset_root=dataset_root,
        episodes=episodes,
        episode_time_s=episode_time_s,
        fps=fps,
        sensor_ids=sensor_ids,
        detect_all=detect_all,
        preview=preview,
        box_enabled=box_enabled,
    )

    cmd = [
        sys.executable,
        str(repo_root / "tools/thor/gmsl2/thor_record.py"),
        "--config-path", str(tmp_config),
        "--repo-root", str(repo_root),
    ]
    if not box_enabled:
        cmd.append("--no-box")
    if no_auto_recover:
        cmd.append("--no-auto-recover")
    if skip_argus_probe:
        cmd.append("--skip-argus-probe")
    if debug:
        cmd.append("--debug")

    start = time.monotonic()
    dataset_root_actual: Path | None = None
    ready_count = 0
    saved_count = 0
    timed_out = False
    protocol_failure: str | None = None
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=repo_root,
        start_new_session=True,
    )
    assert proc.stdout is not None
    output_queue: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_queue.put(line)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    with driver_log.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")

        def handle_line(line: str) -> bool:
            nonlocal dataset_root_actual, ready_count, saved_count, protocol_failure
            log.write(line)
            log.flush()
            text = line.strip()
            if text.startswith("Dataset root:"):
                dataset_root_actual = Path(text.split(":", 1)[1].strip())
            if (
                not continue_on_failure
                and (
                    text.startswith("ERROR:")
                    or text == "Episode discarded"
                    or text.startswith("Stream exited early:")
                )
            ):
                protocol_failure = text
                log.write(f"ERROR: burn-in fail-fast after recorder output: {text}\n")
                _kill_process_group(proc)
                return True
            if re.fullmatch(r"Episode \d+ ready", text):
                ready_count += 1
                if proc.stdin is not None:
                    proc.stdin.write("\n")
                    proc.stdin.flush()
            elif text == "Episode saved.":
                saved_count += 1
            return False

        while True:
            if time.monotonic() - start > run_timeout_s:
                timed_out = True
                log.write(f"ERROR: burn-in driver timed out after {run_timeout_s:.1f}s\n")
                _kill_process_group(proc)
                break
            if proc.poll() is not None and output_queue.empty():
                break
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if handle_line(line):
                break
        while not output_queue.empty():
            handle_line(output_queue.get_nowait())
    reader_thread.join(timeout=1.0)
    elapsed = time.monotonic() - start
    return DriverResult(
        rc=proc.returncode,
        elapsed_s=elapsed,
        dataset_root=dataset_root_actual,
        log_path=driver_log,
        timed_out=timed_out,
        protocol_failure=protocol_failure,
        ready_count=ready_count,
        saved_count=saved_count,
    )


def _format_map(values: dict[str, Any] | None) -> str:
    if not values:
        return "{}"
    parts = [f"{k}: {v}" for k, v in sorted(values.items(), key=lambda kv: _camera_sort_key(kv[0]))]
    return "{" + ", ".join(parts) + "}"


def write_report(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# argus_online_sync burn-in sync analysis report",
        "",
        "## Summary",
        "",
        f"- dataset_root: `{report['dataset_root']}`",
        f"- ok: `{summary['ok']}`",
        f"- episodes_found: `{summary['episodes_found']}`",
        f"- expected_episodes: `{summary.get('expected_episodes')}`",
        f"- expected_frames: `{summary.get('expected_frames')}`",
        f"- max_delta_ns: `{summary.get('max_delta_ns')}`",
        "- postprocessing: no ffmpeg materialization/re-encode; "
        f"ffprobe QC enabled = `{report['postprocessing']['ffprobe_enabled']}`",
        "",
    ]
    driver = report.get("driver")
    if driver:
        lines.extend([
            "## Driver",
            "",
            f"- rc: `{driver.get('rc')}`",
            f"- timed_out: `{driver.get('timed_out')}`",
            f"- protocol_failure: `{driver.get('protocol_failure')}`",
            f"- elapsed_s: `{driver.get('elapsed_s')}`",
            f"- ready_count: `{driver.get('ready_count')}`",
            f"- saved_count: `{driver.get('saved_count')}`",
            f"- log_path: `{driver.get('log_path')}`",
            "",
        ])
    failures = summary.get("failures") or []
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    lines.extend(["## Episodes", ""])
    for ep in report["episodes"]:
        lines.extend([
            f"### {ep['episode']}",
            "",
            f"- ok: `{ep['ok']}`",
            f"- actual_frames: `{ep.get('actual_frames')}`",
            f"- target_frames: `{ep.get('target_frames')}`",
            f"- active_cameras: `{', '.join(ep.get('active_cameras') or [])}`",
            f"- manifest frame counts: `{_format_map(ep.get('manifest_frame_count_by_camera'))}`",
            f"- sidecar rows: `{_format_map(ep.get('sidecar_rows_by_camera'))}`",
            f"- video frames: `{_format_map(ep.get('ffprobe_video_frames_by_camera'))}`",
            f"- max_abs_delta_ns_by_camera: `{_format_map(ep.get('max_abs_delta_ns_by_camera'))}`",
            f"- duration_s: `{ep.get('duration_s')}`",
            f"- cleanup_duration_s: `{ep.get('cleanup_duration_s')}`",
            "",
        ])
        if ep.get("failures"):
            lines.append("Failures:")
            lines.extend(f"- {failure}" for failure in ep["failures"])
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def _parse_sensor_ids(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part) for part in text.split(",") if part.strip()]


def _default_dataset_base(episodes: int, episode_time_s: float) -> Path:
    seconds = int(round(episode_time_s))
    return REPO_ROOT / "outputs/datasets" / f"online_sync_burnin_{episodes}x{seconds}"


def _default_timeout(episodes: int, episode_time_s: float) -> float:
    return max(300.0, 180.0 + episodes * (episode_time_s + 180.0))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run thor_record.py via the UI stdin protocol")
    run.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run.add_argument("--episodes", type=int, default=10)
    run.add_argument("--episode-time-s", type=float, default=60.0)
    run.add_argument("--fps", type=int, default=60)
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--log-dir", type=Path)
    run.add_argument("--sensor-ids", type=str, help="comma-separated sids; omitted means detect_all")
    run.add_argument("--preview", action="store_true", help="leave idle preview enabled during burn-in")
    run.add_argument("--with-box", action="store_true", help="record BOX sensors too; default is camera-only")
    run.add_argument("--allow-auto-recover", action="store_true")
    run.add_argument(
        "--skip-argus-probe",
        action="store_true",
        help=(
            "pass locked/requested cameras directly to the recorder session; "
            "online-sync preflight remains the authoritative camera check"
        ),
    )
    run.add_argument(
        "--continue-on-failure",
        action="store_true",
        help=(
            "keep driving new episodes after recorder ERROR/discard/stream-exit "
            "lines; default stops immediately to avoid repeatedly stressing a "
            "wedged camera stack"
        ),
    )
    run.add_argument("--run-timeout-s", type=float)
    run.add_argument("--ffprobe", action="store_true", help="read-only video frame-count QC")
    run.add_argument("--debug", action="store_true")

    analyze = sub.add_parser("analyze", help="analyze an existing dataset root")
    analyze.add_argument("dataset_root", type=Path)
    analyze.add_argument("--expected-episodes", type=int)
    analyze.add_argument("--expected-frames", type=int)
    analyze.add_argument("--tolerance-ns", type=int)
    analyze.add_argument("--ffprobe", action="store_true")
    analyze.add_argument("--report-path", type=Path)
    analyze.add_argument("--summary-json", type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "analyze":
        report = analyze_dataset(
            args.dataset_root,
            expected_episodes=args.expected_episodes,
            expected_frames=args.expected_frames,
            tolerance_ns=args.tolerance_ns,
            run_ffprobe=args.ffprobe,
        )
        report_path = args.report_path or args.dataset_root / "online_sync_burnin_sync_report.md"
        summary_json = args.summary_json or args.dataset_root / "online_sync_burnin_summary.json"
        write_report(report, report_path)
        summary_json.write_text(json.dumps(report, indent=2))
        print(f"report: {report_path}")
        print(f"summary: {summary_json}")
        return 0 if report["summary"]["ok"] else 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_base = args.dataset_root or _default_dataset_base(args.episodes, args.episode_time_s)
    log_dir = args.log_dir or Path(tempfile.gettempdir()) / f"online_sync_burnin_{stamp}"
    timeout_s = args.run_timeout_s or _default_timeout(args.episodes, args.episode_time_s)
    sensor_ids = _parse_sensor_ids(args.sensor_ids)
    driver = run_ui_burnin(
        config_path=args.config_path,
        repo_root=args.repo_root,
        dataset_root=dataset_base,
        log_dir=log_dir,
        episodes=args.episodes,
        episode_time_s=args.episode_time_s,
        fps=args.fps,
        sensor_ids=sensor_ids,
        detect_all=sensor_ids is None,
        preview=args.preview,
        box_enabled=args.with_box,
        no_auto_recover=not args.allow_auto_recover,
        skip_argus_probe=args.skip_argus_probe,
        run_timeout_s=timeout_s,
        continue_on_failure=args.continue_on_failure,
        debug=args.debug,
    )
    dataset_root = driver.dataset_root or dataset_base
    expected_frames = int(round(args.episode_time_s * args.fps))
    report = analyze_dataset(
        dataset_root,
        expected_episodes=args.episodes,
        expected_frames=expected_frames,
        expected_cameras=(
            [f"cam_{sid:02d}" for sid in sensor_ids]
            if sensor_ids is not None else None
        ),
        tolerance_ns=1_000_000,
        run_ffprobe=args.ffprobe,
    )
    report["driver"] = {
        "rc": driver.rc,
        "elapsed_s": driver.elapsed_s,
        "dataset_root": str(driver.dataset_root) if driver.dataset_root else None,
        "log_path": str(driver.log_path),
        "timed_out": driver.timed_out,
        "protocol_failure": driver.protocol_failure,
        "ready_count": driver.ready_count,
        "saved_count": driver.saved_count,
    }
    if driver.rc != 0:
        report["summary"]["ok"] = False
        report["summary"]["failures"].append(f"driver rc={driver.rc}")
    if driver.timed_out:
        report["summary"]["ok"] = False
        report["summary"]["failures"].append("driver timed out")
    if driver.protocol_failure:
        report["summary"]["ok"] = False
        report["summary"]["failures"].append(
            f"driver fail-fast after recorder output: {driver.protocol_failure}"
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    report_path = dataset_root / "online_sync_burnin_sync_report.md"
    summary_json = dataset_root / "online_sync_burnin_summary.json"
    write_report(report, report_path)
    summary_json.write_text(json.dumps(report, indent=2))
    print(f"dataset_root: {dataset_root}")
    print(f"driver_log: {driver.log_path}")
    print(f"report: {report_path}")
    print(f"summary: {summary_json}")
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
