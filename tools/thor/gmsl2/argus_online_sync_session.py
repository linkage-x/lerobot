"""Drop-in Thor recorder session for encoder-front online synchronization."""

from __future__ import annotations

import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from tools.thor.gmsl2 import argus_metadata_session as ams
from tools.thor.gmsl2 import persistent_session as ps

logger = logging.getLogger("argus_online_sync_session")


DEFAULT_BINARY_PATH = Path("/tmp/lerobot_argus_online_sync_video_recorder")
DEFAULT_PREVIEW_FRAME_BUS_DIR = Path("/dev/shm/lerobot_online_sync_preview")


class ArgusOnlineSyncCameraSession(ams.ArgusMetadataCameraSession):
    """Session wrapper around ``argus_online_sync_video_recorder``.

    It intentionally reuses the metadata session's process lifecycle and
    preflight/drop-bad-camera behavior, but the recorder itself aligns frames
    before hardware encoding. The normal UI path passes ``target_frames=0`` and
    stops by sending ``STOP`` at the UI episode boundary; the C++ process then
    closes on the next full SOF cluster. Finite ``target_frames`` remains
    available for direct recorder smoke tests.
    """

    def __init__(
        self,
        streams: list[ps.StreamConfig],
        warmup_dir: Path,
        *,
        repo_root: Path,
        binary_path: Path | None = None,
        auto_build: bool = True,
        connect_timeout_s: float = 30.0,
        connect_stable_s: float = 0.0,
        stop_timeout_s: float = 120.0,
        preflight_frames: int = 2,
        target_frames: int = 0,
        tolerance_ms: float = 1.0,
        startup_full_clusters: int = 30,
        frame_timeout_ms: int = 1000,
        preflight_timeout_s: float = 30.0,
        single_preflight_timeout_s: float = 10.0,
        missing_frame_policy: str = "fail_episode",
        stop_mode: str = "full_cluster",
        frame_bus_dir: str | Path = "",
        frame_bus_every_n: int = 1,
        preview_frame_bus_dir: str | Path = "",
        preview_frame_bus_every_n: int = 12,
    ):
        super().__init__(
            streams,
            warmup_dir,
            repo_root=repo_root,
            binary_path=binary_path or DEFAULT_BINARY_PATH,
            auto_build=auto_build,
            connect_timeout_s=connect_timeout_s,
            connect_stable_s=connect_stable_s,
            stop_timeout_s=stop_timeout_s,
            preflight_frames=preflight_frames,
        )
        self.target_frames = max(0, int(target_frames))
        self.tolerance_ms = float(tolerance_ms)
        self.startup_full_clusters = max(0, int(startup_full_clusters))
        self.frame_timeout_ms = max(1, int(frame_timeout_ms))
        self.preflight_timeout_s = max(1.0, float(preflight_timeout_s))
        self.single_preflight_timeout_s = max(1.0, float(single_preflight_timeout_s))
        self.missing_frame_policy = str(missing_frame_policy)
        self.stop_mode = str(stop_mode)
        self.frame_bus_dir = str(frame_bus_dir) if frame_bus_dir else ""
        self.frame_bus_every_n = max(1, int(frame_bus_every_n))
        self.preview_frame_bus_dir = (
            str(preview_frame_bus_dir) if preview_frame_bus_dir else ""
        )
        self.preview_frame_bus_every_n = max(1, int(preview_frame_bus_every_n))
        self.start_retry_settle_s = 2.0
        self.start_retries = 1
        self._persistent_ready_evt = threading.Event()
        self._episode_done_evt = threading.Event()
        self._episode_result: dict[str, Any] | None = None
        self._preview_warning_emitted = False
        self._preview_bridge_proc: subprocess.Popen[str] | None = None
        self._preview_down_logged = False

    def _recorder_source(self) -> Path:
        return Path(__file__).with_name("argus_online_sync_video_recorder.cpp")

    def _build_binary(self) -> None:
        self.binary_path.parent.mkdir(parents=True, exist_ok=True)
        src = self._recorder_source()
        cuda_compat = (
            "mkdir -p /tmp/lerobot_cuda_compat/crt && "
            "printf '%s\\n' "
            "'#pragma once' "
            "'#ifndef __host__' '#define __host__' '#endif' "
            "'#ifndef __device__' '#define __device__' '#endif' "
            "'#ifndef __global__' '#define __global__' '#endif' "
            "'#ifndef __shared__' '#define __shared__' '#endif' "
            "'#ifndef __align__' '#define __align__(n) __attribute__((aligned(n)))' '#endif' "
            "'#ifndef __builtin_align__' '#define __builtin_align__(n) __attribute__((aligned(n)))' '#endif' "
            "'#ifndef __noinline__' '#define __noinline__ __attribute__((noinline))' '#endif' "
            "'#ifndef __forceinline__' '#define __forceinline__ inline __attribute__((always_inline))' '#endif' "
            "'#ifndef __device_builtin__' '#define __device_builtin__' '#endif' "
            "'#ifndef __cudart_builtin__' '#define __cudart_builtin__' '#endif' "
            "'#ifndef __dv' '#define __dv(v) = v' '#endif' "
            "'#ifndef CUDARTAPI' '#define CUDARTAPI' '#endif' "
            "> /tmp/lerobot_cuda_compat/crt/host_defines.h && "
        )
        cmd = (
            cuda_compat +
            "g++ -std=c++14 -O2 "
            "-I/tmp/lerobot_cuda_compat "
            "-I/usr/src/jetson_multimedia_api/argus/include "
            "-I/usr/src/jetson_multimedia_api/argus/samples/utils "
            "-I/usr/src/jetson_multimedia_api/include "
            "-I/usr/src/jetson_multimedia_api/include/libjpeg-8b "
            "-I/usr/src/jetson_multimedia_api/samples/common/classes "
            "-I/usr/include/libdrm "
            "-I/usr/local/cuda/include "
            "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0) "
            f"{src} "
            "/usr/src/jetson_multimedia_api/argus/samples/utils/ArgusHelpers.cpp "
            "/usr/src/jetson_multimedia_api/argus/samples/utils/NativeBuffer.cpp "
            "/usr/src/jetson_multimedia_api/argus/samples/utils/nvmmapi/NvNativeBuffer.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvBufSurface.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2Element.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2ElementPlane.cpp "
            "/usr/src/jetson_multimedia_api/samples/common/classes/NvVideoEncoder.cpp "
            "-L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient "
            "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0) "
            "-lnvv4l2 -lnvbufsurface -lnvbufsurftransform -lnvmm_jpeg "
            "-lnvosd -ldrm -lcuda -lcudart -lvulkan "
            "-L/usr/local/cuda/lib64 "
            "-lEGL -lGLESv2 -lX11 -lpthread "
            f"-o {self.binary_path}"
        )
        logger.info("building Argus online-sync recorder: %s", self.binary_path)
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=max(1.0, self.connect_timeout_s),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "failed to build argus_online_sync_video_recorder "
                f"(rc={result.returncode}): {detail}"
            )

    def _build_record_command(
        self,
        streams: list[ps.StreamConfig],
        episode_dir: Path,
        *,
        frames: int,
        include_frame_bus: bool = True,
    ) -> list[str]:
        cmd = super()._build_record_command(streams, episode_dir, frames=frames)
        startup_full_clusters = self.startup_full_clusters
        if frames > 0 and (self.target_frames <= 0 or frames != self.target_frames):
            startup_full_clusters = min(startup_full_clusters, 2)
        cmd.extend([
            "--tolerance-ms",
            f"{self.tolerance_ms:g}",
            "--startup-full-clusters",
            str(startup_full_clusters),
            "--frame-timeout-ms",
            str(self.frame_timeout_ms),
            "--missing-frame-policy",
            self.missing_frame_policy,
            "--stop-mode",
            self.stop_mode,
        ])
        if include_frame_bus and self.frame_bus_dir:
            cmd.extend([
                "--frame-bus-dir",
                self.frame_bus_dir,
                "--frame-bus-every-n",
                str(self.frame_bus_every_n),
            ])
        if include_frame_bus and self.preview_frame_bus_dir:
            cmd.extend([
                "--preview-frame-bus-dir",
                self.preview_frame_bus_dir,
                "--preview-frame-bus-every-n",
                str(self.preview_frame_bus_every_n),
            ])
        return cmd

    def _build_persistent_command(self) -> list[str]:
        daemon_dir = self.warmup_dir / "argus_online_sync_persistent"
        cmd = self._build_record_command(self._stream_cfgs, daemon_dir, frames=0)
        cmd.append("--persistent")
        return cmd

    def connect(self) -> None:
        super().connect()
        try:
            self._start_persistent_daemon()
        except Exception:
            self.disconnect()
            raise

    def _start_persistent_daemon(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self.disable_previews()
        self.warmup_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_persistent_command()
        self._persistent_ready_evt.clear()
        self._recording_started_evt.clear()
        self._episode_done_evt.clear()
        self._reader_stop.clear()
        with self._lock:
            self._recent_output = []
            self._episode_result = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            with self._lock:
                self._proc = None
            raise
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(proc,),
            name="argus-online-sync-persistent-log",
            daemon=True,
        )
        self._reader_thread.start()
        with self._lock:
            self._proc = proc
        deadline = time.monotonic() + max(1.0, self.connect_timeout_s)
        while not self._persistent_ready_evt.is_set():
            rc = proc.poll()
            if rc is not None:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=1.0)
                    self._reader_thread = None
                with self._lock:
                    recent = "; ".join(self._recent_output[-8:])
                    self._proc = None
                message = f"online-sync persistent recorder exited during connect rc={rc}"
                if recent:
                    message = f"{message}: {recent}"
                self._errors.append(ps.StreamError(
                    sid=-1,
                    name="argus_online_sync",
                    message=message,
                ))
                raise RuntimeError(message)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not self._persistent_ready_evt.is_set():
            self._terminate_proc(proc)
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=1.0)
                self._reader_thread = None
            with self._lock:
                self._proc = None
                recent = "; ".join(self._recent_output[-8:])
            message = "online-sync persistent recorder did not report persistent ready"
            if recent:
                message = f"{message}: {recent}"
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_online_sync",
                message=message,
            ))
            raise RuntimeError(message)

    def _send_daemon_command(self, line: str) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise RuntimeError("Argus online-sync persistent recorder is not running")
        proc.stdin.write(line.rstrip("\n") + "\n")
        proc.stdin.flush()

    def _terminate_proc(self, proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            proc.wait(timeout=timeout_s)

    @staticmethod
    def _is_preflight_timeout(message: str) -> bool:
        return "preflight timed out" in message

    @staticmethod
    def _recorder_error_stream_names(
        message: str,
        candidates: list[ps.StreamConfig],
    ) -> set[str]:
        """Camera names that appear as recorder error prefixes.

        Do not treat a camera as failed merely because it appears in context
        text such as ``for cam_06,cam_09``.  The C++ recorder emits actionable
        per-camera failures as ``cam_XX: ...``; only that form is specific
        enough to drop a camera automatically.
        """

        failed: set[str] = set()
        for stream in candidates:
            pattern = rf"(?:^|[;\s]){re.escape(stream.name)}\s*:"
            if re.search(pattern, message):
                failed.add(stream.name)
        return failed

    def _preflight_streams(self, streams: list[ps.StreamConfig]) -> list[ps.StreamConfig]:
        """Return online-sync-capable streams, failing fast on group timeouts.

        A group timeout means the all-camera recorder did not make progress as
        a group. On Thor this can leave Argus/driver state fragile; immediately
        running N single-camera probes can turn a useful Connect failure into a
        long wait and a dirtier camera stack. If the recorder error names a
        camera, still drop that camera and retry. Otherwise, isolate only for
        non-timeout group failures.
        """

        candidates = list(streams)
        dropped: set[int] = set()

        def drop_stream(stream: ps.StreamConfig, message: str) -> None:
            if stream.sid in dropped:
                return
            dropped.add(stream.sid)
            self._errors.append(ps.StreamError(
                sid=stream.sid,
                name=stream.name,
                message=f"Argus online-sync preflight failed; dropping camera: {message}",
            ))
            logger.warning(
                "dropping %s from Argus online-sync session after preflight failure: %s",
                stream.name,
                message,
            )

        while candidates:
            try:
                self._run_preflight_for_streams(candidates)
                return candidates
            except Exception as group_exc:
                message = str(group_exc)
                if len(candidates) <= 1:
                    raise
                logger.warning(
                    "Argus online-sync preflight failed for candidate set %s: %s",
                    [stream.name for stream in candidates],
                    message,
                )

                failed_names = self._recorder_error_stream_names(message, candidates)
                named_failures = [
                    stream for stream in candidates if stream.name in failed_names
                ]
                if named_failures:
                    for stream in named_failures:
                        drop_stream(stream, message)
                    candidates = [
                        stream for stream in candidates
                        if stream.sid not in {failed.sid for failed in named_failures}
                    ]
                    continue

                if self._is_preflight_timeout(message):
                    raise RuntimeError(
                        "Argus online-sync group preflight timed out; "
                        "not running sequential single-camera isolation because "
                        "the all-camera Argus/driver stack may already be wedged. "
                        f"Candidate cameras: {','.join(stream.name for stream in candidates)}. "
                        f"Original error: {message}"
                    )

                passing: list[ps.StreamConfig] = []
                failing: list[tuple[ps.StreamConfig, str]] = []
                for stream in candidates:
                    try:
                        self._run_preflight_for_streams([stream])
                        passing.append(stream)
                    except Exception as exc:  # noqa: BLE001 - keep probing remaining cameras
                        failing.append((stream, str(exc)))

                if not passing:
                    details = "; ".join(
                        f"{stream.name}: {fail_message}"
                        for stream, fail_message in failing
                    )
                    raise RuntimeError(
                        f"Argus online-sync preflight failed for every camera: {details}"
                    )
                if not failing:
                    raise RuntimeError(
                        "Argus online-sync preflight failed for the camera group, "
                        "but every camera passed alone; cannot isolate a bad stream: "
                        f"{message}"
                    )
                for stream, fail_message in failing:
                    drop_stream(stream, fail_message)
                candidates = passing

        raise RuntimeError("Argus online-sync preflight left no usable cameras")

    def _run_preflight_for_streams(self, streams: list[ps.StreamConfig]) -> None:
        """Run a bounded online-sync preflight and kill wedged process groups.

        The inherited metadata preflight uses the global Connect timeout. That
        is too coarse for online-sync: a stuck all-camera group should return
        quickly so the caller can either isolate bad cameras or fail Connect
        with useful recorder output.
        """

        probe_dir = Path(tempfile.mkdtemp(prefix="lerobot_argus_online_sync_preflight_"))
        frames = self.preflight_frames
        cmd = self._build_record_command(
            streams, probe_dir, frames=frames, include_frame_bus=False
        )
        timeout_s = (
            self.single_preflight_timeout_s
            if len(streams) == 1 else self.preflight_timeout_s
        )
        logger.info(
            "running Argus online-sync preflight (timeout %.1fs): %s",
            timeout_s, " ".join(cmd),
        )
        try:
            output, rc, timed_out = self._run_preflight_process(
                cmd, timeout_s, cwd=self.repo_root,
            )
            tail = "; ".join(output.splitlines()[-10:])
            if timed_out:
                raise RuntimeError(
                    "Argus online-sync recorder preflight timed out "
                    f"after {timeout_s:.1f}s for "
                    f"{','.join(stream.name for stream in streams)}"
                    + (f": {tail}" if tail else "")
                )
            if rc != 0:
                raise RuntimeError(
                    f"Argus online-sync recorder preflight failed rc={rc}: {tail}"
                )

            short: list[str] = []
            for stream in streams:
                sidecar = probe_dir / f"{stream.name}.argus_frame_metadata.csv"
                count = self._sidecar_row_count(sidecar)
                if count < frames:
                    short.append(f"{stream.name} rows={count}/{frames}")
            if short:
                raise RuntimeError(
                    "Argus online-sync recorder preflight produced incomplete "
                    "sidecars: " + ", ".join(short)
                )
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    @staticmethod
    def _run_preflight_process(
        cmd: list[str],
        timeout_s: float,
        *,
        cwd: Path,
    ) -> tuple[str, int | None, bool]:
        proc = subprocess.Popen(
            cmd,
            text=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = proc.communicate(timeout=timeout_s)
            return output or "", proc.returncode, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                output, _ = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = proc.communicate(timeout=2.0)
            return output or "", proc.returncode, True

    def _reader_loop(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        done_re = re.compile(
            r"episode done idx=(?P<idx>\d+) ok=(?P<ok>true|false) "
            r"frames=(?P<frames>\d+)(?: failure=(?P<failure>.*))?$"
        )
        for line in proc.stdout:
            if self._reader_stop.is_set():
                break
            text = line.rstrip()
            if text:
                with self._lock:
                    self._recent_output.append(text)
                    self._recent_output = self._recent_output[-20:]
                logger.info("[argus-online-sync] %s", text)
                if text == "persistent ready":
                    self._persistent_ready_evt.set()
                if "recording started" in text:
                    self._recording_started_evt.set()
                match = done_re.search(text)
                if match:
                    result = {
                        "idx": int(match.group("idx")),
                        "ok": match.group("ok") == "true",
                        "frames": int(match.group("frames")),
                        "failure": match.group("failure") or "",
                    }
                    with self._lock:
                        self._episode_result = result
                    self._episode_done_evt.set()

    def start_episode(self, episode_dir: Path, idx: int) -> ps.EpisodeHandle:
        return self._start_episode_once(episode_dir, idx)

    def _drop_start_failure_error(self, message: str) -> None:
        with self._lock:
            self._errors = [
                error for error in self._errors
                if not (
                    error.sid == -1
                    and error.name == "argus_online_sync"
                    and error.message == message
                )
            ]

    @staticmethod
    def _is_transient_start_failure(message: str) -> bool:
        needles = (
            "Cannot create camera provider",
            "create CameraProvider failed",
            "Connection reset by peer",
            "Failed socket read",
            "Argus Error 0x00030003",
        )
        return any(needle in message for needle in needles)

    def _start_episode_once(self, episode_dir: Path, idx: int) -> ps.EpisodeHandle:
        if not self._active_sids:
            raise RuntimeError("start_episode() before connect()")
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("Argus online-sync persistent recorder is not running")
        self._recording_active = True
        self.disable_previews()
        episode_dir.mkdir(parents=True, exist_ok=True)
        self._recording_started_evt.clear()
        self._episode_done_evt.clear()
        with self._lock:
            self._episode_result = None
        try:
            self._send_daemon_command(f"START {int(idx)} {int(self.target_frames)} {episode_dir}")
        except Exception:
            self._recording_active = False
            raise
        deadline = time.monotonic() + max(1.0, self.connect_timeout_s)
        while not self._recording_started_evt.is_set():
            rc = proc.poll()
            if rc is not None:
                with self._lock:
                    recent = "; ".join(self._recent_output[-6:])
                    self._proc = None
                message = f"online-sync persistent recorder exited before recording started rc={rc}"
                if recent:
                    message = f"{message}: {recent}"
                self._errors.append(ps.StreamError(
                    sid=-1,
                    name="argus_online_sync",
                    message=message,
                ))
                self._recording_active = False
                raise RuntimeError(message)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not self._recording_started_evt.is_set():
            with self._lock:
                recent = "; ".join(self._recent_output[-6:])
            message = "online-sync recorder did not report recording started"
            if recent:
                message = f"{message}: {recent}"
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_online_sync",
                message=message,
            ))
            self._recording_active = False
            raise RuntimeError(message)
        t0_wall = time.time()
        t0_mono = time.monotonic()
        return ps.EpisodeHandle(
            idx=idx,
            directory=episode_dir,
            t0_wall_s=t0_wall,
            t0_mono_s=t0_mono,
        )

    def stop_episode(self, handle: ps.EpisodeHandle) -> ps.EpisodeHandle:
        handle.stop_wall_s = time.time()
        with self._lock:
            proc = self._proc
        if proc is None:
            rc = 0
        else:
            rc = proc.poll()

        if proc is not None and rc is None and not self._episode_done_evt.is_set():
            # Fixed-frame episodes normally complete by themselves. Give that
            # path a short grace window before sending STOP so automatic
            # duration-based captures keep the exact episode_time_s * 60 frame
            # count whenever the recorder is on pace.
            if self.target_frames > 0:
                self._episode_done_evt.wait(timeout=min(2.0, max(0.0, self.stop_timeout_s)))
            if not self._episode_done_evt.is_set():
                try:
                    self._send_daemon_command("STOP")
                except Exception as exc:
                    self._errors.append(ps.StreamError(
                        sid=-1,
                        name="argus_online_sync",
                        message=f"failed to send STOP to persistent recorder: {exc}",
                    ))

        deadline = time.monotonic() + max(1.0, self.stop_timeout_s)
        while not self._episode_done_evt.is_set():
            if proc is None:
                break
            rc = proc.poll()
            if rc is not None:
                break
            if time.monotonic() >= deadline:
                self._errors.append(ps.StreamError(
                    sid=-1,
                    name="argus_online_sync",
                    message="persistent recorder did not finish episode within timeout",
                ))
                self._terminate_proc(proc)
                with self._lock:
                    self._proc = None
                break
            time.sleep(0.05)

        with self._lock:
            result = dict(self._episode_result or {})
        if result and not result.get("ok", False):
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_online_sync",
                message=(
                    "persistent recorder episode failed: "
                    f"{result.get('failure') or 'unknown failure'}"
                ),
            ))

        rc = proc.poll() if proc is not None else 0
        if rc not in (0, None) and not result:
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_online_sync",
                message=f"persistent recorder exited with rc={rc}",
            ))
            with self._lock:
                self._proc = None
        self._recording_active = False

        for stream in self._stream_cfgs:
            suffix = ".mp4" if stream.container == "mp4" else ".mkv"
            path = handle.directory / f"{stream.name}{suffix}"
            handle.fragments[stream.name] = ps.FragmentInfo(
                sid=stream.sid,
                name=stream.name,
                fragment_id=handle.idx,
                path=path,
                first_pts_s=None,
                first_wall_s=handle.t0_wall_s,
                state=ps.FragmentState.EPISODE,
            )
        return handle

    def discard_episode(self, handle: ps.EpisodeHandle) -> None:
        super().discard_episode(handle)
        try:
            (handle.directory / "online_sync_manifest.json").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("failed to unlink online_sync_manifest.json: %s", exc)

    def disconnect(self) -> None:
        self.disable_previews()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                proc.wait(timeout=max(1.0, self.stop_timeout_s))
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._terminate_proc(proc)
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        self._recording_active = False
        self._active_sids = []

    def _preview_streams(self) -> list[ps.StreamConfig]:
        return [stream for stream in self._stream_cfgs if stream.preview_jpeg_path]

    def _preview_bridge_command(self) -> list[str]:
        script = Path(__file__).with_name("online_sync_preview_bridge.py")
        cmd = [
            sys.executable,
            str(script),
            "--frame-bus-dir",
            self.preview_frame_bus_dir,
            "--fps",
            str(ps.PREVIEW_FPS),
            "--preview-width",
            str(ps.PREVIEW_WIDTH),
        ]
        for stream in self._preview_streams():
            cmd.extend(["--camera", f"{stream.name}={stream.preview_jpeg_path}"])
        return cmd

    def _start_preview_bridge(self) -> None:
        proc = self._preview_bridge_proc
        if proc is not None and proc.poll() is None:
            return
        streams = self._preview_streams()
        if not self.preview_frame_bus_dir or not streams:
            return
        for stream in streams:
            if stream.preview_jpeg_path:
                path = Path(stream.preview_jpeg_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._preview_bridge_proc = subprocess.Popen(
            self._preview_bridge_command(),
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _stop_preview_bridge(self) -> None:
        proc = self._preview_bridge_proc
        self._preview_bridge_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    def enable_previews(self, *, stagger_s: float = 0.5) -> None:
        del stagger_s
        if self._recording_active:
            return
        if not self.preview_frame_bus_dir:
            if not self._preview_warning_emitted:
                logger.info(
                    "Argus online-sync preview requested but preview frame bus "
                    "is not configured"
                )
                self._preview_warning_emitted = True
            return
        if not self._preview_streams():
            return
        try:
            self._send_daemon_command("PREVIEW_ON")
            self._start_preview_bridge()
            self._preview_down_logged = False
        except Exception as exc:  # noqa: BLE001 - preview must not break recording
            try:
                self._send_daemon_command("PREVIEW_OFF")
            except Exception:
                pass
            self._stop_preview_bridge()
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_online_sync_preview",
                message=f"failed to enable online-sync preview: {exc}",
            ))

    def disable_previews(self) -> None:
        try:
            with self._lock:
                proc = self._proc
            if proc is not None and proc.poll() is None:
                self._send_daemon_command("PREVIEW_OFF")
        except Exception:
            pass
        self._stop_preview_bridge()
        for stream in self._preview_streams():
            if not stream.preview_jpeg_path:
                continue
            try:
                Path(stream.preview_jpeg_path).unlink(missing_ok=True)
            except OSError:
                pass

    def wait_preview_frames(self, *, timeout_s: float = 5.0) -> list[str]:
        pending = {
            stream.name: Path(stream.preview_jpeg_path)
            for stream in self._preview_streams()
            if stream.preview_jpeg_path
        }
        deadline = time.monotonic() + max(0.0, timeout_s)
        while pending and time.monotonic() < deadline:
            for name, path in list(pending.items()):
                try:
                    if path.exists() and path.stat().st_size > 0:
                        pending.pop(name, None)
                except OSError:
                    pass
            if pending:
                time.sleep(0.05)
        return sorted(pending)

    def refresh_stale_previews(self, *, max_age_s: float) -> list[str]:
        if self._recording_active:
            return []
        now = time.time()
        stale: list[str] = []
        for stream in self._preview_streams():
            if not stream.preview_jpeg_path:
                continue
            path = Path(stream.preview_jpeg_path)
            try:
                mtime = path.stat().st_mtime
                if now - mtime <= max_age_s:
                    continue
            except OSError:
                pass
            stale.append(stream.name)
        proc = self._preview_bridge_proc
        bridge_dead = proc is not None and proc.poll() is not None
        if not stale and not bridge_dead:
            self._preview_down_logged = False
            return []
        if stale and not self._preview_down_logged:
            logger.warning("online-sync preview stale for %s", ", ".join(stale))
            self._preview_down_logged = True
        self._stop_preview_bridge()
        self.enable_previews(stagger_s=0.0)
        return stale
