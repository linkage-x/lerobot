"""Drop-in Thor recorder session for encoder-front online synchronization."""

from __future__ import annotations

import logging
import os
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from tools.thor.gmsl2 import argus_metadata_session as ams
from tools.thor.gmsl2 import persistent_session as ps

logger = logging.getLogger("argus_online_sync_session")


DEFAULT_BINARY_PATH = Path("/tmp/lerobot_argus_online_sync_video_recorder")


class ArgusOnlineSyncCameraSession(ams.ArgusMetadataCameraSession):
    """Session wrapper around ``argus_online_sync_video_recorder``.

    It intentionally reuses the metadata session's process lifecycle and
    preflight/drop-bad-camera behavior, but the recorder itself aligns frames
    before hardware encoding. Therefore fixed-duration episodes pass a finite
    frame target to the C++ process instead of relying on post-save slicing.
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
        self.start_retry_settle_s = 2.0
        self.start_retries = 1

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
        return cmd

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
        cmd = self._build_record_command(streams, probe_dir, frames=frames)
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
        for line in proc.stdout:
            if self._reader_stop.is_set():
                break
            text = line.rstrip()
            if text:
                with self._lock:
                    self._recent_output.append(text)
                    self._recent_output = self._recent_output[-20:]
                logger.info("[argus-online-sync] %s", text)
                if "recording started" in text:
                    self._recording_started_evt.set()

    def start_episode(self, episode_dir: Path, idx: int) -> ps.EpisodeHandle:
        last_exc: RuntimeError | None = None
        for attempt in range(self.start_retries + 1):
            try:
                return self._start_episode_once(episode_dir, idx)
            except RuntimeError as exc:
                last_exc = exc
                message = str(exc)
                if attempt >= self.start_retries or not self._is_transient_start_failure(message):
                    raise
                logger.warning(
                    "Argus online-sync recorder start failed with a transient "
                    "Argus provider error; retrying once after %.1fs: %s",
                    self.start_retry_settle_s,
                    message,
                )
                self._drop_start_failure_error(message)
                time.sleep(self.start_retry_settle_s)
        assert last_exc is not None
        raise last_exc

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
        if self._proc and self._proc.poll() is None:
            raise RuntimeError("previous Argus online-sync recorder is still running")
        self._recording_active = True
        self.disable_previews()
        episode_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_record_command(
            self._stream_cfgs,
            episode_dir,
            frames=self.target_frames,
        )
        self._recording_started_evt.clear()
        with self._lock:
            self._recent_output = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except Exception:
            self._recording_active = False
            raise
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(proc,),
            name="argus-online-sync-recorder-log",
            daemon=True,
        )
        self._reader_thread.start()
        with self._lock:
            self._proc = proc
        deadline = time.monotonic() + max(1.0, self.connect_timeout_s)
        while not self._recording_started_evt.is_set():
            rc = proc.poll()
            if rc is not None:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=1.0)
                    self._reader_thread = None
                with self._lock:
                    recent = "; ".join(self._recent_output[-6:])
                    self._proc = None
                message = f"online-sync recorder exited before recording started rc={rc}"
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
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=1.0)
                self._reader_thread = None
            with self._lock:
                self._proc = None
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

    def discard_episode(self, handle: ps.EpisodeHandle) -> None:
        super().discard_episode(handle)
        try:
            (handle.directory / "online_sync_manifest.json").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("failed to unlink online_sync_manifest.json: %s", exc)
