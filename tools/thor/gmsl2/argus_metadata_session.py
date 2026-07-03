"""Drop-in Thor recorder session backed by Libargus frame metadata.

The GUI-facing recorder loop talks to a session object with the same small API
as :class:`persistent_session.PersistentCameraSession`: connect once, then
start/stop/discard episodes.  This module keeps that API but replaces the
per-camera ``nvarguscamerasrc`` workers with one Libargus recorder process that
owns all cameras and writes both video and per-frame metadata sidecars.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import threading
import time
import tempfile
from pathlib import Path
from typing import Any

from tools.thor.gmsl2 import persistent_session as ps

logger = logging.getLogger("argus_metadata_session")


DEFAULT_BINARY_PATH = Path("/tmp/lerobot_argus_metadata_video_recorder")


class ArgusMetadataCameraSession:
    """Session-compatible wrapper around ``argus_metadata_video_recorder``."""

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
        stop_timeout_s: float = 10.0,
        preflight_frames: int = 2,
    ):
        self._stream_cfgs = list(streams)
        self.warmup_dir = warmup_dir
        self.repo_root = Path(repo_root)
        self.binary_path = Path(binary_path or DEFAULT_BINARY_PATH)
        self.auto_build = bool(auto_build)
        self.connect_timeout_s = float(connect_timeout_s)
        self.connect_stable_s = max(0.0, float(connect_stable_s))
        self.stop_timeout_s = float(stop_timeout_s)
        self.preflight_frames = max(0, int(preflight_frames))
        self.connect_duration_s = 0.0
        self._active_sids: list[int] = []
        self._errors: list[ps.StreamError] = []
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._recording_started_evt = threading.Event()
        self._recent_output: list[str] = []
        self._preview_procs: dict[str, subprocess.Popen[bytes]] = {}
        self._recording_active = False
        self._lock = threading.Lock()

    @property
    def active_sids(self) -> list[int]:
        return list(self._active_sids)

    def _recorder_source(self) -> Path:
        return Path(__file__).with_name("argus_metadata_video_recorder.cpp")

    def _needs_build(self) -> bool:
        if not self.binary_path.exists():
            return True
        try:
            return self.binary_path.stat().st_mtime < self._recorder_source().stat().st_mtime
        except OSError:
            return True

    def _build_binary(self) -> None:
        self.binary_path.parent.mkdir(parents=True, exist_ok=True)
        src = self._recorder_source()
        cmd = (
            "g++ -std=c++14 -O2 "
            "-I/usr/src/jetson_multimedia_api/argus/include "
            "-I/usr/src/jetson_multimedia_api/argus/samples/utils "
            "$(pkg-config --cflags gstreamer-1.0 glib-2.0) "
            f"{src} "
            "/usr/src/jetson_multimedia_api/argus/samples/utils/ArgusHelpers.cpp "
            "-L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient "
            "$(pkg-config --libs gstreamer-1.0 glib-2.0) "
            f"-lEGL -lGLESv2 -lpthread -o {self.binary_path}"
        )
        logger.info("building Argus metadata recorder: %s", self.binary_path)
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
                "failed to build argus_metadata_video_recorder "
                f"(rc={result.returncode}): {detail}"
            )

    def connect(self) -> None:
        if self._active_sids:
            raise RuntimeError("connect() called twice without disconnect()")
        t0 = time.time()
        if self.auto_build and self._needs_build():
            self._build_binary()
        if not self.binary_path.exists():
            raise RuntimeError(f"Argus metadata recorder binary not found: {self.binary_path}")
        probe = subprocess.run(
            [str(self.binary_path), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1.0, self.connect_timeout_s),
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"Argus metadata recorder probe failed rc={probe.returncode}: "
                f"{(probe.stdout or '').strip()}"
            )
        if self.preflight_frames > 0:
            self._stream_cfgs = self._preflight_streams(self._stream_cfgs)
        self._active_sids = sorted(stream.sid for stream in self._stream_cfgs)
        self.warmup_dir.mkdir(parents=True, exist_ok=True)
        if self.connect_stable_s > 0:
            time.sleep(self.connect_stable_s)
        self.connect_duration_s = time.time() - t0

    def disconnect(self) -> None:
        self.disable_previews()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        self._recording_active = False
        self._active_sids = []

    def poll_errors(self) -> list[ps.StreamError]:
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        return errors

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
                logger.info("[argus-metadata] %s", text)
                if "recording started" in text:
                    self._recording_started_evt.set()

    def _common_stream_value(self, field: str, streams: list[ps.StreamConfig] | None = None) -> Any:
        streams = self._stream_cfgs if streams is None else streams
        values = {getattr(stream, field) for stream in streams}
        if len(values) != 1:
            raise ValueError(f"Argus metadata recorder requires one {field}, got {sorted(values)}")
        return next(iter(values))

    @staticmethod
    def _name_prefix_for_streams(streams: list[ps.StreamConfig]) -> str:
        prefixes: set[str] = set()
        for stream in streams:
            suffix = f"_{stream.sid:02d}"
            if not stream.name.endswith(suffix):
                raise ValueError(
                    "Argus metadata recorder requires stream names to end with "
                    f"{suffix!r}, got {stream.name!r}"
                )
            prefixes.add(stream.name[:-len(suffix)])
        if len(prefixes) != 1:
            raise ValueError(
                "Argus metadata recorder requires one name_prefix, got "
                f"{sorted(prefixes)}"
            )
        return next(iter(prefixes))

    def _build_record_command(
        self,
        streams: list[ps.StreamConfig],
        episode_dir: Path,
        *,
        frames: int,
    ) -> list[str]:
        codec = str(self._common_stream_value("codec", streams))
        container = str(self._common_stream_value("container", streams))
        cmd = [
            str(self.binary_path),
            "--sids",
            ",".join(str(stream.sid) for stream in sorted(streams, key=lambda s: s.sid)),
            "--frames",
            str(int(frames)),
            "--episode-dir",
            str(episode_dir),
            "--fps",
            str(int(self._common_stream_value("fps", streams))),
            "--codec",
            codec,
            "--container",
            container,
            "--bitrate",
            str(int(self._common_stream_value("bitrate_kbps", streams)) * 1000),
            "--iframe-interval",
            str(int(self._common_stream_value("iframe_interval", streams))),
            "--preset-level",
            str(int(self._common_stream_value("preset_level", streams))),
            "--control-rate",
            str(int(self._common_stream_value("control_rate", streams))),
            "--sensor-mode",
            str(int(self._common_stream_value("sensor_mode", streams))),
            "--name-prefix",
            self._name_prefix_for_streams(streams),
        ]
        return cmd

    @staticmethod
    def _sidecar_row_count(path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0

    def _preflight_streams(self, streams: list[ps.StreamConfig]) -> list[ps.StreamConfig]:
        """Return streams that can satisfy the production metadata contract.

        Verify the full set because the real recorder owns all cameras in one
        process. If the recorder points at a specific failed camera, drop that
        camera and retry the survivors. If it cannot name a camera, run the
        same recorder per camera to identify link-locked but no-frame cameras.
        This keeps the GUI usable when detected cameras are bad while still
        rejecting failures that cannot be isolated to particular streams.
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
                message=f"Argus metadata preflight failed; dropping camera: {message}",
            ))
            logger.warning(
                "dropping %s from Argus metadata session after preflight failure: %s",
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
                    "Argus metadata preflight failed for candidate set %s: %s",
                    [stream.name for stream in candidates],
                    message,
                )

                named_failures = [
                    stream for stream in candidates if stream.name in message
                ]
                if named_failures:
                    for stream in named_failures:
                        drop_stream(stream, message)
                    candidates = [
                        stream for stream in candidates
                        if stream.sid not in {failed.sid for failed in named_failures}
                    ]
                    continue

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
                        f"Argus metadata preflight failed for every camera: {details}"
                    )
                if not failing:
                    raise RuntimeError(
                        "Argus metadata preflight failed for the camera group, "
                        "but every camera passed alone; cannot isolate a bad stream: "
                        f"{message}"
                    )
                for stream, fail_message in failing:
                    drop_stream(stream, fail_message)
                candidates = passing

        raise RuntimeError("Argus metadata preflight left no usable cameras")

    def _run_preflight_for_streams(self, streams: list[ps.StreamConfig]) -> None:
        """Run the real metadata recorder briefly during Connect.

        The old nvarguscamerasrc probe can prove that Argus can open a sensor,
        but the production contract also requires the metadata consumer and
        video branch to advance together. This catches link-locked cameras that
        do not deliver frame metadata before the UI reports Connected.
        """

        probe_dir = Path(tempfile.mkdtemp(prefix="lerobot_argus_metadata_preflight_"))
        frames = self.preflight_frames
        cmd = self._build_record_command(streams, probe_dir, frames=frames)
        logger.info("running Argus metadata preflight: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(10.0, self.connect_timeout_s),
                check=False,
            )
            output = (result.stdout or "").strip()
            if result.returncode != 0:
                tail = "; ".join(output.splitlines()[-8:])
                raise RuntimeError(
                    f"Argus metadata recorder preflight failed rc={result.returncode}: {tail}"
                )

            short: list[str] = []
            for stream in streams:
                sidecar = probe_dir / f"{stream.name}.argus_frame_metadata.csv"
                count = self._sidecar_row_count(sidecar)
                if count < frames:
                    short.append(f"{stream.name} rows={count}/{frames}")
            if short:
                raise RuntimeError(
                    "Argus metadata recorder preflight produced incomplete sidecars: "
                    + ", ".join(short)
                )
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def start_episode(self, episode_dir: Path, idx: int) -> ps.EpisodeHandle:
        if not self._active_sids:
            raise RuntimeError("start_episode() before connect()")
        if self._proc and self._proc.poll() is None:
            raise RuntimeError("previous Argus metadata recorder is still running")
        # Idle preview uses separate nvarguscamerasrc clients. Stop them before
        # the metadata recorder opens Libargus sessions for the episode.
        self._recording_active = True
        self.disable_previews()
        episode_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_record_command(self._stream_cfgs, episode_dir, frames=0)
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
            name="argus-metadata-recorder-log",
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
                message = f"recorder exited before recording started rc={rc}"
                if recent:
                    message = f"{message}: {recent}"
                self._errors.append(ps.StreamError(
                    sid=-1,
                    name="argus_metadata",
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
            message = "recorder did not report recording started"
            if recent:
                message = f"{message}: {recent}"
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_metadata",
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
            self._proc = None
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
                self._errors.append(ps.StreamError(
                    sid=-1,
                    name="argus_metadata",
                    message="recorder did not stop within timeout",
                ))
        rc = proc.returncode if proc is not None else 0
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if rc not in (0, None):
            self._errors.append(ps.StreamError(
                sid=-1,
                name="argus_metadata",
                message=f"recorder exited with rc={rc}",
            ))
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
        for info in handle.fragments.values():
            try:
                info.path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("failed to unlink %s: %s", info.path, exc)
            sidecar = handle.directory / f"{info.name}.argus_frame_metadata.csv"
            try:
                sidecar.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("failed to unlink %s: %s", sidecar, exc)

    def enable_previews(self, *, stagger_s: float = 0.5) -> None:
        if self._recording_active:
            return None
        for i, stream in enumerate(self._stream_cfgs):
            if not stream.preview_jpeg_path:
                continue
            if i > 0 and stagger_s > 0:
                time.sleep(stagger_s)
            self._ensure_preview(stream)
        return None

    def disable_previews(self) -> None:
        with self._lock:
            procs = list(self._preview_procs.values())
            self._preview_procs.clear()
        for proc in procs:
            self._terminate_preview_proc(proc)
        for stream in self._stream_cfgs:
            if not stream.preview_jpeg_path:
                continue
            try:
                Path(stream.preview_jpeg_path).unlink(missing_ok=True)
            except OSError:
                pass

    def wait_preview_frames(self, *, timeout_s: float = 5.0) -> list[str]:
        deadline = time.time() + max(0.0, timeout_s)
        missing: set[str] = {
            stream.name for stream in self._stream_cfgs if stream.preview_jpeg_path
        }
        while missing and time.time() < deadline:
            for stream in self._stream_cfgs:
                if stream.name not in missing or not stream.preview_jpeg_path:
                    continue
                path = Path(stream.preview_jpeg_path)
                try:
                    if path.stat().st_size > 0:
                        missing.discard(stream.name)
                except OSError:
                    pass
            if missing:
                time.sleep(0.1)
        return sorted(missing)

    def refresh_stale_previews(self, *, max_age_s: float) -> list[str]:
        if self._recording_active:
            return []
        restarted: list[str] = []
        now = time.time()
        for stream in self._stream_cfgs:
            if not stream.preview_jpeg_path:
                continue
            proc = self._preview_procs.get(stream.name)
            stale = proc is None or proc.poll() is not None
            path = Path(stream.preview_jpeg_path)
            try:
                stale = stale or path.stat().st_size <= 0 or now - path.stat().st_mtime > max_age_s
            except OSError:
                stale = True
            if stale:
                self._stop_preview(stream.name)
                self._ensure_preview(stream)
                restarted.append(stream.name)
        return restarted

    def roll_warmup(self) -> None:
        return None

    def cleanup_warmup_files(self, *, keep_last_n: int = 3) -> int:
        return 0

    @staticmethod
    def _preview_command(stream: ps.StreamConfig, output_path: Path) -> list[str]:
        output_width = ps.PREVIEW_WIDTH
        output_height = max(1, round(output_width * stream.height / max(stream.width, 1)))
        caps_in = (
            "video/x-raw(memory:NVMM),"
            f"format=NV12,width={stream.width},height={stream.height},framerate={stream.fps}/1"
        )
        caps_out = f"video/x-raw,format=I420,width={output_width},height={output_height}"
        return [
            "gst-launch-1.0",
            "-q",
            "nvarguscamerasrc",
            f"sensor-id={stream.sid}",
            f"sensor-mode={stream.sensor_mode}",
            "do-timestamp=true",
            "!",
            caps_in,
            "!",
            "nvvidconv",
            "!",
            caps_out,
            "!",
            "videorate",
            "!",
            f"video/x-raw,framerate={ps.PREVIEW_FPS}/1",
            "!",
            "jpegenc",
            "quality=65",
            "!",
            "multifilesink",
            f"location={output_path}",
            "max-files=1",
        ]

    @staticmethod
    def _terminate_preview_proc(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)

    def _stop_preview(self, name: str) -> None:
        with self._lock:
            proc = self._preview_procs.pop(name, None)
        if proc is not None:
            self._terminate_preview_proc(proc)

    def _ensure_preview(self, stream: ps.StreamConfig) -> None:
        if self._recording_active or not stream.preview_jpeg_path:
            return
        with self._lock:
            existing = self._preview_procs.get(stream.name)
            if existing is not None and existing.poll() is None:
                return
        path = Path(stream.preview_jpeg_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._preview_command(stream, path)
        proc = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._lock:
            self._preview_procs[stream.name] = proc
