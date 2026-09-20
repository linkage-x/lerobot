"""Laser tracker as a recorder device, driven by Connect / Start Episode.

The tracker is not on Thor.  It hangs off a Windows box (DESKTOP-API) that owns
the vendor SDK, and the two machines are married afterwards by a
``QPC <-> CLOCK_MONOTONIC`` fit.  That is why this is an ssh driver and not a
device object: everything here runs `over there`, and the only thing Thor
contributes is the clock responder and, at the end, a place to put the files.

Lifecycle, deliberately the same shape as ``box_collection.BoxPool`` so the
recorder's episode loop treats it like any other sensor:

    start()                 Connect      responder up, probe running, link alive
    start_recording(...)    Start        one lt_realtime_logger per episode
    stop_recording()        Stop         stop that logger, count what it wrote
    stop(land_to=...)       Disconnect   stop probe, seal, land, verify

**Nothing here raises into the recorder.**  The tracker is a shared instrument
that is unavailable more often than not, and a laboratory that cannot record
nine cameras because a tracker in another room is warming up is a worse
laboratory.  Every entry point catches, records into ``last_error``, and returns
a falsy value; the recorder surfaces that as a warning and keeps going.

Two properties of the tools this drives shape the design, and both cost a
session before they were understood (see
``third_party/opencv_kalibr/metrology/laser_tracker/README.md``):

* the clock probe must **bracket every sample**, so it runs for the whole
  Connect..Disconnect window rather than per episode.  It streams to disk and
  stops on a stop-file, so an open-ended operator-driven session is fine and a
  crash costs one flush interval instead of the entire fit.
* ``lt_realtime_logger`` writes ``<out>/<session>.*`` flat and creates no
  directory, while ``session_manifest.py seal`` takes a directory and rglobs it.
  So ``--out`` is the session directory and ``--session`` is the episode name:
  one tracker session per Connect, one ``.rt.csv`` per episode inside it.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SSH_BASE = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new")


@dataclass
class LaserTrackerConfig:
    """Everything the driver needs to reach the capture PC and the tracker."""

    enabled: bool = False
    win_host: str = ""
    ssh_key: str = "~/.ssh/id_ed25519_lt"
    win_root: str = r"D:\lt"
    logger_exe: str = r"D:\sdk\collector_win\bin\lt_realtime_logger.exe"
    python: str = "python"
    tracker_ip: str = "192.168.0.168"
    thor_host: str = "192.168.111.122"
    responder_port: int = 17123
    responder_script: str = "third_party/opencv_kalibr/metrology/laser_tracker/thor_time_responder.sh"
    probe_interval_s: float = 0.01
    # Backstops only: the stop-file ends both in normal operation.  They exist so
    # a gateway that died without cleaning up cannot leave a process on the
    # capture PC forever.
    session_cap_s: float = 24 * 3600.0
    episode_cap_s: float = 3600.0
    land: bool = True
    note: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.win_host)


def config_from_yaml_dict(
    raw: dict[str, Any] | None, *, enabled_override: bool | None = None
) -> LaserTrackerConfig:
    """Build a config from the ``laser_tracker:`` block, if there is one.

    ``enabled_override`` is how the GUI's per-session toggle wins over the
    file: the yaml declares the hardware (addresses, paths) and defaults to
    off, and the operator decides per Connect whether this session carries the
    tracker.  A toggle that could not turn it *off* would be useless on the days
    the instrument is booked by someone else.
    """
    cfg = LaserTrackerConfig()
    if isinstance(raw, dict):
        for key, value in raw.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, type(getattr(cfg, key))(value) if not isinstance(value, bool) else value)
    if enabled_override is not None:
        cfg.enabled = bool(enabled_override)
    if cfg.enabled and not cfg.configured:
        logger.warning("laser_tracker enabled but win_host is empty; treating as disabled")
        cfg.enabled = False
    return cfg


@dataclass
class EpisodeRecord:
    """What one episode's tracker capture produced, for the episode meta."""

    episode_index: int
    session_name: str = ""
    started: bool = False
    rt_rows: int = 0
    error: str = ""
    t_start_wall_s: float = 0.0


@dataclass
class LaserTrackerStatus:
    """Everything the GUI needs to draw one device row honestly."""

    enabled: bool = False
    connected: bool = False
    session_id: str = ""
    win_host: str = ""
    win_session_dir: str = ""
    probe_running: bool = False
    sync_rows: int = 0
    last_error: str = ""
    episodes: list[dict[str, Any]] = field(default_factory=list)


class LaserTrackerSession:
    """Drives the capture PC over ssh for one Connect..Disconnect window."""

    def __init__(self, cfg: LaserTrackerConfig, *, session_id: str | None = None) -> None:
        self.cfg = cfg
        self.session_id = session_id or datetime.now(timezone.utc).strftime("lt_%Y%m%d_%H%M%S")
        self.last_error = ""
        self._connected = False
        self._probe: subprocess.Popen[str] | None = None
        self._episode: EpisodeRecord | None = None
        self._episodes: list[EpisodeRecord] = []
        self._logger_proc: subprocess.Popen[str] | None = None

    # ---------------------------------------------------------------- ssh --

    @property
    def win_dir(self) -> str:
        return f"{self.cfg.win_root}\\{self.session_id}"

    @property
    def probe_stop_file(self) -> str:
        return f"{self.win_dir}\\STOP_PROBE"

    @property
    def episode_stop_file(self) -> str:
        return f"{self.win_dir}\\STOP_EPISODE"

    def _ssh_argv(self) -> list[str]:
        key = str(Path(self.cfg.ssh_key).expanduser())
        return ["ssh", "-i", key, *_SSH_BASE, self.cfg.win_host]

    def _run(self, remote_cmd: str, *, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self._ssh_argv(), remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def _spawn(self, remote_cmd: str) -> subprocess.Popen[str]:
        """Hold the remote process on its own ssh channel.

        Backgrounding on the Windows side would need detachment and later
        hunting; keeping the channel means the process dies with it, and the
        local handle is the liveness signal.
        """
        return subprocess.Popen(
            [*self._ssh_argv(), remote_cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    # ------------------------------------------------------------ connect --

    def start(self) -> bool:
        """Connect: responder up, session directory made, probe streaming."""
        if not self.cfg.enabled:
            return False
        try:
            self._ensure_responder()
            probe = self._start_probe()
            if not probe:
                return False
            self._connected = True
            self.last_error = ""
            logger.info("laser tracker session %s connected (%s)", self.session_id, self.win_dir)
            return True
        except Exception as exc:  # never break Connect for the other devices
            self.last_error = f"connect failed: {exc}"
            logger.warning("laser tracker connect failed: %s", exc)
            self._teardown_probe()
            return False

    def _ensure_responder(self) -> None:
        """Start Thor's clock responder if it is down.

        Per boot, not per session: ``CLOCK_MONOTONIC`` restarts at reboot and the
        offline fit refuses to cross one, so a responder from before the last
        reboot is not merely stale, it is the wrong clock.  Nothing downstream
        notices it is gone until the probe writes an empty file, which is how
        2026-09-18's reboot cost 2026-09-20's morning.
        """
        script = Path(__file__).resolve().parents[3] / self.cfg.responder_script
        if not script.exists():
            raise RuntimeError(f"responder script not found: {script}")
        subprocess.run([str(script), "start"], capture_output=True, text=True, timeout=30)

    def _start_probe(self) -> bool:
        out = f"{self.win_dir}\\{self.session_id}.sync.csv"
        mk = self._run(f'if not exist "{self.win_dir}" mkdir "{self.win_dir}"')
        if mk.returncode != 0:
            self.last_error = f"cannot create {self.win_dir}: {(mk.stderr or mk.stdout).strip()}"
            return False
        self._run(f'del /q "{self.probe_stop_file}"')  # a stale one would stop us at once
        cmd = (
            f'cd /d {self.cfg.win_root} && {self.cfg.python} probe_time_link.py '
            f'--host {self.cfg.thor_host} --port {self.cfg.responder_port} '
            f'--out "{out}" --stop-file "{self.probe_stop_file}" '
            f'--interval {self.cfg.probe_interval_s:g} --duration {self.cfg.session_cap_s:g}'
        )
        self._probe = self._spawn(cmd)
        # The probe is its own link check: if rows are landing, the responder is
        # up, UDP is open and the path works.  A separate 30 s --check would just
        # make Connect slower to learn the same thing.
        time.sleep(2.0)
        if self._probe.poll() is not None:
            self.last_error = f"probe exited immediately: {(self._probe.stdout.read() if self._probe.stdout else '').strip()[:300]}"
            self._probe = None
            return False
        rows = self.sync_rows()
        if rows <= 0:
            self.last_error = "clock probe is producing no exchanges -- responder down or UDP blocked"
            self._teardown_probe()
            return False
        logger.info("clock probe alive: %d exchanges in the first 2 s", rows)
        return True

    def sync_rows(self) -> int:
        return self._count_rows(f"{self.session_id}.sync.csv")

    def _count_rows(self, filename: str) -> int:
        code = "import sys;print(sum(1 for _ in open(sys.argv[1]))-1)"
        res = self._run(f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{code}" {filename}', timeout_s=20)
        try:
            return int((res.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return -1

    # ------------------------------------------------------------ episode --

    def start_recording(self, episode_index: int, t_start_wall_s: float) -> bool:
        """Start Episode: one logger run, named after the episode."""
        if not self._connected:
            return False
        name = f"episode_{episode_index:06d}"
        rec = EpisodeRecord(episode_index=episode_index, session_name=name, t_start_wall_s=t_start_wall_s)
        try:
            self._run(f'del /q "{self.episode_stop_file}"')
            note = self.cfg.note or f"{name} @ {datetime.now(timezone.utc).isoformat()}"
            cmd = (
                f'cd /d {self.cfg.win_root} && "{self.cfg.logger_exe}" '
                f'--ip {self.cfg.tracker_ip} --out "{self.win_dir}" --session {name} '
                f'--note "{note}" --duration {self.cfg.episode_cap_s:g} '
                f'--stop-file "{self.episode_stop_file}"'
            )
            self._logger_proc = self._spawn(cmd)
            time.sleep(0.5)
            if self._logger_proc.poll() is not None:
                out = (self._logger_proc.stdout.read() if self._logger_proc.stdout else "").strip()
                # The commonest cause by far: the tracker admits one client and
                # SA is holding it.  Say so rather than printing an SDK ordinal.
                rec.error = f"logger exited at once: {out[:300]}"
                self.last_error = rec.error
                self._logger_proc = None
            else:
                rec.started = True
        except Exception as exc:
            rec.error = f"start failed: {exc}"
            self.last_error = rec.error
        self._episode = rec
        return rec.started

    def stop_recording(self) -> dict[str, Any]:
        """Stop Episode: stop-file, wait for the logger, count what landed."""
        rec = self._episode
        self._episode = None
        if rec is None:
            return {}
        if rec.started:
            try:
                self._run(f'type nul > "{self.episode_stop_file}"')
                if self._logger_proc is not None:
                    try:
                        self._logger_proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self._logger_proc.terminate()
                        rec.error = "logger did not stop within 30 s; terminated"
                        self.last_error = rec.error
                rec.rt_rows = self._count_rows(f"{rec.session_name}.rt.csv")
                if rec.rt_rows <= 0:
                    rec.error = rec.error or "tracker stream is empty for this episode"
                    self.last_error = rec.error
            except Exception as exc:
                rec.error = f"stop failed: {exc}"
                self.last_error = rec.error
            finally:
                self._logger_proc = None
        self._episodes.append(rec)
        return asdict(rec)

    # --------------------------------------------------------- disconnect --

    def stop(self, *, land_to: Path | None = None) -> dict[str, Any]:
        """Disconnect: stop the probe, seal, and land the session."""
        result: dict[str, Any] = {"session_id": self.session_id, "landed_to": "", "sealed": False}
        if not self._connected and self._probe is None:
            return result
        try:
            if self._episode is not None:
                self.stop_recording()  # an exit mid-episode still gets its rows counted
            self._teardown_probe()
            result["sync_rows"] = self.sync_rows()
            seal = self._run(
                f'cd /d {self.cfg.win_root} && {self.cfg.python} session_manifest.py seal "{self.win_dir}"',
                timeout_s=300,
            )
            result["sealed"] = seal.returncode == 0
            result["seal_output"] = (seal.stdout or seal.stderr or "").strip()[:500]
            if not result["sealed"]:
                self.last_error = f"seal refused: {result['seal_output']}"
            if self.cfg.land and land_to is not None:
                result["landed_to"] = self._land(land_to)
        except Exception as exc:
            self.last_error = f"stop failed: {exc}"
            logger.warning("laser tracker stop failed: %s", exc)
        finally:
            self._connected = False
        result["episodes"] = [asdict(r) for r in self._episodes]
        result["last_error"] = self.last_error
        return result

    def _teardown_probe(self) -> None:
        if self._probe is None:
            return
        try:
            self._run(f'type nul > "{self.probe_stop_file}"')
            try:
                self._probe.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Survivable now, unlike before: the probe streams, so whatever
                # reached disk is a valid file.  It was not always so.
                self._probe.terminate()
                self.last_error = "clock probe did not stop within 30 s; terminated"
        except Exception as exc:
            self.last_error = f"probe teardown failed: {exc}"
        finally:
            self._probe = None

    def _land(self, dest_root: Path) -> str:
        """Copy the session back to Thor and verify it against its manifest."""
        dest_root.mkdir(parents=True, exist_ok=True)
        src = f"{self.cfg.win_host}:{self.win_dir.replace(chr(92), '/')}"
        key = str(Path(self.cfg.ssh_key).expanduser())
        # scp, not rsync: Windows OpenSSH ships no rsync server and the failure
        # is an unhelpful "connection unexpectedly closed".
        res = subprocess.run(
            ["scp", "-i", key, *_SSH_BASE, "-r", src, str(dest_root)],
            capture_output=True, text=True, timeout=1800,
        )
        if res.returncode != 0:
            self.last_error = f"landing failed: {(res.stderr or res.stdout).strip()[:300]}"
            return ""
        landed = dest_root / self.session_id
        verifier = Path(__file__).resolve().parents[3] / (
            "third_party/opencv_kalibr/metrology/laser_tracker/session_manifest.py"
        )
        if verifier.exists():
            check = subprocess.run(
                ["python3", str(verifier), "verify", str(landed)],
                capture_output=True, text=True, timeout=600,
            )
            if check.returncode != 0:
                # A truncated copy fails verify rather than being analysed --
                # that is the whole point of sealing before moving.
                self.last_error = f"landed copy does not match its manifest: {(check.stdout or check.stderr).strip()[:300]}"
        return str(landed)

    # ------------------------------------------------------------- status --

    def status(self) -> LaserTrackerStatus:
        return LaserTrackerStatus(
            enabled=self.cfg.enabled,
            connected=self._connected,
            session_id=self.session_id,
            win_host=self.cfg.win_host,
            win_session_dir=self.win_dir,
            probe_running=self._probe is not None and self._probe.poll() is None,
            sync_rows=0,
            last_error=self.last_error,
            episodes=[asdict(r) for r in self._episodes],
        )
