"""Laser tracker as a recorder device, driven by Connect / Start Episode.

The tracker is not on Thor.  It hangs off a Windows box (DESKTOP-API) that owns
the vendor SDK, and the two machines are married afterwards by a
``QPC <-> CLOCK_MONOTONIC`` fit.  That is why this is an ssh driver and not a
device object: everything here runs `over there`, and the only thing Thor
contributes is the clock responder and, at the end, a place to put the files.

Lifecycle, deliberately the same shape as ``box_collection.BoxPool`` so the
recorder's episode loop treats it like any other sensor:

    start()                 Connect      responder + clock probe + tracker stream
    start_recording(...)    Start        mark this episode's boundary in the stream
    stop_recording()        Stop         close the boundary, check the stream is alive
    stop(land_to=...)       Disconnect   stop both, seal, land, verify

**One logger per Connect, not per episode.**  Measured on DESKTOP-API
2026-09-20: `lt_realtime_logger` takes **15-16 s** from launch to its first
sample, almost all of it inside the SDK's `Connect()` handshake (reproduced
twice: 16 s and 15 s).  Paying that per episode would either block Start Episode
for a quarter of a minute or silently drop the first 16 s of every episode's
tracker data, and a GT session cannot afford either.  So the stream runs for the
whole Connect window and Start/Stop Episode record boundaries into it.

That is also the honest shape.  The 1 kHz stream is stamped by the controller
and only becomes comparable to camera frames through the offline
`QPC <-> CLOCK_MONOTONIC` fit, so slicing it per episode is an offline operation
either way -- doing it at record time would buy nothing and cost 16 s.

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

import collections
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)

_SSH_BASE = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new")


def _decode(raw: bytes) -> str:
    """Decode whatever the Windows console said, without ever raising.

    cmd.exe answers in the system OEM codepage, not UTF-8: on a Chinese install
    `del` on a missing file replies 找不到文件, whose first byte is 0xD5, and
    ``text=True`` then dies with "'utf-8' codec can't decode byte 0xd5".  That
    turned a routine "the stop-file was not there" into a failed Connect.

    latin-1 cannot fail, so the chain always terminates: mojibake in a log line
    is a far better outcome than losing the session it was describing.
    """
    for encoding in ("utf-8", "cp936", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _default_repo_root() -> Path:
    """Walk up until the tree looks like the repo, rather than counting levels.

    ``parents[3]`` was right until this file was read from somewhere else, and
    then it raised ``IndexError(3)`` -- whose entire string form is ``3``, so the
    operator's warning read "laser tracker unavailable: connect failed: 3".
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "third_party").is_dir() or (parent / ".git").exists():
            return parent
    return here.parent


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
    # Generous: the SDK handshake alone measured 15-16 s.
    logger_ready_timeout_s: float = 60.0
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
    started: bool = False
    t_start_wall_s: float = 0.0
    t_end_wall_s: float = 0.0
    # Total rows the ONE session-long stream had written when this episode
    # ended -- not this episode's count, which is an offline slice by timestamp.
    # Named so nobody reads it as the latter.
    rt_rows_total_at_stop: int = 0
    stream_advanced: bool = True
    # -1 when it could not be read; 0.0 is a real answer and a bad one.
    beam_valid_fraction: float = -1.0
    beam_tracking_fraction: float = -1.0
    error: str = ""


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
    # Read from the instrument, never from config: see WriteMeta in
    # collector_win/lt_realtime_logger.cpp.
    device: dict[str, str] = field(default_factory=dict)
    beam_valid_fraction: float = -1.0
    beam_tracking_fraction: float = -1.0
    episodes: list[dict[str, Any]] = field(default_factory=list)


class LaserTrackerSession:
    """Drives the capture PC over ssh for one Connect..Disconnect window."""

    def __init__(
        self,
        cfg: LaserTrackerConfig,
        *,
        session_id: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
        self.session_id = session_id or datetime.now(timezone.utc).strftime("lt_%Y%m%d_%H%M%S")
        self.last_error = ""
        self._connected = False
        self._probe: subprocess.Popen[str] | None = None
        self._episode: EpisodeRecord | None = None
        self._episodes: list[EpisodeRecord] = []
        self._rows_mark = 0
        self.device_info: dict[str, str] = {}
        self.beam_valid_fraction = -1.0
        self.beam_tracking_fraction = -1.0
        self._logger_tail: collections.deque[str] = collections.deque(maxlen=40)
        self._logger_proc: subprocess.Popen[str] | None = None

    # ---------------------------------------------------------------- ssh --

    @property
    def win_dir(self) -> str:
        return f"{self.cfg.win_root}\\{self.session_id}"

    @property
    def probe_stop_file(self) -> str:
        return f"{self.win_dir}\\STOP_PROBE"

    @property
    def logger_stop_file(self) -> str:
        return f"{self.win_dir}\\STOP_LOGGER"

    def _ssh_argv(self) -> list[str]:
        key = str(Path(self.cfg.ssh_key).expanduser())
        return ["ssh", "-i", key, *_SSH_BASE, self.cfg.win_host]

    def _run(self, remote_cmd: str, *, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
        # Captured as bytes and decoded here rather than text=True: see _decode.
        raw = subprocess.run(
            [*self._ssh_argv(), remote_cmd],
            capture_output=True,
            timeout=timeout_s,
        )
        return subprocess.CompletedProcess(
            raw.args, raw.returncode, _decode(raw.stdout or b""), _decode(raw.stderr or b"")
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
            errors="replace",
            start_new_session=True,
        )

    # ------------------------------------------------------------ connect --

    def start(self) -> bool:
        """Connect: responder up, session directory made, probe streaming."""
        if not self.cfg.enabled:
            return False
        # Cleared here, not on success: _await_logger sets a warning for a
        # tracker that is streaming but locked on nothing, and clearing at the
        # end of a successful connect wiped exactly the message the operator
        # needed before recording anything.
        self.last_error = ""
        try:
            self._ensure_responder()
            if not self._spawn_probe():
                return False
            # Spawned before either is awaited so the probe's ~2 s and the
            # tracker's ~16 s handshake overlap instead of adding up.
            self._spawn_logger()
            if not self._await_probe():
                return False
            if not self._await_logger():
                self._teardown_probe()
                return False
            self._connected = True
            logger.info("laser tracker session %s connected (%s)", self.session_id, self.win_dir)
            return True
        except Exception as exc:  # never break Connect for the other devices
            self.last_error = f"connect failed: {type(exc).__name__}: {exc}"
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
        script = self.repo_root / self.cfg.responder_script
        if not script.exists():
            raise RuntimeError(f"responder script not found: {script}")
        subprocess.run([str(script), "start"], capture_output=True, text=True, errors="replace", timeout=30)

    def _spawn_probe(self) -> bool:
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
        return True

    def _spawn_logger(self) -> None:
        """One stream for the whole Connect window. See the module docstring."""
        self._run(f'del /q "{self.logger_stop_file}"')
        note = self.cfg.note or f"{self.session_id} (gateway session)"
        cmd = (
            f'cd /d {self.cfg.win_root} && "{self.cfg.logger_exe}" '
            f'--ip {self.cfg.tracker_ip} --out "{self.win_dir}" --session {self.session_id} '
            f'--note "{note}" --duration {self.cfg.session_cap_s:g} '
            f'--stop-file "{self.logger_stop_file}"'
        )
        self._logger_proc = self._spawn(cmd)
        Thread(target=self._read_logger_stdout, args=(self._logger_proc,),
               daemon=True, name="lt-logger-stdout").start()

    def _read_logger_stdout(self, proc: subprocess.Popen[str]) -> None:
        """Own the logger's stdout so the identity line is not lost.

        `GetDeviceInformation()` is announced once, right after connecting, and
        only on stdout -- the meta.json that also carries it is not written
        until the session stops, which is far too late to put in a device row.
        """
        if proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            self._logger_tail.append(line)
            if line.startswith("device:"):
                fields = {}
                for token in line[len("device:"):].strip().split():
                    key, sep, value = token.partition("=")
                    if sep:
                        fields[key] = value
                # The model name is the leading words before the first key=value.
                head = line[len("device:"):].strip().split(" sn=")[0].strip()
                self.device_info = {"model": head, **fields}
                logger.info("tracker identified: %s", self.device_info)

    def _await_probe(self) -> bool:
        # The probe is its own link check: if rows are landing, the responder is
        # up, UDP is open and the path works.  A separate 30 s --check would just
        # make Connect slower to learn the same thing.
        #
        # Polled rather than sampled once: rows appear on disk a flush at a time,
        # so a single read at a fixed instant races the writer and reports zero
        # exchanges on a perfectly healthy link.
        deadline = time.monotonic() + 8.0
        rows = 0
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if self._probe.poll() is not None:
                tail = (self._probe.stdout.read() if self._probe.stdout else "").strip()
                self.last_error = f"probe exited immediately: {tail[:300]}"
                self._probe = None
                return False
            rows = self.sync_rows()
            if rows > 0:
                break
        if rows <= 0:
            self.last_error = (
                "clock probe produced no exchanges in 8 s -- is the responder up on "
                f"{self.cfg.thor_host}:{self.cfg.responder_port}, and is UDP open?"
            )
            self._teardown_probe()
            return False
        logger.info("clock probe alive: %d exchanges", rows)
        return True

    def _await_logger(self) -> bool:
        """Wait for the tracker stream to actually produce samples.

        Measured 15-16 s to the first sample, so the budget is generous; a
        process that is merely alive proves nothing, because every failure mode
        worth catching (SA holding the instrument, not warmed up, no beam lock)
        leaves it alive and silent.
        """
        deadline = time.monotonic() + max(60.0, self.cfg.logger_ready_timeout_s)
        rows = 0
        while time.monotonic() < deadline:
            time.sleep(2.0)
            if self._logger_proc is not None and self._logger_proc.poll() is not None:
                # Commonest cause by far: the tracker admits one client and SA
                # has it. Say that rather than printing an SDK ordinal.
                tail = " | ".join(self._logger_tail)
                self.last_error = f"tracker logger exited: {tail[:300]}"
                self._logger_proc = None
                return False
            rows = self._count_rows(f"{self.session_id}.rt.csv")
            if rows > 0:
                break
        if rows <= 0:
            self.last_error = (
                f"tracker produced no samples in {self.cfg.logger_ready_timeout_s:g} s -- "
                "is it warmed up, locked on an SMR, and not held by SA?"
            )
            return False
        self._rows_mark = rows
        logger.info("tracker stream alive: %d samples", rows)
        # Streaming is not the same as measuring. Said here so the SMR can be
        # acquired before anything is recorded, rather than discovered in the
        # data afterwards.
        valid, tracking = self.beam_quality()
        self.beam_valid_fraction, self.beam_tracking_fraction = valid, tracking
        if 0.0 <= tracking < 0.5:
            self.last_error = (
                f"tracker is streaming but NOT locked on a target "
                f"(valid {valid * 100:.0f}%, tracking {tracking * 100:.0f}%) -- "
                "acquire the SMR before recording, or this session measures nothing"
            )
        return True

    def sync_rows(self) -> int:
        return self._count_rows(f"{self.session_id}.sync.csv")

    def beam_quality(self, sample_bytes: int = 200_000) -> tuple[float, float]:
        """Fraction of recent samples that are ``valid`` and ``tracking``.

        Row count alone says nothing about whether the tracker can see anything:
        with no SMR in the beam it still streams a full 1 kHz of
        ``valid=0 tracking=0 dist=0``, which is exactly what the first real
        session produced -- 81343 rows, 0 dropped, and not one measurement.
        Only the tail is read, so the cost does not grow with session length.
        """
        code = (
            "import sys;"
            "f=open(sys.argv[1],'rb');"
            "f.seek(0,2);n=f.tell();f.seek(max(0,n-int(sys.argv[2])));"
            "ls=f.read().decode('ascii','replace').splitlines()[1:];"
            "rs=[l.split(',') for l in ls if l.count(',')>=16];"
            "v=sum(1 for r in rs if r[9].strip() in ('1','true'));"
            "t=sum(1 for r in rs if r[10].strip() in ('1','true'));"
            "print(len(rs),v,t)"
        )
        res = self._run(
            f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{code}" '
            f'{self.session_id}.rt.csv {sample_bytes}',
            timeout_s=30,
        )
        try:
            total, valid, tracking = (int(x) for x in res.stdout.strip().splitlines()[-1].split())
        except (ValueError, IndexError):
            return (-1.0, -1.0)
        if total <= 0:
            return (-1.0, -1.0)
        return (valid / total, tracking / total)

    def _count_rows(self, filename: str) -> int:
        code = "import sys;print(sum(1 for _ in open(sys.argv[1]))-1)"
        res = self._run(f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{code}" {filename}', timeout_s=20)
        try:
            return int((res.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return -1

    # ------------------------------------------------------------ episode --

    def start_recording(self, episode_index: int, t_start_wall_s: float) -> bool:
        """Start Episode: open a boundary. No remote call, so no added latency.

        The stream is already running (see the module docstring), so this is
        bookkeeping. Deliberately: anything here lands on the same critical path
        as the cameras' own episode start.
        """
        if not self._connected:
            return False
        self._episode = EpisodeRecord(
            episode_index=episode_index, started=True, t_start_wall_s=t_start_wall_s
        )
        return True

    def stop_recording(self) -> dict[str, Any]:
        """Stop Episode: close the boundary and check the stream is still alive.

        One remote call, after the episode has already ended, so it costs the
        operator nothing. The number it reads is the session total; this
        episode's own rows are an offline slice by timestamp, and the field is
        named so that it cannot be mistaken for one.
        """
        rec = self._episode
        self._episode = None
        if rec is None:
            return {}
        rec.t_end_wall_s = time.time()
        try:
            total = self._count_rows(f"{self.session_id}.rt.csv")
            rec.rt_rows_total_at_stop = total
            # If the stream did not advance at all across an episode the tracker
            # stalled or lost the beam -- the one failure this can catch now,
            # while the rig is still set up.
            rec.stream_advanced = total > self._rows_mark
            valid, tracking = self.beam_quality()
            rec.beam_valid_fraction = valid
            rec.beam_tracking_fraction = tracking
            if not rec.stream_advanced:
                rec.error = "tracker stream did not advance during this episode"
                self.last_error = rec.error
            elif 0.0 <= tracking < 0.5:
                # The failure that looks like success: a full-rate stream of
                # nothing at all.
                rec.error = (
                    f"tracker not locked on a target during this episode "
                    f"(valid {valid * 100:.0f}%, tracking {tracking * 100:.0f}%)"
                )
                self.last_error = rec.error
            self._rows_mark = max(total, self._rows_mark)
        except Exception as exc:
            rec.error = f"stop failed: {type(exc).__name__}: {exc}"
            self.last_error = rec.error
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
                self.stop_recording()  # an exit mid-episode still closes its boundary
            self._teardown_logger()
            self._teardown_probe()
            result["sync_rows"] = self.sync_rows()
            result["rt_rows"] = self._count_rows(f"{self.session_id}.rt.csv")
            # Written before sealing so the manifest covers it: without the
            # boundaries the 1 kHz stream cannot be cut back into episodes, and
            # a sealed session that cannot be cut is not a usable session.
            self._write_episode_index()
            # The stop-files are plumbing, not data. Sealing them would put two
            # zero-byte sentinels in a manifest that is supposed to describe the
            # capture, and every later `verify` would carry them along.
            self._run(f'del /q "{self.logger_stop_file}" "{self.probe_stop_file}"')
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
            self.last_error = f"stop failed: {type(exc).__name__}: {exc}"
            logger.warning("laser tracker stop failed: %s", exc)
        finally:
            self._connected = False
        result["episodes"] = [asdict(r) for r in self._episodes]
        result["last_error"] = self.last_error
        return result

    def _write_episode_index(self) -> None:
        """Put the episode boundaries next to the stream they cut."""
        payload = {
            "session_id": self.session_id,
            "stream": f"{self.session_id}.rt.csv",
            "clock_link": f"{self.session_id}.sync.csv",
            "note": (
                "t_*_wall_s are Thor wall-clock seconds taken from the camera "
                "episode's own t0. Cut the stream with the QPC <-> "
                "CLOCK_MONOTONIC fit in metrology.laser_tracker_clock."
            ),
            "episodes": [asdict(r) for r in self._episodes],
        }
        local = Path(f"/tmp/{self.session_id}.episodes.json")
        local.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        key = str(Path(self.cfg.ssh_key).expanduser())
        dest = f"{self.cfg.win_host}:{self.win_dir.replace(chr(92), '/')}/{self.session_id}.episodes.json"
        subprocess.run(["scp", "-i", key, *_SSH_BASE, str(local), dest],
                       capture_output=True, timeout=120)
        local.unlink(missing_ok=True)

    def _teardown_logger(self) -> None:
        if self._logger_proc is None:
            return
        try:
            self._run(f'type nul > "{self.logger_stop_file}"')
            try:
                self._logger_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._logger_proc.terminate()
                self.last_error = "tracker logger did not stop within 60 s; terminated"
        except Exception as exc:
            self.last_error = f"logger teardown failed: {type(exc).__name__}: {exc}"
        finally:
            self._logger_proc = None

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
            self.last_error = f"probe teardown failed: {type(exc).__name__}: {exc}"
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
            capture_output=True, timeout=1800,
        )
        if res.returncode != 0:
            detail = (_decode(res.stderr or b"") or _decode(res.stdout or b"")).strip()
            self.last_error = f"landing failed: {detail[:300]}"
            return ""
        landed = dest_root / self.session_id
        verifier = self.repo_root / (
            "third_party/opencv_kalibr/metrology/laser_tracker/session_manifest.py"
        )
        if verifier.exists():
            check = subprocess.run(
                ["python3", str(verifier), "verify", str(landed)],
                capture_output=True, text=True, errors="replace", timeout=600,
            )
            if check.returncode != 0:
                # A truncated copy fails verify rather than being analysed --
                # that is the whole point of sealing before moving.
                self.last_error = f"landed copy does not match its manifest: {(check.stdout or check.stderr).strip()[:300]}"
        return str(landed)

    # ------------------------------------------------------------- status --

    def describe(self) -> str:
        """One line naming the instrument this session is being taken with."""
        d = self.device_info
        if not d:
            return f"{self.cfg.tracker_ip} via {self.cfg.win_host}"
        parts = [d.get("model", "tracker")]
        if d.get("sn"):
            parts.append(f"S/N {d['sn']}")
        if d.get("fw"):
            parts.append(f"fw {d['fw']}")
        acc = d.get("accessory", "none")
        parts.append(f"accessory {acc}" if acc != "none" else "no accessory")
        parts.append(f"1 kHz @ {self.cfg.tracker_ip}")
        return ", ".join(parts)

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
            device=dict(self.device_info),
            beam_valid_fraction=self.beam_valid_fraction,
            beam_tracking_fraction=self.beam_tracking_fraction,
            episodes=[asdict(r) for r in self._episodes],
        )
