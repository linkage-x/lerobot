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
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)

_SSH_BASE = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new")

# api::lt::OperationMode, in declaration order.  Reported by name because the
# ordinal hides the one distinction that matters: `Position` holds the gimbal
# and *cannot* track, so an SMR placed in the beam would still not be acquired.
_PROGRESS_RE = re.compile(r"^\s*rows \d+\s+dropped \d+\s*$")
_OP_MODES = ("Idle", "Tracking", "Position", "TrackIdle", "Searching", "Internal")

# Shared head of the snippets that run on the capture PC: argv = rt.csv, tail
# bytes, seconds-ago of the window's start and end, <snippet's own>, [QPC
# frequency, QPC now -- tests only]. Leaves ``rs``: the rows whose QPC stamp falls
# in the window on the PC's own clock, so a stream in gaps cannot stretch the
# window back into whatever came before it (as "the last N rows" does).
# Columns: 1 ctrl_ts_ms, 2 qpc, 5 dist_mm, 6-8 x/y/z, 9 valid, 10 tracking,
# 12 op_mode, 18 ranged (absent on v1 streams, which ranged nothing separately).
# No double quotes and no percent signs: these travel inside cmd's "...".
_QPC_WINDOW_PRELUDE = (
    "import sys,collections,ctypes;"
    "A=sys.argv;"
    "fq,now=(int(A[6]),int(A[7])) if len(A)>7 else (lambda k,f,c:(k.QueryPerformanceFrequency("
    "ctypes.byref(f)),k.QueryPerformanceCounter(ctypes.byref(c)),f.value,c.value)[2:])("
    "ctypes.windll.kernel32,ctypes.c_int64(),ctypes.c_int64());"
    "fh=open(A[1],'rb');fh.seek(0,2);n=fh.tell();fh.seek(max(0,n-int(A[2])));"
    "ls=fh.read().decode('ascii','replace').splitlines()[1:];"
    "a=now-int(float(A[3])*fq);b=now-int(float(A[4])*fq);"
    "T=lambda v:v.strip() in ('1','true');"
    "rs=[r for r in (l.split(',') for l in ls if l.count(',')>=16)"
    " if r[2].strip().isdigit() and a<=int(r[2])<=b];"
)

# argv[5] = nest distance (mm). One line out: rows, distinct controller ms,
# valid, tracking, modal op_mode, min/max distance of the ranged valid points,
# how many of those sit on the nest and off it, the longest gap (ms, window
# edges included), window length (s).
_EPISODE_STATS_CODE = _QPC_WINDOW_PRELUDE + (
    "N=float(A[5]);"
    "vr=[r for r in rs if T(r[9])];"
    "d=[float(r[5]) for r in vr if len(r)<19 or T(r[18])];"
    "q=[a]+[int(r[2]) for r in rs]+[b];"
    "g=max(y-x for x,y in zip(q,q[1:]))*1000.0/fq;"
    "m=collections.Counter(r[12].strip() for r in rs).most_common(1);"
    "print(len(rs),len(set(r[1] for r in rs)),len(vr),sum(1 for r in rs if T(r[10])),"
    "m[0][0] if m else -1,min(d) if d else -1,max(d) if d else -1,"
    "sum(1 for x in d if x<N),sum(1 for x in d if x>=N),round(g,1),round((b-a)/fq,3))"
)

# argv[5] = at most this many points. One line out: "x y z;x y z;..." (mm) of the
# ranged valid rows, thinned evenly.
_SEGMENT_POINTS_CODE = _QPC_WINDOW_PRELUDE + (
    "vr=[r for r in rs if T(r[9]) and (len(r)<19 or T(r[18]))];"
    "k=max(1,len(vr)//int(A[5]));"
    "print(';'.join(r[6]+' '+r[7]+' '+r[8] for r in vr[::k]))"
)


def parse_segment_points(stdout: str) -> list[tuple[float, float, float]]:
    """``_SEGMENT_POINTS_CODE``'s line as points; malformed items are skipped."""
    out: list[tuple[float, float, float]] = []
    lines = stdout.strip().splitlines()
    for item in lines[-1].split(";") if lines else []:
        try:
            x, y, z = (float(v) for v in item.split())
        except ValueError:
            continue
        out.append((x, y, z))
    return out


def parse_episode_stats(stdout: str) -> dict[str, Any]:
    """Fractions of one episode's window from ``_EPISODE_STATS_CODE``'s line; rows -1 if unreadable."""
    out: dict[str, Any] = {
        "rows": -1, "valid": -1.0, "tracking": -1.0, "mode": -1,
        "dist_min_mm": -1.0, "dist_max_mm": -1.0, "window_s": -1.0, "coverage": -1.0,
        "max_gap_ms": -1.0, "nest_fraction": -1.0, "usable_fraction": -1.0,
    }
    try:
        f = stdout.strip().splitlines()[-1].split()
        rows, distinct, valid, tracking, mode = (int(x) for x in f[:5])
        dmin, dmax = float(f[5]), float(f[6])
        nest, usable = int(f[7]), int(f[8])
        gap_ms, window_s = float(f[9]), float(f[10])
    except (ValueError, IndexError):
        return out
    if window_s <= 0.0:
        return out
    expected = window_s * 1000.0  # one row per controller millisecond
    per_row = 1.0 / rows if rows > 0 else 0.0
    out.update(
        rows=rows,
        valid=valid * per_row,
        tracking=tracking * per_row,
        mode=mode,
        dist_min_mm=dmin,
        dist_max_mm=dmax,
        window_s=window_s,
        coverage=min(1.0, distinct / expected),
        max_gap_ms=gap_ms,
        nest_fraction=min(1.0, nest / expected),
        usable_fraction=min(1.0, usable / expected),
    )
    return out


def episode_verdict(
    st: dict[str, Any], *, min_usable_fraction: float, nest_max_mm: float, gimbal_mode: str = ""
) -> tuple[str, bool]:
    """``(error, incomplete)`` for one episode's window stats.

    ``incomplete`` is the gate: too little of the episode has a ranged point off the
    nest for it to be laser GT. The error names the cause the operator can fix.
    A window that could not be read is reported but is not called incomplete --
    unknown is not the same as bad, and the episode is not thrown away for it.
    """
    if st["rows"] < 0:
        return ("could not read this episode's tracker rows from the capture PC: "
                "check it after Disconnect, before using it as GT", False)
    valid, tracking, usable = st["valid"], st["tracking"], st["usable_fraction"]
    if st["rows"] > 0 and tracking < 0.5:
        # The failure that looks like success: a full-rate stream of nothing at all.
        return (f"tracker not measuring during this episode: gimbal in "
                f"{gimbal_mode or 'unknown mode'} "
                f"(valid {valid * 100:.0f}%, tracking {tracking * 100:.0f}%)", True)
    if tracking >= 0.5 > valid and usable < 0.05:
        # Tracking without measuring. The gimbal follows the SMR and az/el/dist keep
        # updating while x/y/z are zeroed, so every liveness check passes and the
        # episode has no usable point in it. 2026-09-20 recorded two episodes this
        # way: the distance reference Home left behind does not survive a beam
        # break, and only a running measurement routine restores it.
        return (f"tracker tracked but did not measure: valid {valid * 100:.0f}% "
                f"of {st['rows']} rows against tracking {tracking * 100:.0f}% "
                "-- is the measurement routine running (--measure-ms)?", True)
    if usable < min_usable_fraction:
        head = (f"only {usable * 100:.0f}% of this {st['window_s']:.1f} s episode has a "
                f"ranged laser point off the home nest (need {min_usable_fraction * 100:.0f}%)")
        if st["coverage"] < min_usable_fraction:
            return (f"{head}: the stream covered {st['coverage'] * 100:.0f}% of it, longest gap "
                    f"{st['max_gap_ms'] / 1000:.1f} s -- is RadianCAL-Pro or another tracker "
                    "client running on the capture PC?", True)
        if st["nest_fraction"] > 1.0 - min_usable_fraction:
            return (f"{head}: the beam was on the home nest (< {nest_max_mm:.0f} mm from the "
                    f"head) for {st['nest_fraction'] * 100:.0f}% of it -- carry the SMR to the "
                    "target with the beam locked before StartEpisode", True)
        return (f"{head}: the beam was off the SMR for most of it (valid {valid * 100:.0f}% of rows, "
                f"tracking {tracking * 100:.0f}%) -- keep the line of sight clear and move slower; "
                "after a break the tracker re-homes to the nest", True)
    span = st["dist_max_mm"] - st["dist_min_mm"] if st["dist_min_mm"] >= 0.0 else -1.0
    if 0.0 <= span < 5.0:
        # Valid, dense, and all at one distance. Not an error -- a dwell recording
        # for fit_tracker_mount looks exactly like this -- but it must not pass for
        # a trajectory.
        return (f"every valid sample is at {st['dist_min_mm']:.0f} mm "
                f"(span {span:.1f} mm): the target did not move -- correct "
                "for a dwell, useless as a trajectory", False)
    return ("", False)

# What a failure means right after the controller has been power-cycled, which
# is exactly when these codes stop meaning what they usually mean.  2026-09-21,
# after an unplanned power cut, four Connects in a row walked through the whole
# cold start -- CommunicationFailed (controller still booting), IndexSearchFailed
# (encoder index search on the first connect after power-on), then two Connects
# that *succeeded* but reported TrackerNotWarmedUp -- and the operator, told to
# close SA and to put the SMR in the nest, gave up on a tracker that only needed
# fifteen minutes.
_CONNECT_HINTS = {
    "CommunicationFailed": (
        " -- either another client holds it (the tracker admits one client at a time: "
        "close SA / RadianCAL / Tracker Studio), or the controller is still booting "
        "after a power cycle (wait until it answers, then reconnect)"
    ),
    "DeviceAlreadyConnected": (
        " -- the tracker admits one client at a time; close SA / RadianCAL / "
        "Tracker Studio and reconnect"
    ),
    "IndexSearchFailed": (
        " -- the first connect after power-on runs an encoder index search, which "
        "rotates the head about both axes: check the Servo switch is on (Power "
        "first, then Servo) and nothing blocks the head, then reconnect"
    ),
}


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
    # Homing establishes the ADM distance reference, which is what makes the
    # samples valid; SwitchToTrack() gets the motors out of Position but leaves
    # the reference missing, so both are needed and neither substitutes for the
    # other. (An earlier note here claimed Home was the ONLY way out of Position
    # and that no mode setter existed. Wrong on both counts -- SwitchToTrack,
    # SwitchToPosition and SwitchToIdle are all in the v7.18 interface.)
    # It drives a shared precision instrument, so it happens only when asked for
    # -- and only with a declared SMR size, because homing against the wrong
    # nominal radius is a metrology error, not a crash.
    home_on_connect: bool = False
    smr_size: str = ""          # "1.5" | "0.5" | "7/8"
    adm_offset_mm: float = 0.0
    home_retry_s: float = 10.0
    # Period of StartTemporalDynamicMeasurement, the routine that enables the
    # optical switch and so decides whether x/y/z are valid at all. 1 ms is the
    # SDK's maximum and matches the feed. 0 disables it, which is the pre-
    # 2026-09-20 behaviour and is kept only so the two can be compared without
    # rebuilding the capture PC's binary.
    measure_ms: int = 1
    # Episode gate. An episode is laser GT only where the stream has a ranged
    # point off the home nest; below this fraction of its duration the recorder
    # discards it instead of saving it (2026-10-09 ep0: 13% -- RadianCAL-Pro left
    # the stream ~1 s in every 5 -- saved, and found three days later). 0.8 is
    # what validate_against_tracker asks of the paired camera frames anyway.
    # Calibration captures are never discarded by it: their own checks decide.
    min_usable_fraction: float = 0.8
    discard_incomplete: bool = True
    # Home leaves the beam on the nest ~158 mm from the head; nothing measured
    # that close is the target.
    nest_max_mm: float = 300.0
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
    # Measured over THIS episode's rows, not the tail of the file: the tail is
    # whatever the operator happened to be doing at Stop, and on 2026-09-20 it
    # reported 0% valid for an episode that was 46% valid, which read as a
    # tracker fault rather than as the real finding.
    beam_valid_fraction: float = -1.0
    beam_tracking_fraction: float = -1.0
    # How many rows the fractions above actually covered, and the spread of the
    # distances among the valid ones.  A session whose valid samples all sit at
    # one distance measured the nest, not the target.
    beam_window_rows: int = 0
    beam_dist_min_mm: float = -1.0
    beam_dist_max_mm: float = -1.0
    # The episode's window is cut on the capture PC's own clock (QPC), not as
    # "the last N rows": with the stream in gaps the last 15341 rows of
    # 2026-10-09 ep0 reached 104 s back, read 100% valid, and passed an episode
    # whose own 15.3 s held 13% of a stream.
    beam_window_s: float = -1.0
    stream_coverage: float = -1.0  # distinct controller ms / window ms
    max_gap_ms: float = -1.0  # longest stretch without a row, window edges included
    nest_fraction: float = -1.0  # ranged points on the home nest / window ms
    usable_fraction: float = -1.0  # ranged points off the nest / window ms
    # Below min_usable_fraction: not laser GT. The recorder discards such an
    # episode unless it is a calibration capture.
    incomplete: bool = False
    discarded: bool = False
    discard_reason: str = ""
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
    gimbal_mode: str = ""
    beam_ready: bool = False
    beam_status: str = "unknown"
    # The logger ranges every beam lock absolutely before its samples count
    # (2026-09-22). This is the last one it reported, step and all: the number
    # an operator needs while judging whether a catch can be trusted.
    range_status: str = ""
    locks_ranged: int = 0
    locks_not_ranged: int = 0
    # Homing is what gives the session an absolute range, and "beam: acquired"
    # does not imply it: W2 (2026-09-21) ran locked-on and green from start to
    # finish without ever homing, and every lock carried a range hundreds of mm
    # off. A session that has not homed is not ground truth.
    homed: bool = False
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
        self.gimbal_mode = ""
        # Live, pushed by the logger rather than polled: 'waiting' is a state
        # the operator is actively changing, not a verdict on the session.
        self.beam_ready = False
        self.beam_status = "unknown"
        self.range_status = ""
        self.locks_ranged = 0
        self.locks_not_ranged = 0
        self.homed = False
        # Whether the current lock's range traces back to a Home without a beam
        # break. None: the logger does not report it (an exe older than
        # 2026-09-23), and then only `homed` is gated on.
        self.range_absolute: bool | None = None
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
        # stdin=DEVNULL is load-bearing: this runs inside the recorder, whose stdin
        # is the GUI's command channel, and ssh switches the stdin it inherits to
        # non-blocking. A Save/Discard arriving during the call then leaves the
        # recorder's next readline() an EAGAIN that reads as "" -- the GUI closed
        # stdin -- and the whole session quits (2026-10-13 10:41, right after a
        # Discard pressed during the episode-stop tracker check).
        raw = subprocess.run(
            [*self._ssh_argv(), remote_cmd],
            stdin=subprocess.DEVNULL,
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
                # A logger that is still alive here is almost always stuck in the
                # SDK handshake, and one that finishes it later would hold the
                # tracker's only client slot until session_cap_s -- so the next
                # Connect would fail as "CommunicationFailed" with no logger in
                # sight. The stop-file is what releases it: the logger checks it
                # the moment it is connected.
                self._teardown_logger(wait_s=5.0)
                self._teardown_probe()
                return False
            self._connected = True
            logger.info("laser tracker session %s connected (%s)", self.session_id, self.win_dir)
            return True
        except Exception as exc:  # never break Connect for the other devices
            self.last_error = f"connect failed: {type(exc).__name__}: {exc}"
            logger.warning("laser tracker connect failed: %s", exc)
            self._teardown_logger(wait_s=5.0)
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
        subprocess.run(
            [str(script), "start"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            errors="replace", timeout=30,
        )

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
        home = ""
        if self.cfg.home_on_connect:
            if self.cfg.smr_size:
                home = (
                    f" --home {self.cfg.smr_size}"
                    f" --adm-offset {self.cfg.adm_offset_mm:g}"
                    f" --home-retry-s {self.cfg.home_retry_s:g}"
                )
            else:
                # Refused rather than guessed: 1.5" and 7/8" are both plausible
                # and the wrong one homes against the wrong nominal radius.
                self.last_error = (
                    "home_on_connect is set but smr_size is empty -- declare the SMR "
                    "diameter (1.5 / 0.5 / 7/8) in the laser_tracker config"
                )
        measure = (
            f" --measure-ms {int(self.cfg.measure_ms)}"
            if self.cfg.measure_ms > 0
            else " --no-measure"
        )
        cmd = (
            f'cd /d {self.cfg.win_root} && "{self.cfg.logger_exe}" '
            f'--ip {self.cfg.tracker_ip} --out "{self.win_dir}" --session {self.session_id} '
            f'--note "{note}" --duration {self.cfg.session_cap_s:g} '
            f'--stop-file "{self.logger_stop_file}"{home}{measure}'
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
            # The progress counter repeats once a second; left in, it would push
            # the one line that explains a failure out of the tail we report.
            if not _PROGRESS_RE.match(line):
                self._logger_tail.append(line)
            # Not `startswith`: the logger's once-a-second progress counter is a
            # bare "\rrows N  dropped M" with no newline, so it shares stdout
            # with these markers and every one after the first arrives glued to
            # the tail of it -- "rows 12000  dropped 0   beam: acquired".  That
            # cost us a green light with the SMR already locked on 2026-09-20.
            beam_at = line.find("beam:")
            if beam_at >= 0:
                what = line[beam_at + len("beam:"):].strip()
                if what.startswith("acquired"):
                    self.beam_ready, self.beam_status = True, "acquired"
                elif what.startswith("lost"):
                    self.beam_ready, self.beam_status = False, "lost"
                elif what.startswith("waiting"):
                    self.beam_ready = False
                    self.beam_status = what  # carries the SDK's reason in brackets
                logger.info("tracker beam: %s", self.beam_status)
            # Recovery and search attempts go to the recorder log: on 2026-09-23 a
            # session sat tracking with no distance for minutes and the only
            # account of why -- what each re-range answered -- was in this tail,
            # in memory, gone with the process.
            for tag in ("recover:", "search:", "home: nest"):
                at = line.find(tag)
                if at >= 0:
                    logger.info("tracker %s", line[at:])
                    break
            range_at = line.find("range:")
            if range_at >= 0:
                what = line[range_at + len("range:"):].strip()
                if "not absolute" in what:
                    self.range_absolute = False
                elif "absolute" in what:
                    self.range_absolute = True
                logger.info("tracker range: %s", what)
            if line.find("home: ok") >= 0:
                self.homed = True
                logger.info("tracker homed: the session has an absolute range")
            lock_at = line.find("lock:")
            if lock_at >= 0:
                self.range_status = line[lock_at + len("lock:"):].strip()
                if "NOT ranged" in self.range_status:
                    self.locks_not_ranged += 1
                else:
                    self.locks_ranged += 1
                logger.info("tracker lock: %s", self.range_status)
            dev_at = line.find("device:")
            if dev_at >= 0:
                fields = {}
                for token in line[dev_at + len("device:"):].strip().split():
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
                tail = " | ".join(self._logger_tail)
                # Commonest cause by far: the tracker admits ONE client, so SA
                # or RadianCAL being open is enough to refuse us -- and the SDK
                # reports that as a generic communication failure, which reads
                # like a network fault and sends people to check cables.
                hint = next((h for code, h in _CONNECT_HINTS.items() if code in tail), "")
                self.last_error = f"tracker logger exited: {tail[:300]}{hint}"
                self._logger_proc = None
                return False
            rows = self._count_rows(f"{self.session_id}.rt.csv")
            if rows > 0:
                break
        if rows <= 0:
            self.last_error = (
                f"tracker produced no samples in {self.cfg.logger_ready_timeout_s:g} s -- "
                "is the controller up (it takes a while after a power cycle), and is the "
                "tracker not held by SA?"
            )
            return False
        self._rows_mark = rows
        logger.info("tracker stream alive: %d samples", rows)
        # Streaming is not the same as measuring. Said here so the SMR can be
        # acquired before anything is recorded, rather than discovered in the
        # data afterwards.
        valid, tracking = self.beam_quality()
        self.beam_valid_fraction, self.beam_tracking_fraction = valid, tracking
        if tracking >= 0.5:
            self.beam_ready, self.beam_status = True, "acquired"
        # Not an error any more. The tracker keeps homing on a timer, so an empty
        # nest resolves itself the moment the operator puts the SMR in it; the
        # recorder shows the wait and refuses to start an episode until it ends.
        # Failing Connect here would make the operator reconnect for something
        # that fixes itself in ten seconds.
        return True

    def sync_rows(self) -> int:
        return self._count_rows(f"{self.session_id}.sync.csv")

    # A generous upper bound on one row of rt.csv, used to size the tail read
    # so that asking for N rows actually gets N rows.  Rows measured 108-118
    # bytes on 2026-09-20; 160 leaves room for wider coordinates.
    _BYTES_PER_ROW = 160

    def beam_stats(self, *, last_rows: int = 0, sample_bytes: int = 200_000) -> dict[str, Any]:
        """What the beam did over the last ``last_rows`` samples, or the tail.

        Row count alone says nothing about whether the tracker can see anything:
        with no SMR in the beam it still streams a full 1 kHz of
        ``valid=0 tracking=0 dist=0``, which is exactly what the first real
        session produced -- 81343 rows, 0 dropped, and not one measurement.

        ``last_rows`` is how an episode asks about *itself*.  The stream is a
        hard 1 kHz with one row per controller millisecond, so an episode's row
        count is its duration in ms and its rows are the last that many at Stop.
        Without it this read the file's tail, which is a statistic about the
        instant the operator pressed Stop and nothing else.

        ``dist_min_mm``/``dist_max_mm`` cover the *valid* rows only. They are
        here because 100% tracking, 45% valid and every valid sample at one
        distance is not a healthy episode -- it is the home nest, measured
        beautifully, while the target was somewhere else entirely.
        """
        budget = sample_bytes
        if last_rows > 0:
            budget = max(budget, (last_rows + 2) * self._BYTES_PER_ROW)
        code = (
            "import sys;"
            "f=open(sys.argv[1],'rb');"
            "f.seek(0,2);n=f.tell();f.seek(max(0,n-int(sys.argv[2])));"
            # [1:] drops the header on a whole-file read and the partial first
            # line on a seek; both are exactly what we want gone.
            "ls=f.read().decode('ascii','replace').splitlines()[1:];"
            "rs=[l.split(',') for l in ls if l.count(',')>=16];"
            "k=int(sys.argv[3]);"
            "rs=rs[-k:] if k>0 else rs;"
            "vr=[r for r in rs if r[9].strip() in ('1','true')];"
            "t=sum(1 for r in rs if r[10].strip() in ('1','true'));"
            "import collections;"
            "m=collections.Counter(r[12].strip() for r in rs).most_common(1);"
            "d=[float(r[5]) for r in vr];"
            "print(len(rs),len(vr),t,m[0][0] if m else -1,"
            "min(d) if d else -1,max(d) if d else -1)"
        )
        res = self._run(
            f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{code}" '
            f'{self.session_id}.rt.csv {budget} {last_rows}',
            timeout_s=45,
        )
        bad = {"rows": 0, "valid": -1.0, "tracking": -1.0, "dist_min_mm": -1.0,
               "dist_max_mm": -1.0}
        try:
            fields = res.stdout.strip().splitlines()[-1].split()
            total, valid, tracking, mode = (int(x) for x in fields[:4])
            dmin, dmax = (float(x) for x in fields[4:6])
        except (ValueError, IndexError):
            return bad
        if total <= 0:
            return bad
        self.gimbal_mode = _OP_MODES[mode] if 0 <= mode < len(_OP_MODES) else f"mode {mode}"
        return {
            "rows": total,
            "valid": valid / total,
            "tracking": tracking / total,
            "dist_min_mm": dmin,
            "dist_max_mm": dmax,
        }

    def episode_window_stats(self, *, ago_start_s: float, ago_end_s: float) -> dict[str, Any]:
        """The stream between ``ago_start_s`` and ``ago_end_s`` seconds before now, on the PC's clock.

        Rows are picked by their QPC stamp against the capture PC's QueryPerformanceCounter
        at the moment of the call, so a stream in gaps cannot stretch the window back into
        whatever came before the episode. The ssh round trip shifts it earlier by its own
        latency (well under a second). ``rows`` is -1 when the read failed.
        """
        budget = int((max(ago_start_s, 0.0) * 1000.0 + 3000.0) * self._BYTES_PER_ROW)
        res = self._run(
            f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{_EPISODE_STATS_CODE}" '
            f"{self.session_id}.rt.csv {budget} {ago_start_s:.3f} {ago_end_s:.3f} {self.cfg.nest_max_mm:g}",
            timeout_s=45,
        )
        st = parse_episode_stats(res.stdout or "")
        mode = st.pop("mode")
        if 0 <= mode < len(_OP_MODES):
            self.gimbal_mode = _OP_MODES[mode]
        return st

    def segment_points(
        self, *, ago_start_s: float, ago_end_s: float, max_points: int = 3000
    ) -> list[tuple[float, float, float]]:
        """x/y/z (mm) of the valid samples between ``ago_start_s`` and ``ago_end_s``
        seconds before now on the capture PC's clock, thinned there to at most
        ``max_points`` so the reply stays small.

        Only for the capture-time geometry preview (``tracker_live_geometry``);
        the solves read the landed stream. A v2 row whose lock is not yet ranged
        (``ranged`` column 0) is left out, as the solves leave it out. Cut by
        time, not as the last N rows: with the stream in gaps those reach back to
        the previous pose and blend it into this one's dwell point or pivot gain.
        """
        if ago_start_s <= ago_end_s:
            return []
        budget = int((max(ago_start_s, 0.0) * 1000.0 + 3000.0) * self._BYTES_PER_ROW)
        res = self._run(
            f'cd /d "{self.win_dir}" && {self.cfg.python} -c "{_SEGMENT_POINTS_CODE}" '
            f"{self.session_id}.rt.csv {budget} {ago_start_s:.3f} {ago_end_s:.3f} {max(1, int(max_points))}",
            timeout_s=45,
        )
        return parse_segment_points(res.stdout or "")


    def beam_quality(self, sample_bytes: int = 200_000) -> tuple[float, float]:  # noqa: D401
        """Fraction of recent samples that are ``valid`` and ``tracking``."""
        st = self.beam_stats(sample_bytes=sample_bytes)
        return (st["valid"], st["tracking"])

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
            # The episode's own window, on the capture PC's clock. Asking about
            # the file's tail answers a question about the instant Stop was
            # pressed; asking for the last (duration in ms) rows answers one
            # about however far back the stream's gaps reach.
            now = time.time()
            st = self.episode_window_stats(
                ago_start_s=max(0.0, now - rec.t_start_wall_s),
                ago_end_s=max(0.0, now - rec.t_end_wall_s),
            )
            rec.beam_valid_fraction = st["valid"]
            rec.beam_tracking_fraction = st["tracking"]
            rec.beam_window_rows = max(0, st["rows"])
            rec.beam_dist_min_mm = st["dist_min_mm"]
            rec.beam_dist_max_mm = st["dist_max_mm"]
            rec.beam_window_s = st["window_s"]
            rec.stream_coverage = st["coverage"]
            rec.max_gap_ms = st["max_gap_ms"]
            rec.nest_fraction = st["nest_fraction"]
            rec.usable_fraction = st["usable_fraction"]
            if not rec.stream_advanced:
                rec.error = "tracker stream did not advance during this episode"
                rec.incomplete = True
            else:
                rec.error, rec.incomplete = episode_verdict(
                    st,
                    min_usable_fraction=self.cfg.min_usable_fraction,
                    nest_max_mm=self.cfg.nest_max_mm,
                    gimbal_mode=self.gimbal_mode,
                )
            if rec.error:
                self.last_error = rec.error
            self._rows_mark = max(total, self._rows_mark)
        except Exception as exc:
            rec.error = f"stop failed: {type(exc).__name__}: {exc}"
            self.last_error = rec.error
        self._episodes.append(rec)
        return asdict(rec)

    def mark_discarded(self, reason: str) -> None:
        """The recorder threw the last episode away; say so in the episode index.

        Its rows stay in the session's one stream (they cannot be cut out of a file
        still being written), and the next take reuses its episode index, so without
        this mark the index would carry two different takes under one number.
        """
        if self._episodes:
            self._episodes[-1].discarded = True
            self._episodes[-1].discard_reason = str(reason)

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
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        local.unlink(missing_ok=True)

    def _teardown_logger(self, *, wait_s: float = 60.0) -> None:
        """Ask the logger to stop, then drop the ssh channel if it will not.

        The stop-file is the part that actually works on the Windows side:
        ``lt_realtime_logger`` polls it in its writer loop, i.e. from the moment
        it is connected. Closing the local ssh channel is only a backstop --
        whether that ends the remote process is up to the Windows OpenSSH
        server, and nothing here relies on it. So on a failed Connect the
        stop-file is left in place for a logger that has not connected yet.
        """
        if self._logger_proc is None:
            return
        try:
            self._run(f'type nul > "{self.logger_stop_file}"')
            try:
                self._logger_proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                self._logger_proc.terminate()
                if not self.last_error:
                    self.last_error = f"tracker logger did not stop within {wait_s:g} s; terminated"
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
            stdin=subprocess.DEVNULL, capture_output=True, timeout=1800,
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
                stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=600,
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

    @property
    def ready(self) -> bool:
        """Whether an episode recorded now would carry ground truth.

        A locked beam is not enough: without a successful Home the session has
        no absolute range, and every lock inherits whatever reference was left
        behind -- W2 (2026-09-21) was locked and green throughout and every
        range was 337-440 mm off.  This is what gates Start Episode.
        """
        return self.beam_ready and self.homed and self.range_absolute is not False

    @property
    def beam_broken(self) -> bool:
        """Homed, but the beam broke since: this lock's range is not absolute.

        A catch after a break takes whatever range the ADM hands back, and a
        single-point measurement only reports that back -- pivot
        lt_20260923_062953 was caught 14 mm out of the nest and carried +4.3 mm.
        """
        return self.homed and self.range_absolute is False

    def beam_summary(self) -> str:
        """What the beam is doing, in the words an operator needs."""
        if self.beam_ready and self.beam_broken:
            return (
                "the beam broke since Home, so this lock has no absolute range — put the "
                "SMR back in the home nest; it re-homes automatically"
            )
        if self.beam_ready and self.homed:
            return "homed, locked on the SMR"
        if self.beam_ready:
            if not self.cfg.home_on_connect:
                return (
                    "locked on the SMR but NOT homed: this Connect does not home "
                    "(home_on_connect is off), so the session has no absolute range"
                )
            return (
                "locked on the SMR but NOT homed yet — put the SMR in the home nest "
                "and wait for Home to succeed; homing retries automatically"
            )
        if self.beam_status.startswith("waiting"):
            # Exactly one layer of brackets: the SDK's reason carries its own
            # numeric code in brackets, so strip(" ()") would eat that one too
            # and hand the operator "NoSmrAtHomePosition (21".
            reason = self.beam_status[len("waiting"):].strip()
            if reason.startswith("(") and reason.endswith(")"):
                reason = reason[1:-1]
            if reason.startswith("NoSmrAtHomePosition"):
                return "waiting for the SMR — put it in the home nest, homing retries automatically"
            if reason.startswith("TrackerNotWarmedUp"):
                # Nothing the operator does with the SMR changes this, and saying
                # "waiting for the SMR" here is what made a healthy tracker look
                # broken after the 2026-09-21 power cut.
                return (
                    "tracker still warming up — homing is refused until the laser is ready "
                    "(at least 15 min after power-on, front LED steady); retries automatically"
                )
            return f"waiting ({reason}) — homing retries automatically"
        if self.beam_status == "lost":
            return "beam lost — reacquiring"
        return "waiting for the SMR — homing retries automatically"

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
            gimbal_mode=self.gimbal_mode,
            beam_ready=self.beam_ready,
            beam_status=self.beam_status,
            range_status=self.range_status,
            homed=self.homed,
            locks_ranged=self.locks_ranged,
            locks_not_ranged=self.locks_not_ranged,
            episodes=[asdict(r) for r in self._episodes],
        )
