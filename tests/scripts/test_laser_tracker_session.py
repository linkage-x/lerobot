"""The laser tracker must never be able to stop a recording.

It is a shared instrument on another machine, booked by other people and
unavailable more often than not. Every test here is about that: the driver is
allowed to fail, and it is not allowed to raise while doing so.
"""

from pathlib import Path

from tools.thor.gmsl2 import laser_tracker_session as lts


def _stats(**over):
    """An episode window as ``episode_window_stats`` returns it: healthy unless told otherwise."""
    base = {"rows": 20000, "valid": 1.0, "tracking": 1.0, "dist_min_mm": 400.0, "dist_max_mm": 1800.0,
            "window_s": 20.0, "coverage": 1.0, "max_gap_ms": 2.0, "nest_fraction": 0.0,
            "usable_fraction": 1.0}
    base.update(over)
    return base


def _cfg(**over):
    base = {"enabled": True, "win_host": "user@10.0.0.1", "ssh_key": "/nonexistent/key"}
    base.update(over)
    return lts.config_from_yaml_dict(base)


def test_absent_block_is_disabled_not_an_error() -> None:
    cfg = lts.config_from_yaml_dict(None)
    assert cfg.enabled is False
    assert cfg.configured is False


def test_enabled_without_a_host_disables_itself() -> None:
    # Declaring enabled: true with no win_host is a config mistake; answering it
    # with a crash at Connect would punish the wrong person.
    cfg = lts.config_from_yaml_dict({"enabled": True})
    assert cfg.enabled is False


def test_gui_toggle_overrides_the_file_in_both_directions() -> None:
    raw = {"enabled": False, "win_host": "user@10.0.0.1"}
    assert lts.config_from_yaml_dict(raw, enabled_override=True).enabled is True
    on = {"enabled": True, "win_host": "user@10.0.0.1"}
    # The direction that matters on the days the instrument is someone else's.
    assert lts.config_from_yaml_dict(on, enabled_override=False).enabled is False


def test_unknown_keys_are_ignored() -> None:
    cfg = lts.config_from_yaml_dict({"win_host": "u@h", "enabled": True, "not_a_field": 3})
    assert cfg.win_host == "u@h"
    assert not hasattr(cfg, "not_a_field")


def test_session_paths_use_windows_separators() -> None:
    session = lts.LaserTrackerSession(_cfg(win_root=r"D:\lt"), session_id="lt_x")
    assert session.win_dir == r"D:\lt\lt_x"
    # --out is the session DIRECTORY: the logger writes <out>/<session>.* flat
    # and seal rglobs a directory, so pointing it at the root would sweep every
    # past session into one manifest.
    assert session.probe_stop_file == r"D:\lt\lt_x\STOP_PROBE"
    assert session.logger_stop_file == r"D:\lt\lt_x\STOP_LOGGER"


def test_disabled_start_is_a_quiet_false() -> None:
    session = lts.LaserTrackerSession(lts.config_from_yaml_dict(None))
    assert session.start() is False
    assert session.last_error == ""


def test_unreachable_host_fails_without_raising() -> None:
    # The whole contract in one test: an unreachable capture PC returns False
    # and records why, so the recorder can warn and carry on with nine cameras.
    session = lts.LaserTrackerSession(_cfg(win_host="user@192.0.2.1"))
    assert session.start() is False
    assert session.last_error
    assert session.status().connected is False


def test_episode_calls_are_inert_when_not_connected() -> None:
    session = lts.LaserTrackerSession(_cfg())
    assert session.start_recording(0, 123.0) is False
    assert session.stop_recording() == {}
    assert session.stop() == {"session_id": session.session_id, "landed_to": "", "sealed": False}


def test_start_recording_makes_no_remote_call() -> None:
    """Start Episode is on the same critical path as the cameras.

    The tracker stream has been running since Connect precisely so that this is
    bookkeeping: the logger needs 15-16 s to produce its first sample, and no
    episode can wait for that.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    calls = []
    session._run = lambda *a, **k: calls.append(a)  # type: ignore[method-assign]

    assert session.start_recording(3, 999.0) is True
    assert calls == []


def test_a_stalled_stream_is_caught_at_episode_stop() -> None:
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 5000
    session._count_rows = lambda _name: 5000  # type: ignore[method-assign]
    session.episode_window_stats = lambda **_k: _stats()  # type: ignore[method-assign]

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is False
    assert rec["incomplete"] is True
    assert "did not advance" in rec["error"]

    # And an advancing, locked-on stream is quiet.
    session._count_rows = lambda _name: 9000  # type: ignore[method-assign]
    session.start_recording(1, 2.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is True
    assert rec["error"] == ""
    assert rec["incomplete"] is False
    assert rec["rt_rows_total_at_stop"] == 9000


def test_a_full_rate_stream_of_nothing_is_not_success() -> None:
    """81343 rows, 0 dropped, and not one measurement -- the first real session.

    Row count says the plumbing works. Only valid/tracking says the instrument
    could see anything, and with no SMR in the beam the tracker still streams a
    full 1 kHz of valid=0 tracking=0 dist=0.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 0
    session._count_rows = lambda _name: 81343  # type: ignore[method-assign]
    session.episode_window_stats = lambda **_k: _stats(  # type: ignore[method-assign]
        valid=0.0, tracking=0.0, dist_min_mm=-1.0, dist_max_mm=-1.0, usable_fraction=0.0
    )
    # op_mode 2 throughout, which is what the first real session recorded.
    session.gimbal_mode = "Position"

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is True          # rows advanced ...
    assert rec["beam_tracking_fraction"] == 0.0    # ... and measured nothing
    assert "not measuring" in rec["error"]
    # Naming the mode is the point: "Position" says an SMR in the beam would
    # still not be acquired, which "0% tracking" alone does not.
    assert "Position" in rec["error"]
    assert rec["incomplete"] is True


def test_tracking_without_measuring_is_caught() -> None:
    """2026-09-20's two lost episodes, as a test.

    The gimbal follows the SMR, az/el/dist keep updating, the beam marker says
    acquired and the device row is green -- and x/y/z are zeroed for every
    sample away from the nest, because the distance reference Home left behind
    does not survive a beam break and no measurement routine was running to
    rebuild it. Every liveness check we had passed.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 0
    session._count_rows = lambda _name: 35322  # type: ignore[method-assign]
    session.episode_window_stats = lambda **_k: _stats(  # type: ignore[method-assign]
        valid=0.465, tracking=0.861, dist_min_mm=157.7, dist_max_mm=158.9,
        nest_fraction=0.465, usable_fraction=0.0,
    )

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is True
    assert "tracked but did not measure" in rec["error"]
    # Name the fix, not just the symptom.
    assert "measure-ms" in rec["error"]


def test_a_motionless_target_is_not_a_trajectory() -> None:
    """Valid, dense, and all at one distance: the target never moved."""
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 0
    session._count_rows = lambda _name: 20000  # type: ignore[method-assign]
    session.episode_window_stats = lambda **_k: _stats(  # type: ignore[method-assign]
        dist_min_mm=1157.7, dist_max_mm=1158.2
    )

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert "did not move" in rec["error"]
    # A dwell recording looks exactly like this and is correct, so the wording
    # has to say which one it is rather than call it a fault -- and it is kept.
    assert "dwell" in rec["error"]
    assert rec["incomplete"] is False
    assert rec["beam_dist_min_mm"] == 1157.7


def test_episode_stats_cover_the_episode_not_the_tail() -> None:
    """The window asked for is the episode's own span, in seconds before now."""
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 0
    session._count_rows = lambda _name: 35322  # type: ignore[method-assign]
    seen = {}

    def _window(*, ago_start_s, ago_end_s):
        seen["span"] = ago_start_s - ago_end_s
        seen["ago_end_s"] = ago_end_s
        return _stats()

    session.episode_window_stats = _window  # type: ignore[method-assign]
    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert abs(seen["span"] - (rec["t_end_wall_s"] - rec["t_start_wall_s"])) < 0.01
    assert 0.0 <= seen["ago_end_s"] < 1.0


_RT_HEADER = (
    "seq,ctrl_ts_ms,qpc,az_deg,el_deg,dist_mm,x_mm,y_mm,z_mm,valid,tracking,warmed,op_mode,"
    "body_temp_c,level_x,level_y,global_ts,lock,ranged\n"
)
_QPC_HZ = 10_000_000


def _rt_csv(path: Path, ms: list[int], dist_mm) -> Path:
    """An rt.csv with one row per controller ms in ``ms``; QPC ticks follow the controller."""
    lines = [_RT_HEADER]
    for i, t in enumerate(ms):
        d = dist_mm(t) if callable(dist_mm) else dist_mm
        lines.append(f"{i},{t},{t * (_QPC_HZ // 1000)},0,0,{d},1,2,3,1,1,1,1,30,0,0,0,1,1\n")
    path.write_text("".join(lines), encoding="ascii")
    return path


def _run_stats_code(rt: Path, *, now_ms: int, ago_start_s: float, ago_end_s: float) -> dict:
    """``_EPISODE_STATS_CODE`` as the capture PC runs it, with the PC's clock handed in."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-I", "-c", lts._EPISODE_STATS_CODE, str(rt), str(10**8),
         f"{ago_start_s}", f"{ago_end_s}", "300", str(_QPC_HZ), str(now_ms * (_QPC_HZ // 1000))],
        capture_output=True, text=True, check=True,
    )
    return lts.parse_episode_stats(out.stdout)


def test_a_stream_in_gaps_is_measured_over_the_episode_not_the_last_rows(tmp_path) -> None:
    """2026-10-09 ep0, as a test: RadianCAL-Pro left ~1 s of stream in every 5.

    The last (duration in ms) rows reached a minute back, read 100% valid, and the
    episode was saved. On the PC's clock the 15 s episode holds 3 s of stream.
    """
    before = list(range(0, 60_000))  # a minute of full-rate stream before StartEpisode
    episode = [t for t in range(60_000, 75_000) if (t - 60_000) % 5000 < 1000]
    rt = _rt_csv(tmp_path / "s.rt.csv", before + episode, 1200.0)
    st = _run_stats_code(rt, now_ms=76_000, ago_start_s=16.0, ago_end_s=1.0)
    assert abs(st["window_s"] - 15.0) < 1e-6
    assert abs(st["coverage"] - 0.2) < 0.01
    assert st["max_gap_ms"] >= 3999
    error, incomplete = lts.episode_verdict(st, min_usable_fraction=0.8, nest_max_mm=300.0)
    assert incomplete is True
    assert "RadianCAL" in error
    assert "covered 20%" in error


def test_the_home_nest_counts_as_no_measurement_of_the_target(tmp_path) -> None:
    """Ready means homed and locked -- on the nest, ~158 mm from the head."""
    rt = _rt_csv(tmp_path / "s.rt.csv", list(range(0, 20_000)), lambda t: 157.7 if t < 8000 else 1200.0 + t / 100)
    st = _run_stats_code(rt, now_ms=20_000, ago_start_s=20.0, ago_end_s=0.0)
    assert abs(st["nest_fraction"] - 0.4) < 0.01
    assert abs(st["usable_fraction"] - 0.6) < 0.01
    error, incomplete = lts.episode_verdict(st, min_usable_fraction=0.8, nest_max_mm=300.0)
    assert incomplete is True
    assert "home nest" in error

    # A healthy episode passes quietly.
    st = _run_stats_code(rt, now_ms=20_000, ago_start_s=10.0, ago_end_s=0.0)
    assert st["usable_fraction"] > 0.99
    assert lts.episode_verdict(st, min_usable_fraction=0.8, nest_max_mm=300.0) == ("", False)


def test_an_unreadable_window_is_reported_but_never_thrown_away() -> None:
    st = lts.parse_episode_stats("ssh: connect to host 10.0.0.1 port 22: timed out")
    assert st["rows"] == -1
    error, incomplete = lts.episode_verdict(st, min_usable_fraction=0.8, nest_max_mm=300.0)
    assert incomplete is False
    assert "could not read" in error


def test_the_stats_code_survives_cmd_quoting() -> None:
    # It travels as python -c "<code>" through cmd.exe.
    assert '"' not in lts._EPISODE_STATS_CODE
    assert "%" not in lts._EPISODE_STATS_CODE


def test_a_discarded_take_is_marked_in_the_episode_index() -> None:
    """The next take reuses the episode index; the index must not carry both as one."""
    session = lts.LaserTrackerSession(_cfg())
    session._connected = True
    session._rows_mark = 0
    session._count_rows = lambda _name: 9000  # type: ignore[method-assign]
    session.episode_window_stats = lambda **_k: _stats(  # type: ignore[method-assign]
        coverage=0.13, usable_fraction=0.13, max_gap_ms=4442.0
    )
    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["incomplete"] is True
    session.mark_discarded("laser_tracker_incomplete")
    assert session._episodes[-1].discarded is True
    assert session._episodes[-1].discard_reason == "laser_tracker_incomplete"


def test_the_gate_is_configurable_from_the_yaml_block() -> None:
    cfg = _cfg(min_usable_fraction=0.9, discard_incomplete=False, nest_max_mm=250)
    assert cfg.min_usable_fraction == 0.9
    assert cfg.discard_incomplete is False
    assert cfg.nest_max_mm == 250.0


def test_position_mode_gets_its_own_instruction() -> None:
    assert lts._OP_MODES[2] == "Position"
    assert lts._OP_MODES[1] == "Tracking"
    assert lts._OP_MODES[3] == "TrackIdle"


def test_describe_names_the_instrument_from_the_instrument() -> None:
    session = lts.LaserTrackerSession(_cfg())
    # Before the logger has spoken, fall back to addresses rather than invent.
    assert "192.168.0.168" in session.describe()

    session.device_info = {
        "model": "Radian Pro", "sn": "65201", "fw": "7.402", "accessory": "none"
    }
    described = session.describe()
    assert "Radian Pro" in described
    assert "S/N 65201" in described
    assert "no accessory" in described


def test_status_is_renderable_before_anything_happens() -> None:
    status = lts.LaserTrackerSession(_cfg()).status()
    assert status.enabled is True
    assert status.connected is False
    assert status.episodes == []
    assert Path(status.win_session_dir).name or status.win_session_dir


# --- gateway side: the row, and what colours it ------------------------------

def _thor_config():
    import yaml

    return yaml.safe_load(Path("tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml").read_text())


def test_row_exists_even_when_the_session_is_off() -> None:
    """"Off" and "not installed" must not look the same.

    A row that appeared only when enabled would tell an operator nothing on the
    day they forgot to tick the box.
    """
    from tools.data_collection_gui import gateway

    config = _thor_config()
    assert config["laser_tracker"]["enabled"] is False
    rows = [d for d in gateway._device_statuses(config) if d["kind"] == "laser_tracker"]
    assert len(rows) == 1
    assert rows[0]["id"] == "laser_tracker"
    assert "one client at a time" in rows[0]["detail"]


def test_no_row_when_the_rig_has_no_tracker() -> None:
    from tools.data_collection_gui import gateway

    config = {k: v for k, v in _thor_config().items() if k != "laser_tracker"}
    assert [d for d in gateway._device_statuses(config) if d["kind"] == "laser_tracker"] == []


def test_set_device_state_touches_exactly_one_row() -> None:
    from tools.data_collection_gui import gateway

    rows = gateway._device_statuses(_thor_config())
    state = type("S", (), {"devices": rows})()
    gateway._set_device_state(state, "laser_tracker", "error", "SA is holding the tracker")

    tracker = [d for d in rows if d["kind"] == "laser_tracker"][0]
    assert tracker["state"] == "error"
    assert tracker["detail"] == "SA is holding the tracker"
    assert {d["state"] for d in rows if d["kind"] != "laser_tracker"} == {"idle"}


def test_set_device_state_ignores_an_undeclared_device() -> None:
    # A recorder talking about hardware the config never declared must not be
    # able to invent a row the operator did not ask for.
    from tools.data_collection_gui import gateway

    rows = gateway._device_statuses(_thor_config())
    before = [dict(d) for d in rows]
    state = type("S", (), {"devices": rows})()
    gateway._set_device_state(state, "not_a_device", "error", "nope")
    assert rows == before


def test_repo_root_is_found_not_counted(tmp_path) -> None:
    """A moved file must not break path resolution with an opaque error.

    ``parents[3]`` raised ``IndexError(3)``, and that exception's whole string
    form is ``3`` -- so the operator's warning read "connect failed: 3".
    """
    session = lts.LaserTrackerSession(_cfg(), repo_root=tmp_path)
    assert session.repo_root == tmp_path
    # and the default still finds the real checkout from the installed location
    assert (lts.LaserTrackerSession(_cfg()).repo_root / "third_party").is_dir()


def test_failure_message_is_a_diagnosis_not_a_number() -> None:
    session = lts.LaserTrackerSession(_cfg(win_host="user@192.0.2.1"))
    assert session.start() is False
    # "connect failed: 3" is what an IndexError used to produce. Whatever the
    # cause, the operator must get words.
    assert len(session.last_error) > 20
    assert not session.last_error.rstrip().endswith(": 3")


def test_windows_oem_output_never_breaks_a_connect() -> None:
    """cmd.exe answers in the OEM codepage, and 0xD5 is not valid UTF-8.

    `del` on a missing stop-file replies 找不到文件 on a Chinese install, which
    is routine; decoding it with utf-8 raised and failed the whole Connect.
    """
    # The exact bytes from the field report: 找不到文件 in CP936.
    assert lts._decode("找不到文件".encode("cp936")) == "找不到文件"
    assert lts._decode(b"ok") == "ok"
    # Anything at all decodes rather than raising -- latin-1 terminates the chain.
    assert lts._decode(bytes(range(256)))
    assert lts._decode(b"\xd5\x00\xff")


def test_an_empty_home_nest_is_a_state_not_a_failure() -> None:
    """The nest being empty is something the operator is about to change.

    Failing Connect for it would make them reconnect for a condition that the
    tracker's own home-retry timer clears in ten seconds.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._logger_tail.append("home failed: NoSmrAtHomePosition (21)")
    session._logger_proc = None
    session._count_rows = lambda _name: 5000  # type: ignore[method-assign]
    session.beam_quality = lambda **_k: (0.0, 0.0)  # type: ignore[method-assign]

    assert session._await_logger() is True       # Connect succeeds ...
    assert session.beam_ready is False           # ... and says it is not ready
    assert "waiting for the SMR" in session.beam_summary()


def test_beam_transitions_are_read_from_the_logger() -> None:
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc([
        "beam: waiting (NoSmrAtHomePosition)\n",
        "beam: acquired\n",
    ]))
    assert session.beam_ready is True
    assert "NOT homed" in session.beam_summary()
    session._read_logger_stdout(_Proc(["home: ok\n"]))
    assert session.beam_summary() == "homed, locked on the SMR"

    session._read_logger_stdout(_Proc(["beam: lost\n"]))
    assert session.beam_ready is False
    assert "reacquiring" in session.beam_summary()


def test_a_marker_glued_to_the_progress_counter_is_still_read() -> None:
    """The logger's progress counter shares stdout and writes no newline.

    Every marker after the first therefore arrives with the counter stuck to
    its front. Parsing with `startswith` read the SMR as missing while the
    tracker was locked on it and the device row stayed amber -- 2026-09-20,
    with the operator looking at a green light on the instrument itself.
    """
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc([
        "rows 12000  dropped 0   beam: acquired\n",
    ]))
    assert session.beam_ready is True

    session._read_logger_stdout(_Proc(["rows 24000  dropped 0   beam: lost\n"]))
    assert session.beam_ready is False

    # And the identity line, which arrives the same way.
    session._read_logger_stdout(_Proc([
        "rows 100  dropped 0   device: model=Radian_Pro sn=65201 fw=7.402\n",
    ]))
    assert session.device_info.get("sn") == "65201"


def test_the_progress_counter_never_crowds_out_a_real_error() -> None:
    """The tail is what an operator is shown when the logger dies."""
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    lines = ["connection failed: CommunicationFailed (13)\n"]
    lines += [f"rows {i * 1000}  dropped 0\n" for i in range(200)]
    session._read_logger_stdout(_Proc(lines))
    assert any("CommunicationFailed" in line for line in session._logger_tail)


def test_the_empty_nest_message_tells_the_operator_what_to_do() -> None:
    session = lts.LaserTrackerSession(_cfg())
    session.beam_status = "waiting (NoSmrAtHomePosition)"
    summary = session.beam_summary()
    assert "home nest" in summary
    assert "retries automatically" in summary



def test_a_busy_tracker_is_named_as_such() -> None:
    """The SDK reports "one client already has it" as CommunicationFailed.

    That reads like a network fault and sends people to check cables, when the
    fix is to close SA. Observed 2026-09-20 while the tracker was being Homed
    from RadianCAL.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._logger_tail.append("connection failed: CommunicationFailed (13)")

    class _Dead:
        stdout = None

        def poll(self):
            return 1

    session._logger_proc = _Dead()  # type: ignore[assignment]
    session._count_rows = lambda _name: 0  # type: ignore[method-assign]
    assert session._await_logger() is False
    assert "one client at a time" in session.last_error
    assert "close SA" in session.last_error


def test_homing_is_off_unless_asked_for() -> None:
    """Driving a shared precision instrument is never a default."""
    cfg = lts.config_from_yaml_dict({"win_host": "u@h", "enabled": True})
    assert cfg.home_on_connect is False
    assert cfg.smr_size == ""


def test_home_without_a_declared_smr_size_is_refused_not_guessed() -> None:
    """1.5" and 7/8" are both plausible; the wrong one homes on the wrong radius."""
    session = lts.LaserTrackerSession(_cfg(home_on_connect=True, smr_size=""))
    sent = []

    class _Proc:
        stdout = None

        def poll(self):
            return None

    session._run = lambda cmd, **k: sent.append(cmd)  # type: ignore[method-assign]
    session._spawn = lambda cmd: (sent.append(cmd), _Proc())[1]  # type: ignore[method-assign]

    session._spawn_logger()
    logger_cmd = next(c for c in sent if "lt_realtime_logger" in c)
    assert "--home" not in logger_cmd
    assert "smr_size is empty" in session.last_error


def test_a_declared_smr_size_reaches_the_logger() -> None:
    session = lts.LaserTrackerSession(
        _cfg(home_on_connect=True, smr_size="1.5", adm_offset_mm=0.25)
    )
    sent = []

    class _Proc:
        stdout = None

        def poll(self):
            return None

    session._run = lambda cmd, **k: sent.append(cmd)  # type: ignore[method-assign]
    session._spawn = lambda cmd: (sent.append(cmd), _Proc())[1]  # type: ignore[method-assign]

    session._spawn_logger()
    logger_cmd = next(c for c in sent if "lt_realtime_logger" in c)
    assert "--home 1.5" in logger_cmd
    assert "--adm-offset 0.25" in logger_cmd
    assert session.last_error == ""


# --- cold start after a power cut (2026-09-21) --------------------------------


def test_warmup_is_not_reported_as_an_smr_problem() -> None:
    """TrackerNotWarmedUp was shown as "waiting for the SMR".

    After the 2026-09-21 power cut that sentence sent the operator to the nest
    while the only fix was to wait for the laser, and two Connects that had in
    fact succeeded were abandoned as failures.
    """
    session = lts.LaserTrackerSession(_cfg())
    session.beam_status = "waiting (TrackerNotWarmedUp (22))"
    summary = session.beam_summary()
    assert "warming up" in summary
    assert "SMR" not in summary
    assert "retries automatically" in summary


def test_an_unrecognised_home_error_does_not_blame_the_smr() -> None:
    session = lts.LaserTrackerSession(_cfg())
    session.beam_status = "waiting (AdmLowIntensity (23))"
    summary = session.beam_summary()
    assert "AdmLowIntensity (23)" in summary
    assert "for the SMR" not in summary


def _dead_logger_session(tail_line: str):
    session = lts.LaserTrackerSession(_cfg())
    session._logger_tail.append(tail_line)

    class _Dead:
        stdout = None

        def poll(self):
            return 1

    session._logger_proc = _Dead()  # type: ignore[assignment]
    session._count_rows = lambda _name: 0  # type: ignore[method-assign]
    return session


def test_a_failed_index_search_points_at_the_servo_switch() -> None:
    session = _dead_logger_session("connection failed: IndexSearchFailed (43)")
    assert session._await_logger() is False
    assert "index search" in session.last_error
    assert "Servo" in session.last_error


def test_communication_failure_also_names_a_booting_controller() -> None:
    """Right after power-on the same code means "not up yet", not "busy"."""
    session = _dead_logger_session("connection failed: CommunicationFailed (13)")
    assert session._await_logger() is False
    assert "still booting" in session.last_error
    assert "one client at a time" in session.last_error


def test_a_logger_that_never_streams_is_told_to_stop(monkeypatch) -> None:
    """Giving up on Connect must not leave the logger holding the tracker.

    The logger only checks its stop-file once connected, so one still in the
    SDK handshake at the deadline would otherwise finish connecting later and
    keep the instrument's single client slot for up to ``session_cap_s``.
    """
    import subprocess

    session = lts.LaserTrackerSession(_cfg())
    ran: list[str] = []

    class _Stuck:
        terminated = False

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("ssh", timeout)

        def terminate(self):
            _Stuck.terminated = True

    def _fake_run(cmd, **_k):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(session, "_ensure_responder", lambda: None)
    monkeypatch.setattr(session, "_spawn_probe", lambda: True)
    monkeypatch.setattr(session, "_spawn_logger", lambda: setattr(session, "_logger_proc", _Stuck()))
    monkeypatch.setattr(session, "_await_probe", lambda: True)
    monkeypatch.setattr(session, "_await_logger", lambda: False)
    monkeypatch.setattr(session, "_run", _fake_run)

    assert session.start() is False
    assert any("type nul" in c and "STOP_LOGGER" in c for c in ran)
    assert _Stuck.terminated
    assert session._logger_proc is None


def test_a_ranged_lock_is_reported_with_the_step_it_moved() -> None:
    """The operator needs the step while judging a catch, not after the session.

    W2 (2026-09-22): a lock caught 337 mm long keeps it until the beam is lost,
    so "ranged, step 337.2 mm" is the line that says this catch was bad and the
    fit built on it would have been wrong.
    """
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc([
        "beam: acquired\n",
        "lock: 0 ranged, step 337.2 mm\n",
        "rows 12000  dropped 0   lock: 1 ranged, step 0.01 mm (target was moving)\n",
        "lock: 2 NOT ranged (LaserBeamBroken)\n",
    ]))
    assert session.locks_ranged == 2 and session.locks_not_ranged == 1
    assert session.range_status == "2 NOT ranged (LaserBeamBroken)"
    assert session.status().range_status.startswith("2 NOT ranged")


def test_a_session_that_never_homed_says_so() -> None:
    """Locked-on is not homed.

    W2 (2026-09-21) ran green from start to finish and never homed -- of its
    679342 valid samples not one was at the home nest -- and every beam lock
    inherited a range hundreds of mm off. The marker is separate for that
    reason: "beam: acquired" must not be read as "this session has a range".
    """
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc(["beam: acquired\n", "lock: 0 ranged, step 0 mm\n"]))
    assert session.beam_ready is True
    assert session.homed is False and session.status().homed is False

    session._read_logger_stdout(_Proc(["home: ok\n"]))
    assert session.homed is True and session.status().homed is True


def test_start_is_gated_on_home_not_only_on_the_beam() -> None:
    """`ready` is what the recorder sends as LT_BEAM ready, and so what gates Start."""
    session = lts.LaserTrackerSession(_cfg(home_on_connect=True, smr_size="1.5"))

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc(["beam: acquired\n"]))
    assert session.ready is False
    assert "home nest" in session.beam_summary()

    session._read_logger_stdout(_Proc(["home: ok\n"]))
    assert session.ready is True

    session._read_logger_stdout(_Proc(["beam: lost\n"]))
    assert session.ready is False  # homed stays, but a blind beam still blocks


def test_a_beam_break_after_home_blocks_start_until_the_nest_re_homes() -> None:
    """Pivot lt_20260923_062953 re-caught 14 mm out of the nest carried +4.3 mm."""
    session = lts.LaserTrackerSession(_cfg(home_on_connect=True, smr_size="1.5"))

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    session._read_logger_stdout(_Proc(["home: ok\n", "range: lock 0 absolute (home)\n", "beam: acquired\n"]))
    assert session.ready is True and session.beam_broken is False

    session._read_logger_stdout(_Proc([
        "beam: lost\n",
        "rows 9000  dropped 0   range: lock 1 not absolute -- the beam broke; put the SMR "
        "back in the home nest to re-home\n",
        "beam: acquired\n",
    ]))
    assert session.beam_broken is True
    assert session.ready is False
    assert "home nest" in session.beam_summary()

    session._read_logger_stdout(_Proc(["recover: re-home ok\n", "range: lock 2 absolute (home)\n"]))
    assert session.ready is True and session.beam_broken is False


def test_a_logger_that_does_not_report_absoluteness_is_gated_on_home_alone() -> None:
    """An exe from before 2026-09-23 prints no `range:` lines; it must still turn green."""
    session = lts.LaserTrackerSession(_cfg(home_on_connect=True, smr_size="1.5"))
    session.beam_ready, session.homed = True, True
    assert session.range_absolute is None
    assert session.ready is True


def test_a_connect_that_does_not_home_says_why_it_never_gets_ready() -> None:
    session = lts.LaserTrackerSession(_cfg(home_on_connect=False))
    session.beam_ready = True
    assert session.ready is False
    assert "home_on_connect is off" in session.beam_summary()


def test_recovery_attempts_reach_the_recorder_log(caplog) -> None:
    """They used to live only in the in-memory tail and die with the process."""
    session = lts.LaserTrackerSession(_cfg())

    class _Proc:
        def __init__(self, lines):
            self.stdout = iter(lines)

    with caplog.at_level("INFO", logger=lts.logger.name):
        session._read_logger_stdout(_Proc([
            "rows 5  dropped 0   recover: re-range TargetNotFound (7)\n",
            "recover: tracking at the home nest without a range -- homing\n",
            "home: nest at az -28.41 el -20.09 deg\n",
        ]))
    text = caplog.text
    assert "recover: re-range TargetNotFound (7)" in text
    assert "at the home nest without a range" in text
    assert "home: nest at az" in text
    assert session.homed is False  # the nest line is not the home marker


def test_the_recorder_discards_only_task_episodes_short_of_laser_data() -> None:
    from tools.thor.gmsl2 import thor_record

    bad = {"incomplete": True, "error": "only 13% of this 15.3 s episode has a ranged laser point"}
    cfg = _cfg()
    assert "13%" in thor_record._tracker_reject_reason(bad, cfg, None)
    assert "13%" in thor_record._tracker_reject_reason(bad, cfg, {"purpose": "task"})
    # A calibration capture is judged by its own checks, never thrown away here.
    assert thor_record._tracker_reject_reason(bad, cfg, {"purpose": "calibration_tracker_mount"}) == ""
    assert thor_record._tracker_reject_reason(bad, _cfg(discard_incomplete=False), None) == ""
    # Unknown (unreadable window) and a dwell are kept.
    assert thor_record._tracker_reject_reason({"incomplete": False, "error": "could not read"}, cfg, None) == ""

    summary = thor_record._tracker_episode_summary(
        4, {"incomplete": True, "usable_fraction": 0.13, "stream_coverage": 0.13, "max_gap_ms": 4442.0,
            "nest_fraction": 0.0, "beam_window_s": 15.3, "error": "x"}, rejected=True, kept=False,
    )
    assert summary == {"episode": 4, "rejected": True, "kept": False, "incomplete": True, "window_s": 15.3, "usable": 0.13,
                       "coverage": 0.13, "max_gap_ms": 4442.0, "nest": 0.0, "error": "x"}


def test_segment_points_are_this_dwell_not_the_pose_before_it(tmp_path) -> None:
    """With the stream in gaps, the last N rows reach back to the previous parked pose."""
    import subprocess
    import sys

    before = list(range(0, 30_000))  # previous pose, full rate
    dwell = [t for t in range(30_000, 40_000) if (t - 30_000) % 5000 < 1000]  # this one, in gaps
    rt = _rt_csv(tmp_path / "s.rt.csv", before + dwell, lambda t: 500.0 if t < 30_000 else 900.0)
    # _rt_csv writes x/y/z = 1/2/3 for every row; put the distance in x to tell the poses apart.
    rt.write_text(
        "".join(
            line if i == 0 else ",".join(f[:6] + [f[5]] + f[7:])
            for i, (line, f) in enumerate((ln, ln.split(",")) for ln in rt.read_text().splitlines(True))
        )
    )
    now_ms = 40_500
    out = subprocess.run(
        [sys.executable, "-I", "-c", lts._SEGMENT_POINTS_CODE, str(rt), str(10**8), "10.5", "0.5", "3000",
         str(_QPC_HZ), str(now_ms * (_QPC_HZ // 1000))],
        capture_output=True, text=True, check=True,
    )
    pts = lts.parse_segment_points(out.stdout)
    assert len(pts) == 2000
    assert {p[0] for p in pts} == {900.0}


def test_a_beam_lost_mid_take_is_named_as_such_not_as_a_missing_routine() -> None:
    """2026-10-13 ep0: lost 5 s in, spiral search timed out, re-homed to the nest.

    Some ranged points off the nest exist, so the measurement routine was running;
    "is --measure-ms running?" sent the operator after the wrong thing.
    """
    st = _stats(valid=0.386, tracking=0.561, dist_min_mm=157.7, dist_max_mm=1205.4,
                window_s=20.3, nest_fraction=0.145, usable_fraction=0.241)
    error, incomplete = lts.episode_verdict(st, min_usable_fraction=0.8, nest_max_mm=300.0)
    assert incomplete is True
    assert "measure-ms" not in error
    assert "line of sight" in error
