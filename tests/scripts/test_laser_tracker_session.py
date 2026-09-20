"""The laser tracker must never be able to stop a recording.

It is a shared instrument on another machine, booked by other people and
unavailable more often than not. Every test here is about that: the driver is
allowed to fail, and it is not allowed to raise while doing so.
"""

from pathlib import Path

from tools.thor.gmsl2 import laser_tracker_session as lts


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
    session.beam_quality = lambda **_k: (1.0, 1.0)  # type: ignore[method-assign]

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is False
    assert "did not advance" in rec["error"]

    # And an advancing, locked-on stream is quiet.
    session._count_rows = lambda _name: 9000  # type: ignore[method-assign]
    session.start_recording(1, 2.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is True
    assert rec["error"] == ""
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
    session.beam_quality = lambda **_k: (0.0, 0.0)  # type: ignore[method-assign]

    session.start_recording(0, 1.0)
    rec = session.stop_recording()
    assert rec["stream_advanced"] is True          # rows advanced ...
    assert rec["beam_tracking_fraction"] == 0.0    # ... and measured nothing
    assert "not locked on a target" in rec["error"]


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


def test_a_connect_warning_survives_a_successful_connect() -> None:
    """The beam warning is the whole point of checking at Connect.

    start() used to clear last_error on success, which erased the one message
    that could still be acted on: acquire the SMR before recording.
    """
    session = lts.LaserTrackerSession(_cfg())
    session._ensure_responder = lambda: None  # type: ignore[method-assign]
    session._spawn_probe = lambda: True  # type: ignore[method-assign]
    session._spawn_logger = lambda: None  # type: ignore[method-assign]
    session._await_probe = lambda: True  # type: ignore[method-assign]

    def _await_logger() -> bool:
        session.beam_valid_fraction = session.beam_tracking_fraction = 0.0
        session.last_error = "tracker is streaming but NOT locked on a target"
        return True

    session._await_logger = _await_logger  # type: ignore[method-assign]
    assert session.start() is True
    assert "NOT locked" in session.last_error
