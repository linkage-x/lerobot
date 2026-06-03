"""Unit tests for the Connect-time auto-recover logic in thor_record.

The recorder runs `tools/thor/gmsl2/recover_argus.sh` when nvargus-daemon is
in a wedged state — either every PersistentCameraSession worker died or only
a small fraction of cameras came up — so the operator doesn't have to ssh
into Thor and run the script by hand. This module exercises the three free
functions that make that decision testable without spawning a subprocess:

  * ``_auto_recover_from_yaml``       — YAML defaults + overrides
  * ``_resolve_recover_sdk_dir``      — explicit sdk_dir vs hardware_sync
                                        fallback, relative vs absolute
  * ``_should_trigger_recovery``      — threshold semantics
  * ``_run_recover_argus``            — rc handling with injected runner
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from tools.thor.gmsl2 import thor_record as tr


# ---------------------------------------------------------------------------
# _auto_recover_from_yaml
# ---------------------------------------------------------------------------


def test_auto_recover_from_yaml_none_returns_defaults():
    cfg = tr._auto_recover_from_yaml(None)
    assert cfg.enabled is True
    assert cfg.sdk_dir is None
    assert cfg.threshold_fraction == 0.6
    assert cfg.max_attempts == 1
    assert cfg.timeout_s == 300.0


def test_auto_recover_from_yaml_empty_dict_returns_defaults():
    cfg = tr._auto_recover_from_yaml({})
    assert cfg.enabled is True
    assert cfg.threshold_fraction == 0.6


def test_auto_recover_from_yaml_picks_up_overrides():
    cfg = tr._auto_recover_from_yaml({
        "enabled": False,
        "sdk_dir": "~/Desktop/SG16A_AGTH_G3Y_A1",
        "threshold_fraction": 0.5,
        "max_attempts": 2,
        "timeout_s": 120,
    })
    assert cfg.enabled is False
    assert cfg.sdk_dir == "~/Desktop/SG16A_AGTH_G3Y_A1"
    assert cfg.threshold_fraction == 0.5
    assert cfg.max_attempts == 2
    assert cfg.timeout_s == 120.0


def test_auto_recover_from_yaml_ignores_non_dict():
    # Defensive: a malformed YAML where someone wrote `auto_recover: true`
    # instead of a block should not crash, just fall back to defaults.
    cfg = tr._auto_recover_from_yaml("not a dict")  # type: ignore[arg-type]
    assert cfg.enabled is True
    assert cfg.threshold_fraction == 0.6


# ---------------------------------------------------------------------------
# _resolve_recover_sdk_dir
# ---------------------------------------------------------------------------


def test_resolve_recover_sdk_dir_uses_explicit_when_set(tmp_path):
    explicit = tmp_path / "vendor_sdk"
    explicit.mkdir()
    auto = tr.AutoRecoverConfig(sdk_dir=str(explicit))
    resolved = tr._resolve_recover_sdk_dir(
        auto, fallback_sdk_dir="tools/thor/gmsl2/sdk", repo_root=tmp_path,
    )
    assert resolved == explicit.resolve()


def test_resolve_recover_sdk_dir_falls_back_to_hardware_sync(tmp_path):
    auto = tr.AutoRecoverConfig(sdk_dir=None)
    resolved = tr._resolve_recover_sdk_dir(
        auto, fallback_sdk_dir="tools/thor/gmsl2/sdk", repo_root=tmp_path,
    )
    assert resolved == (tmp_path / "tools" / "thor" / "gmsl2" / "sdk").resolve()


def test_resolve_recover_sdk_dir_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    auto = tr.AutoRecoverConfig(sdk_dir="~/some_sdk")
    resolved = tr._resolve_recover_sdk_dir(
        auto, fallback_sdk_dir="ignored", repo_root=tmp_path,
    )
    assert resolved == (tmp_path / "some_sdk")


# ---------------------------------------------------------------------------
# _should_trigger_recovery
# ---------------------------------------------------------------------------


def test_should_trigger_recovery_below_threshold():
    # 4/11 = 0.36 < 0.6 → trigger
    assert tr._should_trigger_recovery(4, 11, 0.6) is True


def test_should_trigger_recovery_exactly_at_threshold():
    # 6/10 = 0.6 not strictly less → do not trigger
    assert tr._should_trigger_recovery(6, 10, 0.6) is False


def test_should_trigger_recovery_above_threshold():
    # 9/11 = 0.82 → fine, do not trigger
    assert tr._should_trigger_recovery(9, 11, 0.6) is False


def test_should_trigger_recovery_zero_expected_returns_false():
    # Defensive: dividing by zero must never crash the recorder.
    assert tr._should_trigger_recovery(0, 0, 0.6) is False


def test_should_trigger_recovery_zero_active_with_real_expected_triggers():
    assert tr._should_trigger_recovery(0, 11, 0.6) is True


# ---------------------------------------------------------------------------
# _run_recover_argus  (injected runner — no real subprocess)
# ---------------------------------------------------------------------------


def _fake_completed(rc=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def test_run_recover_argus_missing_script_returns_false(tmp_path):
    # repo_root pointing at an empty tree: script does not exist.
    ok, tail = tr._run_recover_argus(tmp_path, tmp_path / "sdk")
    assert ok is False
    assert "recover_argus.sh not found" in tail


def _make_repo_with_script(tmp_path) -> Path:
    """Set up just enough of the repo layout for _run_recover_argus's
    is_file check to pass without actually running anything."""
    script_dir = tmp_path / "tools" / "thor" / "gmsl2"
    script_dir.mkdir(parents=True)
    script = script_dir / "recover_argus.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    script.chmod(0o755)
    return tmp_path


def test_run_recover_argus_returns_true_on_zero_rc(tmp_path):
    repo = _make_repo_with_script(tmp_path)
    runner_calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        runner_calls.append(cmd)
        assert "--sdk" in cmd
        return _fake_completed(rc=0, stdout="RECOVER_OK_SIDS=0,2,3\n")

    ok, tail = tr._run_recover_argus(
        repo, Path("/some/sdk"), _runner=runner,
    )
    assert ok is True
    assert tail == ""
    assert len(runner_calls) == 1
    assert runner_calls[0][0] == "bash"
    assert runner_calls[0][runner_calls[0].index("--sdk") + 1] == "/some/sdk"
    assert "--skip-kill" in runner_calls[0]


def test_run_recover_argus_returns_false_on_nonzero_rc(tmp_path):
    repo = _make_repo_with_script(tmp_path)

    def runner(cmd, **kwargs):
        return _fake_completed(rc=7, stderr="modprobe rejected")

    ok, tail = tr._run_recover_argus(repo, Path("/sdk"), _runner=runner)
    assert ok is False
    assert "modprobe rejected" in tail


def test_run_recover_argus_returns_false_on_timeout(tmp_path):
    repo = _make_repo_with_script(tmp_path)

    def runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

    ok, tail = tr._run_recover_argus(repo, Path("/sdk"), _runner=runner)
    assert ok is False
    assert "timed out" in tail


def test_run_recover_argus_returns_false_on_unexpected_exception(tmp_path):
    repo = _make_repo_with_script(tmp_path)

    def runner(cmd, **kwargs):
        raise OSError("permission denied")

    ok, tail = tr._run_recover_argus(repo, Path("/sdk"), _runner=runner)
    assert ok is False
    assert "permission denied" in tail


def test_run_recover_argus_caps_tail_to_400_chars(tmp_path):
    repo = _make_repo_with_script(tmp_path)
    huge = "x" * 5000

    def runner(cmd, **kwargs):
        return _fake_completed(rc=1, stderr=huge)

    ok, tail = tr._run_recover_argus(repo, Path("/sdk"), _runner=runner)
    assert ok is False
    assert len(tail) == 400


# ---------------------------------------------------------------------------
# _connect_session_with_deadline
# ---------------------------------------------------------------------------


class _ConnectSessionOk:
    def connect(self) -> None:
        return None


class _ConnectSessionRaises:
    def connect(self) -> None:
        raise RuntimeError("Argus refused CaptureSession")


class _ConnectSessionBlocks:
    def __init__(self) -> None:
        self.disconnect_called = False

    def connect(self) -> None:
        time.sleep(1.0)

    def disconnect(self) -> None:
        self.disconnect_called = True


def test_connect_session_with_deadline_success():
    ok, message = tr._connect_session_with_deadline(
        _ConnectSessionOk(), timeout_s=1.0,
    )

    assert ok is True
    assert message == ""


def test_connect_session_with_deadline_reports_runtime_error():
    ok, message = tr._connect_session_with_deadline(
        _ConnectSessionRaises(), timeout_s=1.0,
    )

    assert ok is False
    assert message == "Argus refused CaptureSession"


def test_connect_session_with_deadline_times_out_and_disconnects():
    session = _ConnectSessionBlocks()
    started = time.monotonic()

    ok, message = tr._connect_session_with_deadline(session, timeout_s=0.02)

    assert ok is False
    assert "connect exceeded global deadline" in message
    assert session.disconnect_called is True
    assert time.monotonic() - started < 0.5
