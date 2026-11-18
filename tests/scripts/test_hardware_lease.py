"""The hardware lease between production and the dev gateway (tools/data_collection_gui/hardware_lease.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.data_collection_gui import hardware_lease as hl


def test_no_file_means_production_holds_and_a_broken_file_means_nobody(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / hl.FILE_NAME
    assert hl.read_lease(path).holder == hl.PRODUCTION
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert hl.read_lease(path).holder == hl.UNKNOWN
    path.write_text(json.dumps({"holder": "someone"}))
    assert hl.read_lease(path).holder == hl.UNKNOWN
    production, dev = hl.HardwareLease(path, hl.PRODUCTION), hl.HardwareLease(path, hl.DEV)
    assert "unreadable" in production.refusal() and "unreadable" in dev.refusal()


def test_each_side_touches_hardware_only_while_it_holds_the_lease(tmp_path: Path) -> None:
    path = tmp_path / hl.FILE_NAME
    production, dev = hl.HardwareLease(path, hl.PRODUCTION), hl.HardwareLease(path, hl.DEV)
    assert production.refusal() is None
    assert "has not borrowed" in dev.refusal()
    production.transfer(hl.Lease(holder=hl.DEV, since="t", devUrl="http://127.0.0.1:8766"),
                        expect=(hl.PRODUCTION,), busy=lambda: [])
    assert dev.refusal() is None
    assert "lent to the dev gateway (http://127.0.0.1:8766" in production.refusal()
    assert production.enter() is not None and production._inflight == 0  # a refusal is not counted
    assert dev.payload()["mine"] is True and production.payload()["mine"] is False


def test_a_transfer_waits_for_requests_in_flight_and_for_busy_hardware(tmp_path: Path) -> None:
    path = tmp_path / hl.FILE_NAME
    production = hl.HardwareLease(path, hl.PRODUCTION)
    to_dev = hl.Lease(holder=hl.DEV)
    assert production.enter() is None  # e.g. a Connect that passed the guard and is still starting
    with pytest.raises(hl.LeaseRefusedError, match="still being answered"):
        production.transfer(to_dev, expect=(hl.PRODUCTION,), busy=lambda: [])
    production.exit()
    with pytest.raises(hl.LeaseRefusedError, match="the recorder is connected"):
        production.transfer(to_dev, expect=(hl.PRODUCTION,), busy=lambda: ["the recorder is connected"])
    with pytest.raises(hl.LeaseRefusedError, match="held by production, not by dev"):
        production.transfer(to_dev, expect=(hl.DEV,), busy=lambda: [])
    assert hl.read_lease(path).holder == hl.PRODUCTION and not path.exists()
    dropped = []
    production.transfer(to_dev, expect=(hl.PRODUCTION,), busy=lambda: [], before_write=lambda: dropped.append(1))
    assert dropped == [1] and hl.read_lease(path).holder == hl.DEV


def test_a_silent_dev_gateways_processes_are_found_by_their_tree_and_stopped(tmp_path: Path) -> None:
    tree = tmp_path / "sandboxes" / "dev"
    (tree / "tools").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    inside = subprocess.Popen(sleep, cwd=tree / "tools", stdin=subprocess.DEVNULL)
    other = subprocess.Popen(sleep, cwd=outside, stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while inside.pid not in hl.processes_in(tree) and time.monotonic() < deadline:
            time.sleep(0.05)
        found = hl.processes_in(tree)
        assert inside.pid in found and other.pid not in found
        assert hl.stop_processes(found) == []
        assert inside.wait(timeout=5) is not None
        assert other.poll() is None
    finally:
        for proc in (inside, other):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_the_sweep_never_covers_a_whole_filesystem_or_home(tmp_path: Path) -> None:
    assert hl.processes_in(Path("/")) == []
    assert hl.processes_in(Path("/home")) == []
    assert hl.stop_processes([0, 1, -1]) == []  # never kill(0), init or kill(-1)


def test_a_peer_that_does_not_answer_is_status_zero() -> None:
    assert hl.post_peer("http://127.0.0.1:9/api/hardware/lend", {}, timeout_s=1.0) == (0, {})
