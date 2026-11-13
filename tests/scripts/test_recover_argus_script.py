"""recover_argus.sh must stop the recorder and never the GUI gateway.

Until 2026-11-13 the manual recovery also `pkill`ed the gateway: every open GUI then showed
"connect: HTTP 500" (the Vite proxy's answer when nothing listens on the gateway port) until
someone redeployed. Its recorder pattern also never matched the recorder the gateway really
spawns (a script path, not `-m`). The command lines below are copied from Thor1.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "thor" / "gmsl2" / "recover_argus.sh"

GATEWAY_CMDLINES = [
    ".venv/bin/python -m tools.data_collection_gui.gateway --profile thor --config-path "
    "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml --datasets-root outputs/datasets --port 8765 "
    "--host 0.0.0.0 --repo-root /home/nvidia/lerobot",
    # a sandbox gateway
    "/home/nvidia/lerobot/.venv/bin/python -m tools.data_collection_gui.gateway --profile thor --port 8771 "
    "--repo-root /home/nvidia/sandboxes/alice --sandbox alice",
]
RECORDER_CMDLINES = [
    "/home/nvidia/lerobot/.venv/bin/python /home/nvidia/lerobot/tools/thor/gmsl2/thor_record.py "
    "--config-path=/home/nvidia/lerobot/tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml --skip-argus-probe",
    "python3 -m tools.thor.gmsl2.thor_record --config-path=x.yaml",
]
NATIVE_RECORDER_CMDLINES = [
    "/tmp/lerobot_argus_online_sync_video_recorder --sids 3,6,7 --frames 0 --fps 60 --codec h265",
]


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _assigned(name: str) -> str:
    match = re.search(rf"^{name}='([^']*)'$", _script_text(), re.MULTILINE)
    assert match, f"{name} is not assigned in {SCRIPT.name}"
    return match.group(1)


def _ere_matches(pattern: str, cmdline: str) -> bool:
    # pkill/pgrep -f use POSIX extended regexes; grep -E is the same dialect.
    return subprocess.run(["grep", "-Eq", "--", pattern], input=cmdline, text=True).returncode == 0


def test_no_kill_targets_the_gateway():
    for line in _script_text().splitlines():
        code = line.split("#", 1)[0]
        if re.search(r"\b(pkill|kill|killall)\b", code):
            assert "data_collection_gui" not in code, line
            assert "gateway" not in code, line


@pytest.mark.skipif(shutil.which("grep") is None, reason="needs grep -E")
def test_kill_patterns_hit_the_recorders_and_spare_the_gateway():
    recorder = _assigned("RECORDER_PATTERN")
    native = _assigned("NATIVE_RECORDER_PATTERN")
    for cmdline in RECORDER_CMDLINES:
        assert _ere_matches(recorder, cmdline), cmdline
    for cmdline in NATIVE_RECORDER_CMDLINES:
        assert _ere_matches(native, cmdline), cmdline
    for cmdline in GATEWAY_CMDLINES:
        assert not _ere_matches(recorder, cmdline), cmdline
        assert not _ere_matches(native, cmdline), cmdline


def test_the_kills_use_the_tested_patterns():
    text = _script_text()
    assert 'pkill -TERM -f "${RECORDER_PATTERN}"' in text
    assert 'pkill -KILL -f "${RECORDER_PATTERN}"' in text
    assert 'pkill -TERM -f "${NATIVE_RECORDER_PATTERN}"' in text
