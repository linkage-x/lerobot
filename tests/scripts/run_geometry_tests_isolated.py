#!/usr/bin/env python
"""Run the rollout-geometry tests without importing the FR3 runtime module.

`tools/fr3/fr3_act_infer_real_runtime.py` imports `lerobot.policies`, whose `__init__` pulls
groot -> transformers, and this checkout's env has transformers pinned <1.0 against an installed
huggingface-hub 1.12.0. The import dies before any test is collected, so the geometry tests in
`test_fr3_act_infer_real.py` cannot run here at all -- an environment block, not a failure.

The reduction those tests cover is pure numpy over a list of samples and depends on none of that.
So this lifts the two halves out of the shipped files verbatim -- the `RolloutGeometryTrace`
section by text slice from the runtime, the test bodies by AST from the test file -- writes them
into a scratch directory and runs pytest there. Nothing is restated: edit either file and this
picks the edit up, which is the whole reason it extracts instead of duplicating.

    python tests/scripts/run_geometry_tests_isolated.py

Delete this the day the env can import the runtime; the tests it runs live in the real file.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tools" / "fr3" / "fr3_act_infer_real_runtime.py"
TESTS = ROOT / "tests" / "scripts" / "test_fr3_act_infer_real.py"

# The section starts at the comment explaining why the gripper command is the event signal and
# runs to the next top-level definition after the trace class. Anchored on those two landmarks
# rather than on line numbers so it survives edits above and below it.
SECTION_START = "# A rollout's gripper command is the event signal"
SECTION_CLASS = "class RolloutGeometryTrace"

HEADER = """from __future__ import annotations

import csv
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Imported for real rather than sliced out with the rest: `tools/fr3/dagger_takeover.py` pulls
# only numpy and lerobot's own rotation helper, so it costs nothing here and stays the single
# definition of how a takeover span is read back out of a trace.
from tools.fr3.dagger_takeover import expert_spans

fr3_act_infer_real_runtime = types.ModuleType('fr3_act_infer_real_runtime')
fr3_act_infer_real_runtime.__dict__.update(
    {
        'np': np,
        'csv': csv,
        'Path': Path,
        'Any': Any,
        'annotations': annotations,
        'expert_spans': expert_spans,
    }
)
exec(
    compile(
        Path(__file__).with_name('geometry_section.py').read_text(),
        'tools/fr3/fr3_act_infer_real_runtime.py<geometry>',
        'exec',
    ),
    fr3_act_infer_real_runtime.__dict__,
)
"""


def geometry_section(source: str) -> str:
    """The trace class and the constants above it, sliced out of the runtime's own text."""
    start = source.index(SECTION_START)
    class_at = source.index(SECTION_CLASS, start)
    tail = re.search(r"\n(?=(?:class |def )\w)", source[class_at:])
    end = class_at + (tail.start() if tail else len(source) - class_at)
    return source[start:end]


def geometry_tests(source: str) -> list[str]:
    """Every test that exercises the trace, plus the helper they share.

    Selected on what a function's body touches, not on what its name says: a name filter also
    catches the gripper tests that need the real module and would fail here for the wrong reason.
    """
    bodies = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node)
        touches_trace = "_trace_from(" in body or "RolloutGeometryTrace" in body
        if node.name == "_trace_from" or (node.name.startswith("test_") and touches_trace):
            bodies.append(body)
    return bodies


def main() -> int:
    section = geometry_section(RUNTIME.read_text())
    bodies = geometry_tests(TESTS.read_text())
    if not bodies:
        print(f"no trace tests found in {TESTS}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as work:
        out = Path(work)
        (out / "geometry_section.py").write_text(section)
        (out / "test_geometry_isolated.py").write_text(HEADER + "\n\n" + "\n\n\n".join(bodies) + "\n")
        env = dict(os.environ)
        # The ROS pytest plugins on this machine fail to load (`lark` missing, then an unknown
        # hook), and none of them are wanted here.
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        names = [b.split("(")[0].removeprefix("def ") for b in bodies if b.startswith("def test_")]
        print(
            f"extracted {len(section.splitlines())} lines of geometry and {len(names)} tests",
            flush=True,
        )
        # The extracted file imports `tools.fr3.dagger_takeover`, which lives in the repo the
        # scratch directory is not inside.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return subprocess.call(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(out)], env=env
        )


if __name__ == "__main__":
    raise SystemExit(main())
