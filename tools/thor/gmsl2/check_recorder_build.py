#!/usr/bin/env python3
"""Compile the three Argus recorders on Thor without touching the live binaries.

The recorders are normally built on demand, straight over the path the next
recording session will execute (``/tmp/lerobot_argus_*_video_recorder``).  That
is fine when the code already compiles and exactly wrong when you are checking
whether it does: a broken build would replace a working binary, and you would
find out at the start of an episode rather than at your desk.

So this builds every target into a scratch directory instead, reusing each
session class's own ``_build_binary`` so the include paths, the link line and
the CUDA shim cannot drift from what the real build does.

    python -m tools.thor.gmsl2.check_recorder_build
    python -m tools.thor.gmsl2.check_recorder_build --out-dir /tmp/my_build_check

Exit codes: 0 all targets compiled, 1 at least one did not (the diagnostics are
printed), 2 it could not be checked here -- the Argus SDK is Thor-only, and an
x86 dev box has no ``/usr/src/jetson_multimedia_api``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARGUS_SDK = Path("/usr/src/jetson_multimedia_api")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_capture_tool(out: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    """The standalone metadata capture tool has no session class to borrow from.

    Its build line is the one recorded in ARGUS_METADATA_SYNC_IMPLEMENTATION.md;
    keep the two in step if either moves.
    """
    src = Path(__file__).with_name("argus_frame_metadata_capture.cpp")
    cmd = (
        "g++ -std=c++14 -O2 "
        f"-I{ARGUS_SDK}/argus/include "
        f"-I{ARGUS_SDK}/argus/samples/utils "
        f"{src} "
        f"{ARGUS_SDK}/argus/samples/utils/ArgusHelpers.cpp "
        "-L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient -lpthread "
        f"-o {out}"
    )
    return subprocess.run(
        cmd, shell=True, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_s
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=None,
                   help="scratch output directory; defaults to a fresh temp dir. "
                        "It must not be the live /tmp/lerobot_argus_* paths")
    p.add_argument("--timeout-s", type=float, default=600.0)
    args = p.parse_args(argv)

    if not ARGUS_SDK.is_dir():
        print(f"cannot check: {ARGUS_SDK} not found -- the Argus SDK is Thor-only, "
              "so this has to run on the rig host, not a dev box.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tmp = None
    if args.out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="lerobot_build_check_")
        out_dir = Path(tmp.name)
    else:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    import tools.thor.gmsl2.argus_metadata_session as ams
    import tools.thor.gmsl2.argus_online_sync_session as aoss

    targets: list[tuple[str, object]] = [
        ("argus_metadata_video_recorder", ams.ArgusMetadataCameraSession),
        ("argus_online_sync_video_recorder", aoss.ArgusOnlineSyncCameraSession),
    ]

    failures: list[str] = []
    try:
        for name, cls in targets:
            out = out_dir / name
            if "lerobot_argus" in str(out) and out.parent == Path("/tmp"):
                print(f"refusing to build over the live binary path {out}", file=sys.stderr)
                return 2
            session = cls([], out_dir, repo_root=REPO_ROOT, binary_path=out,
                          auto_build=False, connect_timeout_s=args.timeout_s)
            print(f"building {name} -> {out}")
            try:
                session._build_binary()
            except RuntimeError as exc:
                failures.append(name)
                print(f"FAIL {name}:\n{exc}", file=sys.stderr)
                continue
            print(f"  ok ({out.stat().st_size} bytes)")

        name = "argus_frame_metadata_capture"
        out = out_dir / name
        print(f"building {name} -> {out}")
        result = _build_capture_tool(out, args.timeout_s)
        if result.returncode != 0:
            failures.append(name)
            print(f"FAIL {name} (rc={result.returncode}):\n"
                  f"{(result.stderr or result.stdout).strip()}", file=sys.stderr)
        else:
            print(f"  ok ({out.stat().st_size} bytes)")
    finally:
        if tmp is not None:
            tmp.cleanup()

    if failures:
        print(f"\n{len(failures)} target(s) failed to compile: "
              f"{', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nOK: all three recorders compile. The live binaries were not touched; "
          "the next session will rebuild them itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
