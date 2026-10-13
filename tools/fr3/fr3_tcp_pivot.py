#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Measure where the pivot insert v2's socket centre sits in corenetic_gripper_ee, with the FR3's FK.

  collect  real robot: teaching mode (the arm is moved by hand), record joint angles while the insert's
           socket stays seated on the ball on the table and the gripper is turned about it
  fit      pinocchio FK of corenetic_gripper_ee -> socket centre in the EE frame (math in
           lerobot.robots.franka_research3.tcp_pivot); also fits Desk's own EE frame (O_T_EE) as a check

Why: the camera trajectory's TCP is that socket centre, the FR3 drives corenetic_gripper_ee. The 10-12
end-to-end replay (laser vs laser) saw a constant (−0.6, −2.9, −10.4) mm in the TCP frame between them;
if this pivot gives about that, the offset is the TCP definition and goes into the bundle / replay target.

Procedure: gripper on the FR3 clamps insert v2 as for E1p; seat the socket on the table ball; run
collect; tilt the gripper about 30° each way in two directions and spin it about its axis, keeping the
socket seated, for the whole recording. On Thor, from ~/lerobot (panda_py needs the cmeel libraries):
  V=~/Code/infer/.venv-fr3; LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/cmeel.prefix/lib \
  PYTHONPATH=src $V/bin/python tools/fr3/fr3_tcp_pivot.py collect
"""

# ruff: noqa: N806  (matrix names: R, T, A follow the maths)
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.franka_research3.tcp_pivot import fit_pivot

REPO = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
DEFAULT_ROBOT_IP = "192.168.11.102"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
# 10-12 end-to-end replay, constant offset in the TCP frame (replay minus recording), mm
E2E_PREDICTION_MM = (-0.6, -2.9, -10.4)


def collect(args) -> int:
    from panda_py import Panda

    panda = Panda(args.robot_ip)
    print(
        f"collect: teaching mode for {args.duration_s:.0f} s at {args.rate_hz:.0f} Hz. Seat the socket on the "
        "ball first; keep it seated while tilting ~30° both ways in two directions and spinning the gripper."
    )
    if (
        not args.yes
        and input("The arm goes limp under gravity compensation. Type yes to start: ").strip() != "yes"
    ):
        print("aborted")
        return 1
    samples = {"t": [], "q": [], "dq": [], "O_T_EE": []}
    panda.teaching_mode(True, [args.damping] * 7)
    try:
        t0 = time.monotonic()
        next_print = 0.0
        while (elapsed := time.monotonic() - t0) < args.duration_s:
            s = panda.get_state()  # live while a controller (teaching mode) runs
            samples["t"].append(elapsed)
            samples["q"].append(list(s.q))
            samples["dq"].append(list(s.dq))
            samples["O_T_EE"].append(list(s.O_T_EE))
            if elapsed >= next_print:
                print(f"  {elapsed:5.1f} s  {len(samples['t'])} samples")
                next_print += 5.0
            time.sleep(1.0 / args.rate_hz)
    finally:
        with contextlib.suppress(Exception):
            panda.teaching_mode(False)
    q = np.asarray(samples["q"])
    unique = len(np.unique(q, axis=0))
    if unique < 0.5 * len(q):
        print(f"WARNING: only {unique}/{len(q)} distinct joint readings -- the state may be a frozen cache")
    out = args.out or REPO / "outputs/fr3_tcp_pivot" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_pivot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "robot_ip": args.robot_ip,
                "note": args.note,
                "created": dt.datetime.now().isoformat(),
                **samples,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(q)} samples, {unique} distinct)")
    return 0


def _fk(urdf: Path, frame: str, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pinocchio as pin

    full = pin.buildModelFromUrdf(str(urdf))
    lock = [i for i, n in enumerate(full.names) if i > 0 and n not in JOINTS]
    model = pin.buildReducedModel(full, lock, pin.neutral(full))
    data = model.createData()
    fid = model.getFrameId(frame)
    R, t = [], []
    for q in qs:
        pin.forwardKinematics(model, data, np.asarray(q))
        pin.updateFramePlacements(model, data)
        R.append(data.oMf[fid].rotation.copy())
        t.append(data.oMf[fid].translation.copy())
    return np.asarray(R), np.asarray(t)


def fit(args) -> int:
    run = json.loads(args.run.read_text(encoding="utf-8"))
    q = np.asarray(run["q"])
    R, t = _fk(args.urdf, args.frame, q)
    result = fit_pivot(R, t, bin_deg=args.bin_deg, reject_mm=args.reject_mm)
    report = {"run": str(args.run), "frame": args.frame, "fk": result.as_dict()}
    # Desk's EE frame (O_T_EE, column-major) as an independent FK: same point if Desk's TCP is the vendor's
    T = np.asarray(run["O_T_EE"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    report["desk_O_T_EE"] = fit_pivot(
        T[:, :3, :3], T[:, :3, 3], bin_deg=args.bin_deg, reject_mm=args.reject_mm
    ).as_dict()
    s = np.asarray(report["fk"]["s_ee_mm"])
    report["e2e_prediction_mm"] = list(E2E_PREDICTION_MM)
    report["minus_prediction_mm"] = np.round(s - np.asarray(E2E_PREDICTION_MM), 2).tolist()
    fk = report["fk"]
    print(
        f"socket centre in {args.frame}: {fk['s_ee_mm']} mm (±{fk['s_sigma_mm']}), residual rms "
        f"{fk['residual_rms_mm']} p95 {fk['residual_p95_mm']} mm, {fk['n_used']}/{fk['n_total']} samples, "
        f"span {fk['span_deg']}°, weak singular {fk['weak_singular']}"
    )
    print(f"Desk O_T_EE frame: {report['desk_O_T_EE']['s_ee_mm']} mm")
    print(
        f"10-12 end-to-end offset {list(E2E_PREDICTION_MM)} mm -> difference {report['minus_prediction_mm']} mm"
    )
    if fk["span_deg"] < 20 or fk["weak_singular"] < 0.1:
        print("WARNING: not enough rotation (want >= 20° about two axes); s is poorly determined")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect", help="real robot, teaching mode (moved by hand)")
    p.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    p.add_argument("--duration-s", type=float, default=45.0)
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--damping", type=float, default=0.0, help="teaching-mode joint damping")
    p.add_argument("--note", default="")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=collect)
    p = sub.add_parser("fit", help="socket centre in the EE frame from a collected run")
    p.add_argument("run", type=Path)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--frame", default="corenetic_gripper_ee")
    p.add_argument("--bin-deg", type=float, default=1.0)
    p.add_argument("--reject-mm", type=float, default=0.5, help="drop samples off the sphere (insert lifted)")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=fit)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
