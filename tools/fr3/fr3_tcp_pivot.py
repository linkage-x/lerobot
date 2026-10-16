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
  fit-joint  several runs, one socket centre: each run turned about the ball from its own heading pins
           what one ±20° sweep leaves loose; one ball centre for all (the ball not moved) or one per run

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
from typing import Any

import numpy as np

from lerobot.robots.franka_research3.tcp_pivot import fit_pivot, fit_pivot_joint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf"
DEFAULT_ROBOT_IP = "192.168.11.102"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
# 10-12 end-to-end replay, constant offset in the TCP frame (replay minus recording), mm
E2E_PREDICTION_MM = (-0.6, -2.9, -10.4)
# The URDF frame the replay drives, set from that end-to-end offset (z only)
SOCKET_FRAME = "corenetic_socket_ee"
JOINT_SCHEMA = "fr3_pivot_joint/v1"
# fit-joint: a run whose own ball centre (s held at the joint value) is this far from the others' had
# the ball moved (10-14: 3 mm between two sessions) -- one shared centre would bend s.
JOINT_CENTRE_MOVED_MM = 1.0


# FR3 joint limits (rad), as in fr3_corenetic_gripper.urdf. panda_py's teaching mode runs a
# virtual-wall controller built on the *Panda's* limits (panda_py.constants), which differ: narrower on
# j2 (±101.0° vs ±102.2°) and j6 at the top (215.0° vs 258.8°). A start pose past a wall makes panda_py
# throw from its 1 kHz thread and the process aborts with no traceback (2026-10-13: q2 = -103.2°,
# past both; Desk stayed green, the robot itself saw no error).
FR3_JOINT_LIMITS = (
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
)
WALL_MARGIN_DEG = 3.0


def teaching_wall_problems(q, lower, upper, margin_deg: float = WALL_MARGIN_DEG) -> tuple[list[str], list[str]]:
    """(beyond, close): joints past panda_py's virtual walls, and joints within ``margin_deg`` of one."""
    beyond, close = [], []
    margin = np.radians(margin_deg)
    for j, (x, lo, hi) in enumerate(zip(q, lower, upper, strict=True), start=1):
        if x < lo or x > hi:
            side, wall = ("below", lo) if x < lo else ("above", hi)
            beyond.append(
                f"j{j} = {np.degrees(x):.1f}° is {side} panda_py's teaching-mode wall {np.degrees(wall):.1f}° "
                f"(FR3 [{np.degrees(FR3_JOINT_LIMITS[j - 1][0]):.1f}°, {np.degrees(FR3_JOINT_LIMITS[j - 1][1]):.1f}°]); "
                "move it at least "
                f"{np.degrees(abs(x - wall)) + margin_deg:.1f}° back inside"
            )
        elif min(x - lo, hi - x) < margin:
            close.append(f"j{j} = {np.degrees(x):.1f}° is within {margin_deg:.0f}° of the teaching-mode wall")
    return beyond, close


def collect(args) -> int:
    from panda_py import Panda
    from panda_py.constants import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER

    panda = Panda(args.robot_ip)
    # Checked before teaching mode starts: past a wall panda_py aborts the process (no traceback,
    # no teaching_mode(False)), so this is the only place a readable refusal can come from.
    beyond, close = teaching_wall_problems(panda.get_state().q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    if beyond:
        print("ERROR: the arm starts outside panda_py's teaching-mode joint walls; nothing was started.")
        for line in beyond:
            print(f"  {line}")
        print("  Move the arm (Desk guiding / pilot buttons) into range, re-seat the socket, then start again.")
        return 2
    for line in close:
        print(f"WARNING: {line}: the wall pushes back there, keep the pivot away from it")
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


def contact_summary(run: dict[str, Any]) -> dict[str, Any] | None:
    """How hard the socket pressed on the ball while pivoting.

    Franka's F_ext_hat_K is what the robot applies to the world (10-12's payload runs read +Fz for a
    load heavier than Desk's: the arm pushing it up). The press is straight down onto the ball's stand,
    so with ``F_ext_frame`` "O" (fr3_base) it is -Fz. Older runs logged K_F_ext_hat_K (tool frame).
    """
    if "F_ext" not in run or not run["F_ext"]:
        return None
    F = np.asarray(run["F_ext"], dtype=float)
    phase = np.asarray(run.get("phase") or [1] * len(F))
    pivot = (phase > 0) & (phase < phase.max()) if phase.max() > 1 else np.ones(len(F), bool)
    down = -F[:, 2] if run.get("F_ext_frame") == "O" else F[:, 2]
    press = down[pivot] if pivot.any() else down
    return {
        "press_median_n": round(float(np.median(press)), 2),
        "press_p05_n": round(float(np.percentile(press, 5)), 2),
        "in_contact_fraction": round(float(np.mean(press > 0.5)), 3),
        "max_force_n": round(float(np.linalg.norm(F[:, :3], axis=1).max()), 2),
    }


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
    report = {
        "run": str(args.run),
        "note": run.get("note", ""),
        "created": run.get("created", ""),
        "robot_ip": run.get("robot_ip", ""),
        "frame": args.frame,
        "fk": result.as_dict(),
    }
    # Desk's EE frame (O_T_EE, column-major) as an independent FK: same point if Desk's TCP is the vendor's
    T = np.asarray(run["O_T_EE"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    report["desk_O_T_EE"] = fit_pivot(
        T[:, :3, :3], T[:, :3, 3], bin_deg=args.bin_deg, reject_mm=args.reject_mm
    ).as_dict()
    # The frame the replay drives: if it is where the socket is, s is ~0 in it.
    try:
        R_s, t_s = _fk(args.urdf, SOCKET_FRAME, q)
        report["socket_frame"] = {
            "frame": SOCKET_FRAME,
            **fit_pivot(R_s, t_s, bin_deg=args.bin_deg, reject_mm=args.reject_mm).as_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - an older URDF without the frame
        report["socket_frame"] = {"frame": SOCKET_FRAME, "error": f"{type(exc).__name__}: {exc}"}
    if run.get("mode") in ("commanded", "guided"):  # fr3_pivot_auto.py execute / guide: pressed into the ball
        report["commanded"] = {
            "kind": run["mode"],
            "plan": run.get("plan", ""),
            "centre": run.get("centre", ""),
            "bias_mm": run.get("bias_mm"),
            "press_mm": run.get("press_mm"),
            "anchor_mm": run.get("anchor_mm"),  # guide: the socket-frame point the spring held
            "anchor_from": run.get("anchor_from", ""),
            "completed": bool(run.get("completed")),
            "contact": contact_summary(run),
        }
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
    sock = report["socket_frame"]
    if "error" not in sock:
        print(f"socket centre in {SOCKET_FRAME} (the replay target): {sock['s_ee_mm']} mm -- ~0 if that frame is right")
    print(f"pivot point in fr3_base: {fk['c_base_mm']} mm (±{fk['c_sigma_mm']})")
    print(
        f"10-12 end-to-end offset {list(E2E_PREDICTION_MM)} mm -> difference {report['minus_prediction_mm']} mm"
    )
    contact = (report.get("commanded") or {}).get("contact")
    if contact:
        print(
            f"commanded run, bias {report['commanded']['bias_mm']} mm: pressed {contact['press_median_n']} N "
            f"(5th pct {contact['press_p05_n']} N), in contact {contact['in_contact_fraction'] * 100:.0f}% of the pivot"
        )
        if contact["in_contact_fraction"] < 0.95:
            print("WARNING: the socket was not pressed on the ball the whole pivot; the fit may follow the command")
    if fk["span_deg"] < 20 or fk["weak_singular"] < 0.1:
        print("WARNING: not enough rotation (want >= 20° about two axes); s is poorly determined")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def fit_joint(args) -> int:
    Rs, ts, Rs_s, ts_s, names = [], [], [], [], []
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        q = np.asarray(run["q"])
        R, t = _fk(args.urdf, args.frame, q)
        Rs.append(R)
        ts.append(t)
        R, t = _fk(args.urdf, SOCKET_FRAME, q)
        Rs_s.append(R)
        ts_s.append(t)
        names.append(path.name)
    kw = {"shared_centre": not args.per_run_centre, "bin_deg": args.bin_deg, "reject_mm": args.reject_mm}
    fk = fit_pivot_joint(Rs, ts, **kw).as_dict()
    sock = fit_pivot_joint(Rs_s, ts_s, **kw).as_dict()
    warnings = []
    moved = [
        f"{name}: {np.linalg.norm(off):.1f} mm"
        for name, off in zip(names, sock["run_centre_offset_mm"], strict=True)
        if np.linalg.norm(off) > JOINT_CENTRE_MOVED_MM
    ]
    if moved and not args.per_run_centre:  # with a centre per run a moved ball costs nothing
        warnings.append(
            f"each run's own ball centre is off the others' by more than {JOINT_CENTRE_MOVED_MM:g} mm ({'; '.join(moved)}): "
            "the ball moved between them -- fit them with --per-run-centre, or apart"
        )
    if sock["span_deg"] < 20 or sock["weak_singular"] < 0.1:
        warnings.append("not enough rotation over all runs (want >= 20° about two axes); s is poorly determined")
    report = {
        "schema": JOINT_SCHEMA,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "runs": names,
        "shared_centre": not args.per_run_centre,
        "frame": args.frame,
        "fk": fk,
        "socket_frame": {"frame": SOCKET_FRAME, **sock},
        "e2e_prediction_mm": list(E2E_PREDICTION_MM),
        "minus_prediction_mm": np.round(np.asarray(fk["s_ee_mm"]) - np.asarray(E2E_PREDICTION_MM), 2).tolist(),
        "warnings": warnings,
    }
    print(
        f"{len(names)} runs, {'one ball centre' if report['shared_centre'] else 'a ball centre per run'}: "
        f"socket centre in {SOCKET_FRAME} {sock['s_ee_mm']} mm (±{sock['s_sigma_mm']}), "
        f"residual rms {sock['residual_rms_mm']} mm, span {sock['span_deg']}°, weak singular {sock['weak_singular']}"
    )
    print(f"socket centre in {args.frame}: {fk['s_ee_mm']} mm")
    for name, rms, off in zip(names, sock["run_rms_mm"], sock["run_centre_offset_mm"], strict=True):
        print(f"  {name}: rms {rms} mm, own ball centre off the mean {off} mm")
    if sock["s_leave_one_run_out_mm"]:
        spread = np.ptp(np.asarray(sock["s_leave_one_run_out_mm"]), axis=0)
        print(f"leave one run out: s spread {np.round(spread, 2).tolist()} mm")
    for line in warnings:
        print(f"WARNING: {line}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
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
    p = sub.add_parser("fit-joint", help="one socket centre from several runs (each from its own heading)")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--per-run-centre", action="store_true", help="the ball may have moved between runs")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--frame", default="corenetic_gripper_ee")
    p.add_argument("--bin-deg", type=float, default=1.0)
    p.add_argument("--reject-mm", type=float, default=0.5, help="drop samples off the sphere (insert lifted)")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=fit_joint)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
