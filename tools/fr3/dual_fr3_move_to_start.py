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

"""Both FR3s to their start joints, checked against each other before either moves.

Single-arm move_to_start (``fr3_move_to_start_runtime.py``) cannot see the other arm. With two arms 0.66 m
apart, an arm that ends a replay reaching toward the middle can sweep through the other on its way home. So
this reads both arms first and only then decides how they go:

* panda_py's ``move_to_joint_position`` to one target is a straight line in joint space (its trajectories are
  piece-wise linear, time-optimal). Each arm's path is that line, sampled every ``STEP_RAD`` of its largest
  joint (the TCP moves at most ~16 mm between samples).
* **together**: every pair (a point on the right's path, a point on the left's) keeps the arms ``d_gate``
  apart -- so it holds whatever the two timings are, an arm that stops early included, and the arms need no
  coordination while they move.
* **one after the other**: the first moves while the other stands where it is, then the second moves to a
  first already home. Taken when "together" is not clear and one order is.
* neither: refused, with the closest pair and how close -- guide the arms apart in Desk.

An arm that is already inside a gap (a replay that ended 55 mm apart, a hand near the table) is held to "never
closer than now" instead (``NEVER_CLOSER_MARGIN_M``): going home is how it gets out.

Each path also keeps ``d_env`` off the table and the cart top (``dual_fr3_scene``'s assumed quarter- and
half-space). ``d_gate`` is the replay runtime's gate, three placement sigmas (``fr3_arms.json``'s
``placement``, a tape measure until B3); refused when the registry declares no placement.

One process per arm holds its FCI session from the check to the end (``spawn``, as the replay runtime's
arms), so nothing else can move an arm between the check and the move; the move still re-reads both arms
and refuses if either moved since. The operator sees the plan (``DUAL_HOME_PLAN``) and answers ``go`` or
``abort`` on stdin (the gateway's Execute / Cancel); ``--check`` stops after the plan.
"""

# ruff: noqa: N803, N806  (T_ for poses and Q for joint paths, as in the replay runtime)
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import select
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import (  # noqa: E402
    dual_fr3_collision as col,
    dual_fr3_replay_runtime as rt,
    dual_fr3_retarget as rg,
    dual_fr3_scene as scene,
    fr3_arms,
    fr3_kinematics as kin,
)

ARMS = ("right", "left")
PLAN_LINE = "DUAL_HOME_PLAN "
READY_LINE = "DUAL_HOME_READY (type go or abort)"
RESULT_LINE = "DUAL_HOME_RESULT "
#: Largest joint step between path samples.
STEP_RAD = 0.02
#: An arm this close to its start on every joint stays put (panda_py's own success_threshold).
AT_START_RAD = 0.01
#: Between the check and the move an arm may not have moved more than this (it was held by its FCI session,
#: so only a hand in Desk's guiding mode could have).
MOVED_SINCE_CHECK_RAD = 0.005
#: An arm already closer than a gate may still go home if its path never comes closer than it is now, less
#: this (what the samples can miss): the placement error is in both readings alike, so the comparison holds
#: where the absolute gap does not -- and the arms are not touching now.
NEVER_CLOSER_MARGIN_M = 0.005
#: Slower than panda_py's default 0.2: two arms in a shared space, someone at each stop.
SPEED_FACTOR = rt.REAL_START_SPEED_FACTOR
WAIT_FOR_GO_S = 120.0
CONNECT_TIMEOUT_S = 30.0
MOVE_TIMEOUT_S = 120.0


# ------------------------------------------------------------------ the plan ---


def joint_path(q0: Sequence[float], q1: Sequence[float], step_rad: float = STEP_RAD) -> np.ndarray:
    """The straight joint-space line from ``q0`` to ``q1``, both ends included, no joint stepping more than
    ``step_rad`` between samples."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    n = max(2, int(math.ceil(float(np.abs(q1 - q0).max()) / step_rad)) + 1)
    return q0 + np.linspace(0.0, 1.0, n)[:, None] * (q1 - q0)


def _arm_arm(model: col.CapsuleModel, Q_r: np.ndarray, Q_l: np.ndarray, T_right_left: np.ndarray):
    c = col.clearance(model, Q_r, Q_l, T_right_left)
    k = int(np.argmin(c["arm_arm_m"]))
    return float(c["arm_arm_m"][k]), list(c["arm_arm_pair"][k]), k


def _option(gap_m: float, pair: list[str], where: str) -> dict[str, Any]:
    return {"min_gap_mm": round(gap_m * 1e3, 1), "pair": pair, "where": where}


def joint_problems(arm: str, q_now: np.ndarray, q_goal: np.ndarray) -> list[str]:
    """What keeps an arm from moving at all: joints past panda_py's walls or the FR3's limits now (a
    controller started there aborts; the transport pose needs Desk), or a start beyond the walls."""
    problems = []
    beyond = np.flatnonzero((q_now < kin._WALL_LOWER) | (q_now > kin._WALL_UPPER))
    if len(beyond):
        problems.append(f"{arm}: {', '.join(f'j{j + 1}' for j in beyond)} past panda_py's walls now")
    past = [j for j in np.flatnonzero((q_now < kin._FR3_LOWER) | (q_now > kin._FR3_UPPER)) if j not in beyond]
    if past:
        problems.append(f"{arm}: {', '.join(f'j{j + 1}' for j in past)} past the FR3 limits now; guide it in Desk")
    outside = np.flatnonzero((q_goal < kin.Q_LOWER) | (q_goal > kin.Q_UPPER))
    if len(outside):
        problems.append(f"{arm}: its start joints put {', '.join(f'j{j + 1}' for j in outside)} past the walls")
    return problems


def plan_home(
    q_now: Mapping[str, Sequence[float]],
    q_goal: Mapping[str, Sequence[float]],
    T_right_left: np.ndarray,
    *,
    d_gate_m: float,
    d_env_m: float = rg.ENV_MARGIN_MM / 1e3,
    obstacles: Sequence[Mapping[str, Any]] | None = None,
    model: col.CapsuleModel | None = None,
) -> dict[str, Any]:
    """How the arms go home: ``stages`` (lists of arms that move together, in order), empty when both are
    there, None when refused (``reasons``); with every option's closest approach."""
    model = model or rt.capsule_model()
    obstacles = scene.environment_obstacles() if obstacles is None else list(obstacles)
    T = np.asarray(T_right_left, dtype=np.float64)
    now = {a: np.asarray(q_now[a], dtype=np.float64) for a in ARMS}
    goal = {a: np.asarray(q_goal[a], dtype=np.float64) for a in ARMS}
    arms: dict[str, Any] = {}
    reasons: list[str] = []
    paths = {}
    for a in ARMS:
        delta = np.abs(goal[a] - now[a])
        moving = bool(delta.max() > AT_START_RAD)
        paths[a] = joint_path(now[a], goal[a]) if moving else now[a][None, :]
        arms[a] = {
            "moving": moving,
            "q_now": now[a].tolist(),
            "q_goal": goal[a].tolist(),
            "max_joint_move_deg": round(math.degrees(float(delta.max())), 1),
            "joint": int(delta.argmax()) + 1,
        }
        reasons += joint_problems(a, now[a], goal[a])
    for a in ARMS:
        # the environment along each arm's own path: the other arm does not change it
        other = "left" if a == "right" else "right"
        Q_other = np.repeat(now[other][None, :], len(paths[a]), axis=0)
        Q_r, Q_l = (paths[a], Q_other) if a == "right" else (Q_other, paths[a])
        c = col.clearance(model, Q_r, Q_l, T, obstacles)
        env = c[f"env_{a}_m"]
        k = int(np.argmin(env))
        need = min(d_env_m, float(env[0]) - NEVER_CLOSER_MARGIN_M)
        arms[a]["min_env_gap_mm"] = None if not np.isfinite(env[k]) else round(float(env[k]) * 1e3, 1)
        arms[a]["min_env_with"] = c[f"env_{a}_with"][k]
        if np.isfinite(env[k]) and env[k] < need:
            part, obstacle = c[f"env_{a}_with"][k]
            reasons.append(
                f"{a}: its {part} comes {env[k] * 1e3:.0f} mm from the {obstacle} on the way "
                f"(needs {need * 1e3:.0f} mm)"
            )
    moving = [a for a in ARMS if arms[a]["moving"]]
    start_gap, _, _ = _arm_arm(model, now["right"][None, :], now["left"][None, :], T)
    gate_m = min(d_gate_m, start_gap - NEVER_CLOSER_MARGIN_M)
    out: dict[str, Any] = {
        "arms": arms,
        "start_gap_mm": round(start_gap * 1e3, 1),
        "d_gate_mm": round(d_gate_m * 1e3, 1),
        # what the paths are held to: d_gate, or "never closer than now" when the arms already are
        "gate_mm": round(gate_m * 1e3, 1),
        "d_env_mm": round(d_env_m * 1e3, 1),
        "options": {},
        "stages": None,
        "chosen": None,
        "reasons": reasons,
    }
    if not moving:
        out.update(stages=[], chosen="at_start")
        return out
    # together: every pair of points on the two paths
    n_r, n_l = len(paths["right"]), len(paths["left"])
    Q_r = np.repeat(paths["right"], n_l, axis=0)
    Q_l = np.tile(paths["left"], (n_r, 1))
    gap, pair, k = _arm_arm(model, Q_r, Q_l, T)
    s_r, s_l = (k // n_l) / max(n_r - 1, 1), (k % n_l) / max(n_l - 1, 1)
    out["options"]["together"] = _option(gap, pair, f"right {s_r:.0%} / left {s_l:.0%} of the way")
    # one after the other (only when both move)
    if len(moving) == 2:
        for first, second in (("right", "left"), ("left", "right")):
            legs = []
            for mover, standing in ((first, now[second]), (second, goal[first])):
                fixed = np.repeat(standing[None, :], len(paths[mover]), axis=0)
                pr, pl = (paths[mover], fixed) if mover == "right" else (fixed, paths[mover])
                g, p, kk = _arm_arm(model, pr, pl, T)
                legs.append((g, p, f"{mover} {kk / max(len(paths[mover]) - 1, 1):.0%} of the way"))
            g, p, where = min(legs, key=lambda leg: leg[0])
            out["options"][f"{first}_then_{second}"] = _option(g, p, where)
    if reasons:
        return out
    clear = {name: o for name, o in out["options"].items() if o["min_gap_mm"] >= gate_m * 1e3}
    if "together" in clear:
        out.update(stages=[moving], chosen="together")
    elif clear:
        name = max(clear, key=lambda n: clear[n]["min_gap_mm"])
        first, _, second = name.partition("_then_")
        out.update(stages=[[first], [second]], chosen=name)
    else:
        best = max(out["options"].values(), key=lambda o: o["min_gap_mm"])
        out["reasons"].append(
            f"every order brings the arms within {best['min_gap_mm']:.0f} mm "
            f"({' / '.join(best['pair'])}, {best['where']}); the gate is {gate_m * 1e3:.0f} mm -- "
            "guide them apart in Desk first"
        )
    return out


# ------------------------------------------------------------------ the arms ---


def _arm_worker(robot_ip: str, panda_py_module: str, conn: Any) -> None:
    """One arm's process: hold its FCI session, read on "read", move on ("move", q, speed), quit on "quit"."""
    import importlib

    def state(panda) -> dict[str, Any]:
        s = panda.get_robot().read_once()
        return {"q": [float(v) for v in s.q], "m_ee": float(s.m_ee), "F_x_Cee": [float(v) for v in s.F_x_Cee]}

    try:
        panda = importlib.import_module(panda_py_module).Panda(robot_ip)
        conn.send(("state", state(panda)))
    except Exception as exc:  # noqa: BLE001 -- the arm's error goes to the coordinator, whatever it is
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        try:
            if message[0] == "read":
                conn.send(("state", state(panda)))
            elif message[0] == "move":
                ok = bool(panda.move_to_joint_position(np.asarray(message[1], dtype=np.float64), speed_factor=message[2]))
                conn.send(("moved", {"ok": ok, **state(panda)}))
            else:
                return
        except Exception as exc:  # noqa: BLE001
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class Arms:
    """The two arm processes and their pipes."""

    def __init__(self, entries: Mapping[str, Mapping[str, Any]], panda_py_module: str = "panda_py"):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = {}, {}
        for a in ARMS:
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_arm_worker, args=(str(entries[a]["robot_ip"]), panda_py_module, child), daemon=True
            )
            proc.start()
            self.conns[a], self.procs[a] = parent, proc

    def receive(self, arms: Sequence[str], timeout_s: float) -> dict[str, tuple[str, Any]]:
        out, deadline = {}, time.monotonic() + timeout_s
        for a in arms:
            if self.conns[a].poll(max(0.0, deadline - time.monotonic())):
                out[a] = self.conns[a].recv()
            else:
                out[a] = ("error", f"no answer in {timeout_s:g} s")
        return out

    def send(self, arms: Sequence[str], message: tuple) -> None:
        for a in arms:
            self.conns[a].send(message)

    def close(self) -> None:
        for a in ARMS:
            with contextlib.suppress(OSError, ValueError):
                self.conns[a].send(("quit",))
        for a in ARMS:
            self.procs[a].join(timeout=5.0)
            if self.procs[a].is_alive():
                self.procs[a].terminate()


def _states(answers: Mapping[str, tuple[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    states, errors = {}, []
    for a, (kind, value) in answers.items():
        if kind == "error":
            errors.append(f"{a}: {value}")
        else:
            states[a] = value
    return states, errors


def _emit(prefix: str, payload: Mapping[str, Any]) -> None:
    print(prefix + json.dumps(payload, default=float), flush=True)


def _wait_for_go(timeout_s: float) -> str:
    print(READY_LINE, flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return "timeout"
    return sys.stdin.readline().strip().lower() or "eof"


def execute(arms: Arms, plan: Mapping[str, Any], goals: Mapping[str, Sequence[float]], speed: float) -> dict[str, Any]:
    """Run the stages in order; a stage that fails (an arm's error, or not at its start) ends the run."""
    result: dict[str, Any] = {"stages": [], "ok": True}
    for stage in plan["stages"]:
        for a in stage:
            arms.conns[a].send(("move", list(goals[a]), speed))
        answers = arms.receive(stage, MOVE_TIMEOUT_S)
        row = {}
        for a in stage:
            kind, value = answers[a]
            if kind == "error":
                row[a] = {"ok": False, "error": value}
                continue
            err = float(np.abs(np.asarray(value["q"]) - np.asarray(goals[a])).max())
            row[a] = {"ok": value["ok"] and err <= AT_START_RAD, "q": value["q"], "max_error_deg": math.degrees(err)}
        result["stages"].append(row)
        if not all(r["ok"] for r in row.values()):
            result["ok"] = False
            break
    return result


def home(
    registry: Mapping[str, Any],
    *,
    check: bool = False,
    go: Any = None,
    speed_factor: float = SPEED_FACTOR,
    panda_py_module: str = "panda_py",
) -> int:
    """Read both arms, plan, and (after ``go()`` says "go") move them; the exit code ``main`` returns.
    ``go`` defaults to asking on stdin."""
    try:
        entries = {a: fr3_arms.resolve(registry, a) for a in ARMS}
    except fr3_arms.ArmConfigError as exc:
        print(f"refused: {exc}")
        return 2
    placement = fr3_arms.placement(registry)
    if placement is None:
        print(f"refused: {fr3_arms.REGISTRY} declares no placement; the arms cannot be checked against each other")
        return 2
    gate = rt.derive_gate(placement["sigma_mm"])
    goals = {a: [float(v) for v in entries[a]["start_joints_rad"]] for a in ARMS}
    arms = Arms(entries, panda_py_module)
    try:
        states, errors = _states(arms.receive(ARMS, CONNECT_TIMEOUT_S))
        if errors:
            print("refused: " + "; ".join(errors))
            return 2
        payload_lines = {}
        for a in ARMS:
            ok, line = fr3_arms.payload_check(entries[a], states[a]["m_ee"], states[a]["F_x_Cee"])
            payload_lines[a] = line
            if not ok:
                errors.append(f"{a}: {line}")
        plan = plan_home(
            {a: states[a]["q"] for a in ARMS}, goals, np.asarray(placement["T_right_left"]), d_gate_m=gate["d_gate_m"]
        )
        plan["reasons"] = errors + plan["reasons"]
        plan["placement"] = {"sigma_mm": placement["sigma_mm"], "source": placement["source"],
                             "T_right_left": placement["T_right_left"]}  # fmt: skip
        plan["payload"] = payload_lines
        plan["speed_factor"] = speed_factor
        plan["ip"] = {a: str(entries[a]["robot_ip"]) for a in ARMS}
        _emit(PLAN_LINE, plan)
        if plan["reasons"] or plan["stages"] is None:
            print("refused: " + "; ".join(plan["reasons"]))
            return 2
        if not plan["stages"]:
            print("both arms are at their start joints; nothing to do")
            return 0
        if check:
            return 0
        answer = (go or (lambda: _wait_for_go(WAIT_FOR_GO_S)))()
        if answer != "go":
            print(f"not moved ({answer})")
            return 3
        arms.send(ARMS, ("read",))
        again, errors = _states(arms.receive(ARMS, CONNECT_TIMEOUT_S))
        moved = [
            a for a in again
            if float(np.abs(np.asarray(again[a]["q"]) - np.asarray(states[a]["q"])).max()) > MOVED_SINCE_CHECK_RAD
        ]  # fmt: skip
        if errors or moved:
            print("refused: " + "; ".join(errors + [f"{a} moved since the check; check again" for a in moved]))
            return 2
        result = execute(arms, plan, goals, speed_factor)
        result["chosen"] = plan["chosen"]
        _emit(RESULT_LINE, result)
        return 0 if result["ok"] else 1
    finally:
        arms.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Both FR3s to their start joints, the paths checked first")
    parser.add_argument("--check", action="store_true", help="print the plan and stop; nothing moves")
    parser.add_argument("--yes", action="store_true", help="go without asking (after a clear plan)")
    parser.add_argument("--speed-factor", type=float, default=SPEED_FACTOR)
    args = parser.parse_args(argv)
    if not 0.0 < args.speed_factor <= 0.2:
        print(f"refused: --speed-factor {args.speed_factor:g} is outside (0, 0.2]")
        return 2
    try:
        registry = fr3_arms.load()
    except fr3_arms.ArmConfigError as exc:
        print(f"refused: {exc}")
        return 2
    return home(registry, check=args.check, go=(lambda: "go") if args.yes else None, speed_factor=args.speed_factor)


if __name__ == "__main__":
    raise SystemExit(main())
