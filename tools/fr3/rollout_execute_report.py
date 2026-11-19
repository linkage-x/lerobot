# ruff: noqa: N806  -- T, R: the usual names of transforms
"""What an execute run did, read back from its record (rollout_execute.Recorder). Read-only; no arm, no GPU.

    python -m tools.fr3.rollout_execute_report outputs/rollout/records/execute_20261118_153000 [...] [--json out.json]

For each run:

- **result**: stage, completed or why it stopped, chunks taken and refused (with the reason).
- **inputs**: how old each frame was when its chunk was predicted (``age_ms``: the predictor's clock minus
  the frame's SOF), the prediction's latency, the BOX reading's staleness.
- **timing**: how many steps each later chunk came late (``k0 - 1``, executed from step k0), how many steps
  each chunk ran before the next took over, the interval between chunks, the loop period.
- **motion**: the TCP (FK of the measured joints) against FK of the command at every tick (what the
  ``tracking_mm`` stop watches), and against the policy's target at each executed step's due time
  ``t0 + offsets[k]`` (``k*dt`` unless retimed) -- what the policy asked against what the arm did; the external force against the first
  tick, the fastest joint, the closest panda_py wall.
- **headroom**: the worst value of each against its stop (``Limits``) -- what to look at before tightening or
  loosening one. A limit is changed from several runs' worst values, not from one -- and not from stage 1,
  which makes a single prediction (one frame age, one latency); stage 2 makes about a dozen in its 5 s.

Several records: each, then the inputs and timing pooled over all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


FEW_PREDICTIONS = 5  # below this the input statistics are single values, not distributions


def load(record_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    """(record.json, chunks.jsonl rows, ticks.npz arrays or {}, events.json rows or [])."""
    record_dir = Path(record_dir)
    meta = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    if meta.get("kind") != "execute":
        raise ValueError(f"{record_dir}: not an execute record (kind={meta.get('kind')!r})")
    rows: list[dict[str, Any]] = []
    path = record_dir / "chunks.jsonl"
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ticks: dict[str, np.ndarray] = {}
    if (record_dir / "ticks.npz").is_file():
        with np.load(record_dir / "ticks.npz") as data:
            ticks = {k: np.asarray(data[k]) for k in data.files}
    events: list[dict[str, Any]] = []
    if (record_dir / "events.json").is_file():
        events = json.loads((record_dir / "events.json").read_text(encoding="utf-8"))
    return meta, rows, ticks, events


def tcp_fk(meta: dict[str, Any], repo_root: Path = _REPO_ROOT) -> Callable[[np.ndarray], np.ndarray]:
    """FK of joints (N, 7) to the policy's TCP (N, 4, 4), with the tool frame the run used (``tcp_tool`` in
    the record), or the arm's current one for records written before it was stored."""
    from tools.fr3 import fr3_kinematics as kin

    if meta.get("tcp_tool") is not None:
        T_tool = np.asarray(meta["tcp_tool"], dtype=np.float64)
    else:
        from tools.fr3 import rollout_relative_tcp as rrt

        T_tool = rrt.arm_tcp_tool(str(meta["arm"]), repo_root=repo_root)

    def fk(q: np.ndarray) -> np.ndarray:
        return kin.NOMINAL.fk(np.atleast_2d(np.asarray(q, dtype=np.float64))) @ T_tool

    return fk


def stats(values: Any) -> dict[str, Any]:
    v = np.asarray([x for x in np.ravel(np.asarray(values, dtype=np.float64)) if math.isfinite(x)])
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3),
            "max": round(float(v.max()), 3), "min": round(float(v.min()), 3)}


def _executed_steps(rows: list[dict[str, Any]], t_end: float, dt: float) -> list[dict[str, Any]]:
    """Per accepted chunk: the steps the loop ran, with their due times. A chunk runs from k0 until the next
    accepted chunk takes over (its acceptance time), the run ends, or its ``end``."""
    taken = [r for r in rows if r.get("accepted") and r.get("t0") is not None]
    out = []
    for i, r in enumerate(taken):
        until = taken[i + 1]["t"] if i + 1 < len(taken) else t_end
        k0, end, t0 = int(r["k0"]), int(r["end"]), float(r["t0"])
        n = len(r["targets_base"])
        offsets = r.get("offsets") or [k * dt for k in range(n)]  # retimed steps (11-19); before: k*dt
        ks = [k for k in range(k0, end + 1) if t0 + offsets[k] <= until + 1e-9]
        out.append({"index": r.get("index"), "k0": k0, "end": end, "t": r["t"], "t0": t0,
                    "steps": ks, "due": [t0 + offsets[k] for k in ks],
                    "stretch": [(offsets[k] - offsets[k - 1]) / dt for k in ks],
                    "targets": np.asarray(r["targets_base"], dtype=np.float64)})
    return out


def analyze(record_dir: Path, *, fk: Callable[[np.ndarray], np.ndarray] | None = None,
            repo_root: Path = _REPO_ROOT) -> dict[str, Any]:
    meta, rows, ticks, events = load(record_dir)
    limits = meta.get("limits") or {}
    contract = meta.get("contract") or {}
    dt = float(contract.get("frame_stride", 1)) / float(contract.get("fps", 30))
    n_exec = int(meta.get("n_action_steps") or 0)
    result = meta.get("result") or {}
    report: dict[str, Any] = {
        "record": str(Path(record_dir)),
        "checkpoint": meta.get("checkpoint"),
        "arm": meta.get("arm"),
        "frame_source": meta.get("frame_source"),
        "stage": (meta.get("stage") or {}).get("number"),
        "stage_label": (meta.get("stage") or {}).get("label"),
        "created": meta.get("created"),
        "finished": meta.get("finished"),
        "result": result or {"stopped": "no result in record.json (the run did not finish writing it)"},
        "n_action_steps": n_exec,
        "step_s": dt,
    }
    accepted = [r for r in rows if r.get("accepted")]
    refused = [r for r in rows if not r.get("accepted")]
    report["chunks"] = {"accepted": len(accepted), "refused": len(refused),
                        "refusals": [r.get("why") or r.get("error") or "" for r in refused]}

    stale = [((r.get("obs_meta") or {}).get("box_stale_s") or math.nan) * 1e3 for r in rows if r.get("obs_meta")]
    report["inputs"] = {
        "age_ms": stats([r["age_ms"] for r in rows if r.get("age_ms")]),
        "latency_ms": stats([r["latency_ms"] for r in rows if r.get("latency_ms")]),
        "box_stale_ms": stats(stale),
    }
    later = accepted[1:]
    gaps = np.diff([r["t"] for r in accepted]) * 1e3 if len(accepted) > 1 else []
    after: dict[str, np.ndarray] = {}
    if ticks and "phase" in ticks:  # 11-19 on: the stop's own tick (1) and the hold after it (2) are recorded
        run = ticks["phase"] == 0
        after = {k: v[~run] for k, v in ticks.items()}
        ticks = {k: v[run] for k, v in ticks.items()} if run.any() else {}
    t_end = float(ticks["t"][-1]) if ticks else (rows[-1]["t"] if rows else 0.0)
    executed = _executed_steps(rows, t_end, dt)
    report["timing"] = {
        "late_steps": stats([int(r["k0"]) - 1 for r in later]),
        "steps_run_per_chunk": stats([len(e["steps"]) for e in executed]),
        # each executed step's duration over dt: 1 = the policy's speed, more = slowed for a joint (retime)
        "stretch": stats([x for e in executed for x in e["stretch"]]),
        "chunk_interval_ms": stats(gaps),
    }

    arm: dict[str, Any] = {}
    if ticks:
        t = ticks["t"]
        loop_ms = np.diff(t) * 1e3
        report["timing"]["loop_ms"] = stats(loop_ms)
        report["timing"]["loop_hz"] = round(float(1e3 / np.mean(loop_ms)), 1) if loop_ms.size else None
        fk = fk or tcp_fk(meta, repo_root)
        P = fk(ticks["q"])[:, :3, 3]
        P_cmd = fk(ticks["q_cmd"])[:, :3, 3]
        moving = ticks["chunk"] >= 0
        arm["tracking_mm"] = stats(np.linalg.norm(P - P_cmd, axis=1)[moving] * 1e3)
        errs = []
        for e in executed:
            if not e["steps"]:
                continue
            due = np.asarray(e["due"])
            inside = (due >= t[0]) & (due <= t[-1])
            achieved = np.stack([np.interp(due[inside], t, P[:, i]) for i in range(3)], axis=1)
            target = e["targets"][np.asarray(e["steps"])[inside]]
            errs.extend((np.linalg.norm(achieved - target, axis=1) * 1e3).tolist())
        arm["target_mm"] = stats(errs)
        arm["path_mm"] = round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1e3), 1)
        arm["net_mm"] = round(float(np.linalg.norm(P[-1] - P[0]) * 1e3), 1)
        f = ticks["f_ext"][:, :3]
        arm["force_n"] = stats(np.linalg.norm(f - f[0], axis=1))
        # what the stop holds against force_n (11-19 on): the force averaged over the loop's window
        from tools.fr3.rollout_execute import FORCE_WINDOW_S, LOOP_HZ

        w = max(1, round(FORCE_WINDOW_S * LOOP_HZ))
        avg = np.stack([np.convolve(f[:, i] - f[0, i], np.ones(w) / w, mode="valid") for i in range(3)], axis=1) if len(f) >= w else f - f[0]
        arm["force_avg_n"] = stats(np.linalg.norm(avg, axis=1))
        speed = np.abs(ticks["dq"]).max(axis=1)
        arm["joint_speed_rad_s"] = stats(speed)
        from tools.fr3 import fr3_kinematics as kin

        q = ticks["q"]
        margin = np.degrees(np.minimum(q - kin._WALL_LOWER, kin._WALL_UPPER - q)).min(axis=1)
        arm["wall_margin_deg"] = stats(margin)
        arm["wall_joint"] = int(np.degrees(np.minimum(q - kin._WALL_LOWER, kin._WALL_UPPER - q)).min(axis=0).argmin()) + 1
    else:
        arm["note"] = "no ticks: the arm was never commanded (the first chunk was refused, or the run stopped before it)"
    if after:
        fk = fk or tcp_fk(meta, repo_root)
        report["stop"] = _stop_block(ticks, after, fk)
    report["predictor_at_stop"] = [ev for ev in events if ev.get("kind") == "predictor"]
    report["prediction_timing_ms"] = [
        {k: round((v - r["timing"]["requested"]) * 1e3, 1) for k, v in r["timing"].items()}
        for r in rows if (r.get("timing") or {}).get("requested") is not None
    ]
    report["motion"] = arm
    report["gripper"] = [ev for ev in events if ev.get("kind") == "gripper"]
    report["headroom"] = _headroom(report, limits, n_exec)
    return report


def _stop_block(ticks: dict[str, np.ndarray], after: dict[str, np.ndarray], fk) -> dict[str, Any]:
    """The tick that tripped the stop (how long after the one before, how far off its command) and the hold
    after it (how far the arm went past where the stop found it)."""
    out: dict[str, Any] = {}
    trip = np.flatnonzero(after["phase"] == 1)
    if trip.size:
        i = int(trip[0])
        P = fk(after["q"][i : i + 1])[0, :3, 3]
        P_cmd = fk(after["q_cmd"][i : i + 1])[0, :3, 3]
        out["trip"] = {
            "loop_dt_ms": round(float(after["t"][i] - ticks["t"][-1]) * 1e3, 1) if ticks else None,
            "tracking_mm": round(float(np.linalg.norm(P - P_cmd)) * 1e3, 1),
            "joint_speed_rad_s": round(float(np.abs(after["dq"][i]).max()), 3),
        }
    hold = after["phase"] == 2
    if hold.any():
        P = fk(after["q"][hold])[:, :3, 3]
        out["hold"] = {"ticks": int(hold.sum()), "travel_mm": round(float(np.linalg.norm(P - P[0], axis=1).max()) * 1e3, 1),
                       "joint_speed_first_rad_s": round(float(np.abs(after["dq"][hold][0]).max()), 3)}
    return out


def _headroom(report: dict[str, Any], limits: dict[str, Any], n_exec: int) -> list[dict[str, Any]]:
    """Worst value against its stop; ``use`` = worst / limit (for the wall: limit / closest margin)."""
    rows = []

    def add(name: str, block: dict[str, Any] | None, limit: float | None, unit: str, *, worst: str = "max",
            inverse: bool = False) -> None:
        if not block or not block.get("n") or limit is None:
            return
        value = float(block[worst])
        use = (limit / value if value > 0 else math.inf) if inverse else (value / limit if limit else math.inf)
        rows.append({"what": name, "worst": round(value, 2), "limit": limit, "unit": unit, "use": round(use, 3)})

    ins, tim, arm = report["inputs"], report["timing"], report["motion"]
    s = 1e3
    add("frame age", ins["age_ms"], _ms(limits.get("frame_age_s"), s), "ms")
    add("prediction latency", ins["latency_ms"], _ms(limits.get("predict_s"), s), "ms")
    add("BOX staleness", ins["box_stale_ms"], _ms(limits.get("box_stale_s"), s), "ms")
    add("late steps", tim["late_steps"], float(n_exec) if n_exec else None, "steps")
    add("loop period", tim.get("loop_ms"), _ms(limits.get("loop_stall_s"), s), "ms")
    add("tracking", arm.get("tracking_mm"), limits.get("tracking_mm"), "mm")
    if limits.get("force_peak_n") is not None:  # 11-19 on: averaged against force_n, one tick against the peak
        add("external force (50 ms)", arm.get("force_avg_n"), limits.get("force_n"), "N")
        add("external force (tick)", arm.get("force_n"), limits.get("force_peak_n"), "N")
    else:
        add("external force", arm.get("force_n"), limits.get("force_n"), "N")
    add("joint speed", arm.get("joint_speed_rad_s"), limits.get("joint_speed_rad_s"), "rad/s")
    add("wall margin", arm.get("wall_margin_deg"), limits.get("wall_deg"), "deg", worst="min", inverse=True)
    return rows


def _ms(value: Any, scale: float) -> float | None:
    return None if value is None else round(float(value) * scale, 3)


def pooled(reports: list[dict[str, Any]], record_dirs: list[Path]) -> dict[str, Any]:
    """Inputs and timing over several runs (the per-prediction and per-tick values, not the per-run stats)."""
    age, lat, late, loop = [], [], [], []
    for d in record_dirs:
        _, rows, ticks, _ = load(d)
        age += [r["age_ms"] for r in rows if r.get("age_ms")]
        lat += [r["latency_ms"] for r in rows if r.get("latency_ms")]
        late += [int(r["k0"]) - 1 for r in [r for r in rows if r.get("accepted")][1:]]
        if ticks:
            t = ticks["t"][ticks["phase"] == 0] if "phase" in ticks else ticks["t"]
            loop += (np.diff(t) * 1e3).tolist()
    return {"runs": len(reports), "age_ms": stats(age), "latency_ms": stats(lat), "late_steps": stats(late),
            "loop_ms": stats(loop)}


def _fmt_stats(block: dict[str, Any] | None, unit: str) -> str:
    if not block or not block.get("n"):
        return "-"
    return f"p50 {block['p50']:g}  p95 {block['p95']:g}  max {block['max']:g} {unit}  (n={block['n']})"


def format_text(report: dict[str, Any]) -> str:
    r = report["result"]
    verdict = "completed" if r.get("completed") else f"stopped: {r.get('stopped') or '?'}"
    lines = [
        f"== {report['record']}",
        f"stage {report['stage']} ({report['stage_label']}), arm {report['arm']}, frames {report['frame_source']}",
        f"result: {verdict}; {report['chunks']['accepted']} chunk(s) taken, {report['chunks']['refused']} refused;"
        f" {r.get('duration_s', '-')} s, {r.get('ticks', '-')} ticks",
    ]
    for why in report["chunks"]["refusals"]:
        lines.append(f"  refused: {why}")
    ins, tim, arm = report["inputs"], report["timing"], report["motion"]
    lines += [
        "inputs",
        f"  frame age          {_fmt_stats(ins['age_ms'], 'ms')}",
        f"  prediction latency {_fmt_stats(ins['latency_ms'], 'ms')}",
        f"  BOX staleness      {_fmt_stats(ins['box_stale_ms'], 'ms')}",
        "timing",
        f"  later chunks late  {_fmt_stats(tim['late_steps'], 'steps')}",
        f"  steps run / chunk  {_fmt_stats(tim['steps_run_per_chunk'], 'steps')}",
        f"  step stretch       {_fmt_stats(tim.get('stretch'), 'x dt')}  (1 = the policy's speed; more = slowed for a joint)",
        f"  chunk interval     {_fmt_stats(tim['chunk_interval_ms'], 'ms')}",
    ]
    if "loop_ms" in tim:
        lines.append(f"  loop period        {_fmt_stats(tim['loop_ms'], 'ms')}  ({tim.get('loop_hz')} Hz)")
    lines.append("arm")
    if "note" in arm:
        lines.append(f"  {arm['note']}")
    else:
        lines += [
            f"  TCP vs command     {_fmt_stats(arm['tracking_mm'], 'mm')}",
            f"  TCP vs target      {_fmt_stats(arm['target_mm'], 'mm')}  (at each executed step's due time)",
            f"  moved              {arm['path_mm']:g} mm along the path, {arm['net_mm']:g} mm net",
            f"  external force     {_fmt_stats(arm['force_n'], 'N')}  (against the first tick)",
            f"    averaged 50 ms   {_fmt_stats(arm.get('force_avg_n'), 'N')}",
            f"  fastest joint      {_fmt_stats(arm['joint_speed_rad_s'], 'rad/s')}",
            f"  wall margin        min {arm['wall_margin_deg']['min']:g} deg (j{arm['wall_joint']})",
        ]
    for ev in report["gripper"]:
        lines.append(f"  gripper {ev.get('action')} at step {ev.get('cursor')}")
    stop = report.get("stop") or {}
    if stop.get("trip"):
        tr = stop["trip"]
        lines.append(f"  the stop's tick   {tr['loop_dt_ms']} ms after the one before, TCP {tr['tracking_mm']} mm off its "
                     f"command, fastest joint {tr['joint_speed_rad_s']} rad/s")
    if stop.get("hold"):
        h = stop["hold"]
        lines.append(f"  hold after stop   TCP moved {h['travel_mm']} mm from where the stop found it ({h['ticks']} ticks)")
    for ev in report.get("predictor_at_stop") or []:
        lines.append(f"  predictor at stop in '{ev.get('phase')}' for {(ev['t'] - ev['since']) * 1e3:.0f} ms")
    for k, tm in enumerate(report.get("prediction_timing_ms") or []):
        phases = ", ".join(f"{name} +{v:g}" for name, v in tm.items() if name != "requested")
        lines.append(f"  prediction {k} ms after request: {phases}")
    if report["headroom"]:
        lines.append("headroom (worst / stop)")
        if report["inputs"]["age_ms"].get("n", 0) < FEW_PREDICTIONS:
            lines.append(f"  (fewer than {FEW_PREDICTIONS} predictions: stage 1 makes one; set input limits from "
                         "stage 2-3 runs, pooled)")
        for h in report["headroom"]:
            flag = "  <- near the stop" if h["use"] >= 0.8 else ""
            lines.append(f"  {h['what']:<19}{h['worst']:>9g} / {h['limit']:g} {h['unit']:<6} {h['use'] * 100:5.0f}%{flag}")
    return "\n".join(lines)


def format_pooled(p: dict[str, Any]) -> str:
    return "\n".join([
        f"== pooled over {p['runs']} runs",
        f"  frame age          {_fmt_stats(p['age_ms'], 'ms')}",
        f"  prediction latency {_fmt_stats(p['latency_ms'], 'ms')}",
        f"  later chunks late  {_fmt_stats(p['late_steps'], 'steps')}",
        f"  loop period        {_fmt_stats(p['loop_ms'], 'ms')}",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", type=Path, nargs="+", help="execute record directories (outputs/rollout/records/execute_*)")
    ap.add_argument("--json", type=Path, help="also write the reports here")
    args = ap.parse_args(argv)
    reports = [analyze(d) for d in args.records]
    print("\n\n".join(format_text(r) for r in reports))
    out: dict[str, Any] = {"runs": reports}
    if len(reports) > 1:
        out["pooled"] = pooled(reports, args.records)
        print("\n" + format_pooled(out["pooled"]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
