#!/usr/bin/env python
"""Relabel the policy's own steps of whole-trial recordings with the funnel's chunk (offline DAgger).

Why: a whole-trial recording (`expert_source.json` `recordsPolicyFrames`) keeps every step the
policy drove before the funnel took over, labelled with what the *policy* did -- and the policy
descends straight above an off-centre peg. On the corrective-153 view those frames are 43% of
the training set and 4:1 against the funnel's own frames in the 60-300 mm band, so a checkpoint
trained on it learns "go straight down": the 10-22 offline probe saw both 045000 and 071246 point
at the peg (cosine 0.73-0.80) and move ~1 mm sideways in a second against 13 mm in the labels.

The states are the right ones -- the policy's own approach is exactly where it will be at
deployment -- only the labels are wrong, and the expert is a program of state and peg position.
So each such frame gets the chunk the funnel would have commanded from that frame's command,
simulated kinematically (`expert_chunk`), not the recorded future: the recorded future never
corrected, and labelling each step with "move toward the peg" while the chunk still comes from
that future would double the correction (every step of one chunk moving toward the same peg).

The command a delta refers to is integrated from the episode's `start_contract_xyzquat`
(`delta_ee_from_prev_cmd` is world-frame translation, `processor_franka_research3`); it tracks
the measured pose within ~9 mm over the whole policy segment. It is not trusted past the first
funnel frame: a late takeover drops the low policy steps (`policy_off_course`), so the recording
has a hole there. The peg is the funnel's close point: the arm settles a fixed offset away from
its command at the close (measured per source on the gap-free episodes, ~(+3.8, -1.8, -3.2) mm
with a +-0.5 mm IQR), and the funnel commands the close at `closeDzMm` below the peg.

The funnel's own frames are left as recorded: they are the same expert, executed.

Arm A's misses (`--grasp-loop-keep-misses`: graded empty or no_close) are the policy's own states
where it goes wrong, and none of what it did there is to be learnt: every frame is either
relabelled or left unsupervised -- above the height band and past the close too.

Output: a new training view beside the source view (data and meta copied, videos linked) with
`meta/action_overrides.npz`, which `LeRobotDataset` swaps in for those frames' action chunks,
and optionally held-out episodes marked not for training (`meta/gui_annotations.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from tools.fr3.grasp_funnel import FUNNEL_ALIGN_Z_SLACK_M, FUNNEL_CLOSE_DZ_MM, FunnelConfig, _walk  # noqa: E402

ACTION_OVERRIDES_RELATIVE_PATH = Path("meta") / "action_overrides.npz"
ANNOTATION_STORE_RELATIVE_PATH = Path("meta") / "gui_annotations.json"
# Steps from the last setpoint move to the close command: the settle test (5 quiet steps) after
# the arm's own stop. Median 14 (trial150, funnel163) and 16 (corrective153), IQR +-1.
EXPERT_CLOSE_WAIT_STEPS = 15
# Recorded episodes end 10 frames after the close; past that a chunk is padding, as on the rig.
EXPERT_POST_CLOSE_STEPS = 10
# A frame whose integrated command stands further than this from the measured pose is not
# relabelled: the integration has lost the command (in the policy segment it stays within ~9 mm).
MAX_COMMAND_LAG_MM = 15.0
# Only frames this far above the peg or lower are relabelled. Above it the policy is still doing
# its gross aiming from the start pose (~350 mm up), which the funnel never demonstrated: the
# expert there would be a new behaviour (a 0.05 m/s dive from the start), not a correction.
DEFAULT_MAX_HEIGHT_MM = 300.0
# A source episode counts as gap-free (its integrated command trusted up to the close) when no
# measured step exceeds this: funnel and policy steps are <= 1.7 mm, tracking adds ~1.7.
GAP_FREE_STEP_MM = 4.5
# How close a grasp-loop row's `closeXyz` must sit to an episode's measured pose at its close for
# the row to be that episode's. Both are the arm's own reading at the close: on the 303 arm-B
# episodes of trial150 + corrective153 every one matched exactly one row well inside this.
TRIAL_ROW_MATCH_MM = 3.0


def expert_chunk(
    start_xyz: np.ndarray,
    peg_xyz: np.ndarray,
    horizon: int,
    *,
    config: FunnelConfig | None = None,
    close_wait_steps: int = EXPERT_CLOSE_WAIT_STEPS,
    post_close_steps: int = EXPERT_POST_CLOSE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """The funnel's next `horizon` commands from command `start_xyz`, as (dx, dy, dz, drz, gripper).

    The corrective DESCEND line wherever it is no flatter than 45 degrees: straight to the peg's
    XY at `centredZ`, then down to the close. Flatter than that, ALIGN's way: across at the
    current height (up to `alignZ` first if below it), then down. Speeds as the funnel's: the
    align speed across, the fast descent above `slowBelowZ` and the slow one below. The wrist
    holds (drz 0), and the fingers close `close_wait_steps` after the last move.

    Returns the chunk and its padding mask (steps more than `post_close_steps` past the close).
    """

    peg = tuple(float(v) for v in peg_xyz)
    cfg = config or FunnelConfig(pegXyz=peg)  # type: ignore[arg-type]
    close = cfg.closeXyz
    setpoint = tuple(float(v) for v in start_xyz)
    off_mm = math.hypot(setpoint[0] - peg[0], setpoint[1] - peg[1]) * 1000.0
    above_centred_mm = (setpoint[2] - cfg.centredZ) * 1000.0

    legs: list[tuple[tuple[float, float, float], str]] = []
    if off_mm <= cfg.captureXyMm or above_centred_mm >= off_mm:
        if setpoint[2] > cfg.centredZ:
            legs.append(((peg[0], peg[1], cfg.centredZ), "descend"))
    else:
        across_z = setpoint[2]
        if setpoint[2] < cfg.alignZ - FUNNEL_ALIGN_Z_SLACK_M:
            across_z = cfg.alignZ
            legs.append(((setpoint[0], setpoint[1], across_z), "align"))
        legs.append(((peg[0], peg[1], across_z), "align"))
    legs.append((close, "descend"))

    actions = np.zeros((horizon, 5), dtype=np.float32)
    actions[:, 4] = cfg.openGripper
    is_pad = np.zeros(horizon, dtype=bool)
    leg, arrived_at = 0, None
    for k in range(horizon):
        if arrived_at is None:
            goal, kind = legs[leg]
            if kind == "align":
                speed = cfg.alignSpeedMS
            else:
                speed = cfg.fastDescendSpeedMS if setpoint[2] > cfg.slowBelowZ + 1e-9 else cfg.descendSpeedMS
            new, arrived = _walk(setpoint, goal, speed * cfg.controlPeriodS)
            actions[k, :3] = np.subtract(new, setpoint)
            setpoint = new
            if arrived:
                leg += 1
                if leg == len(legs):
                    # The settle wait counts from the last step that moved.
                    arrived_at = k if np.any(actions[k, :3]) else k - 1
            continue
        since = k - arrived_at
        if since > close_wait_steps:
            actions[k, 4] = cfg.closedGripper
        if since > close_wait_steps + post_close_steps:
            is_pad[k] = True
    return actions, is_pad


def command_path(actions: np.ndarray, start_xyz: np.ndarray) -> np.ndarray:
    """`prev_cmd` at every frame: the command each frame's delta was measured from."""

    issued = np.asarray(start_xyz, dtype=np.float64) + np.cumsum(actions[:, :3].astype(np.float64), axis=0)
    return np.vstack([np.asarray(start_xyz, dtype=np.float64)[None], issued[:-1]])


def _read_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise ValueError(f"{root} has no data parquet files")
    frames = pd.concat(
        pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "observation.state", "action", "is_intervention"])
        for f in files
    )
    return frames.sort_values("index").reset_index(drop=True)


def _episode_records(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "meta" / "episode_frames.jsonl"
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["episode_index"])] = record
    return records


def load_trial_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """The grasp-loop rows whose trial became an episode (`dataset.kept`), from run JSONL files.

    A kept miss that never closed has no `closeXyz`; its row carries the dataset `root` instead.
    """

    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kept = row.get("dataset") if row.get("kind") == "trial" else None
            if kept and kept.get("kept") and kept.get("firstEpisode") is not None:
                rows.append(row)
    return rows


def _to_dataset_frame(T_B_Ws: Any, xyz: Any) -> np.ndarray:
    """A base-frame point as the dataset stores poses: inv(T_B_Ws) @ p."""

    T = np.asarray(T_B_Ws, dtype=np.float64)
    return T[:3, :3].T @ (np.asarray(xyz, dtype=np.float64) - T[:3, 3])


def match_trial_row(
    root_name: str,
    episode: int,
    record: dict[str, Any],
    close_states_xyz: np.ndarray | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The row that recorded `episode` of dataset `root_name`: same trial, same first episode, and
    the same close -- or, for an episode that never closed, a row without one from the same root.

    The trial number and episode index alone repeat across runs appending to one dataset and
    across datasets; the close position does not, and neither does the dataset (one root is
    appended to, so its episode indices never repeat). Rows written before the root was recorded
    are matched on the close alone.

    `close_states_xyz` is the measured pose at the onset of each closed run of the episode. The
    row's close is the one that settled, which is not always the first: R4 045000 closes for 1-9
    steps, lifts and comes back (10-27 R6: 5 of 90 episodes, the row's close 5-31 mm from the
    first onset and within 0.5 mm of a later one).
    """

    def same(row: dict[str, Any]) -> bool:
        if int(row["trial"]) != int(record["rollout"]) or int(row["dataset"]["firstEpisode"]) != int(episode):
            return False
        root = row["dataset"].get("root")
        if root is not None and Path(root).name != root_name:
            return False
        if row.get("closeXyz") is None or close_states_xyz is None:
            return row.get("closeXyz") is None and close_states_xyz is None and root is not None
        offsets = _to_dataset_frame(record["T_B_Ws"], row["closeXyz"]) - np.atleast_2d(close_states_xyz)
        return bool(np.linalg.norm(offsets, axis=1).min() * 1000.0 <= TRIAL_ROW_MATCH_MM)

    found = [row for row in rows if same(row)]
    if len(found) > 1:
        raise ValueError(f"episode {episode} matches {len(found)} grasp-loop rows; pass each run's rows once")
    return found[0] if found else None


def _close_index(actions: np.ndarray, closed_below: float = 0.5) -> int | None:
    closed = np.flatnonzero(actions[:, 4] < closed_below)
    return int(closed[0]) if len(closed) else None


def _close_onsets(actions: np.ndarray, closed_below: float = 0.5) -> np.ndarray:
    closed = (actions[:, 4] < closed_below).astype(np.int8)
    return np.flatnonzero(np.diff(np.r_[0, closed]) == 1)


class NoFunnelCloseError(ValueError):
    """A source without a funnel close, relabelled with no funnel offset to borrow."""


def close_tracking_offset(episodes: list[dict[str, Any]]) -> np.ndarray:
    """Median (command - measured) at the close over the gap-free episodes of one source."""

    offsets = []
    for ep in episodes:
        c = ep["close"]
        steps = np.linalg.norm(np.diff(ep["state"][: c + 1, :3], axis=0), axis=1) * 1000.0
        if len(steps) and steps.max() <= GAP_FREE_STEP_MM:
            offsets.append(ep["prev_cmd"][c] + ep["action"][c, :3] - ep["state"][c, :3])
    if not offsets:
        raise ValueError("no gap-free episode to measure the close tracking offset on")
    return np.median(np.asarray(offsets), axis=0)


def _offset_mm_or_none(episodes: list[dict[str, Any]]) -> list[float] | None:
    try:
        return [round(float(v) * 1000.0, 2) for v in close_tracking_offset(episodes)]
    except ValueError:
        return None


def relabel_source(
    root: Path,
    horizon: int,
    max_height_mm: float = DEFAULT_MAX_HEIGHT_MM,
    trial_rows: list[dict[str, Any]] | None = None,
    funnel_offset: np.ndarray | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Per source episode: the relabelled frames' chunks; plus a summary of what was done.

    A policy-only episode (arm A: no funnel frame) closed wherever the policy closed, so its close
    is not the peg. Its peg is the grasp-loop row's `pegXyz` (`trial_rows`), carried into the
    dataset frame and shifted by the same tracking offset as the funnel's closes -- on arm-B
    episodes that reproduces the close-derived peg to (+0.6, -0.3, -0.4) mm (median, 303 eps). Its
    frames from the close on are the policy's off-centre close and what followed it: they get no
    supervision (padding throughout), which `action_supervision_mask` drops from the loss.

    A miss (its row's verdict is not held) gets no supervision on any frame it does not relabel:
    neither above `max_height_mm` nor where the command is lost. One that never closed (no_close)
    has no close to stop at: it is relabelled to its last frame.
    """

    frames = _read_frames(root)
    records = _episode_records(root)
    episodes = []
    for ep, g in frames.groupby("episode_index", sort=True):
        state = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(g["action"].to_numpy()).astype(np.float64)
        close = _close_index(action)
        intervention = g["is_intervention"].to_numpy().astype(float) > 0.5
        if close is None and intervention.any():
            continue
        first_funnel = int(np.argmax(intervention)) if intervention.any() else (len(state) if close is None else close)
        record = records[int(ep)]
        episodes.append(
            {
                "episode": int(ep),
                "state": state,
                "action": action,
                "close": close,
                "first_funnel": first_funnel,
                "policy_only": not intervention.any(),
                "record": record,
                "prev_cmd": command_path(action, np.asarray(record["start_contract_xyzquat"][:3], dtype=np.float64)),
                "row": None
                if not trial_rows
                else match_trial_row(
                    root.name, int(ep), record, None if close is None else state[_close_onsets(action), :3], trial_rows
                ),
            }
        )
    funnel_closed = [ep for ep in episodes if not ep["policy_only"]]
    policy_closed = [ep for ep in episodes if ep["policy_only"] and ep["close"] is not None]
    # A source with no funnel close (all arm A) borrows `funnel_offset`, measured on other sources'
    # funnel closes. Not its own policy closes: those are taken on the move, at ~(+0.7, -0.6, -0.3)
    # mm on R3, while the expert closes after a settle, where the arm sits ~(+3.8, -1.8, -3.2) mm
    # off its command -- and that is the offset `pegXyz` reproduces the funnel's own peg with. The
    # policy's would aim every R3 label 3 mm short in x, the direction the policy already errs in.
    if funnel_closed:
        offset = close_tracking_offset(funnel_closed)
    elif funnel_offset is not None:
        offset = np.asarray(funnel_offset, dtype=np.float64)
    else:
        raise NoFunnelCloseError(
            f"{root.name} has no funnel close to measure the close tracking offset on; relabel it "
            "together with a source that has (build_relabelled_view does), or pass funnel_offset."
        )
    out: dict[int, dict[str, Any]] = {}
    lagged = high = masked = masked_miss = 0
    row_vs_close_mm = []
    for ep in episodes:
        close_dz = np.array([0.0, 0.0, FUNNEL_CLOSE_DZ_MM / 1000.0])
        row_peg = None
        if ep["row"] is not None:
            row_peg = _to_dataset_frame(ep["record"]["T_B_Ws"], ep["row"]["pegXyz"]) + offset
        if ep["policy_only"]:
            if row_peg is None:
                raise ValueError(
                    f"{root.name} episode {ep['episode']} is policy-only: its close is where the policy "
                    "closed, not the peg. Pass the grasp-loop rows that recorded it (--trial-rows)."
                )
            peg = row_peg
        else:
            peg = ep["state"][ep["close"], :3] + offset - close_dz
            if row_peg is not None:
                row_vs_close_mm.append((row_peg - peg) * 1000.0)
        cfg = FunnelConfig(pegXyz=tuple(float(v) for v in peg))  # type: ignore[arg-type]
        missed = ep["row"] is not None and ep["row"].get("verdict") != "held"
        frame_ids, chunks, pads, unsupervised = [], [], [], []

        def unsupervise(t: int) -> None:
            frame_ids.append(t)
            chunks.append(np.zeros((horizon, 5), dtype=np.float32))
            pads.append(np.ones(horizon, dtype=bool))
            unsupervised.append(True)

        for t in range(ep["first_funnel"]):
            skip = None
            if (ep["state"][t, 2] - peg[2]) * 1000.0 > max_height_mm:
                skip = "high"
            elif np.linalg.norm(ep["prev_cmd"][t] - ep["state"][t, :3]) * 1000.0 > MAX_COMMAND_LAG_MM:
                skip = "lagged"
            if skip is not None:
                if missed:
                    unsupervise(t)
                    masked_miss += 1
                elif skip == "high":
                    high += 1
                else:
                    lagged += 1
                continue
            chunk, pad = expert_chunk(ep["prev_cmd"][t], peg, horizon, config=cfg)
            frame_ids.append(t)
            chunks.append(chunk)
            pads.append(pad)
            unsupervised.append(False)
        if ep["policy_only"] and ep["close"] is not None:
            for t in range(ep["close"], len(ep["state"])):
                unsupervise(t)
                masked += 1
        if frame_ids:
            out[ep["episode"]] = {
                "frames": np.asarray(frame_ids, dtype=np.int64),
                "action": np.stack(chunks),
                "is_pad": np.stack(pads),
                # Frames whose override only removes supervision (a policy-only episode's close
                # on, and a miss's frames that are not relabelled).
                "masked": np.asarray(unsupervised, dtype=bool),
                "peg": peg,
                "state": ep["state"],
            }
    summary = {
        "source_root": str(root),
        "episodes": len(episodes),
        "policy_only_episodes": sum(ep["policy_only"] for ep in episodes),
        "relabelled_episodes": len(out),
        "missed_episodes": sum(ep["row"] is not None and ep["row"].get("verdict") != "held" for ep in episodes),
        "never_closed_episodes": sum(ep["close"] is None for ep in episodes),
        "relabelled_frames": int(sum(len(v["frames"]) for v in out.values())) - masked - masked_miss,
        "masked_after_policy_close_frames": masked,
        "masked_unrelabelled_miss_frames": masked_miss,
        "skipped_lagged_frames": lagged,
        "kept_above_max_height_frames": high,
        "max_height_mm": max_height_mm,
        "close_tracking_offset_mm": [round(float(v) * 1000.0, 2) for v in offset],
        "close_tracking_offset_from": "funnel closes" if funnel_closed else "other sources' funnel closes",
        # For reference only: what the policy's own closes would have given.
        "policy_close_offset_mm": _offset_mm_or_none(policy_closed),
        "trial_row_matched_episodes": sum(ep["row"] is not None for ep in episodes),
    }
    if row_vs_close_mm:
        diff = np.asarray(row_vs_close_mm)
        summary["row_peg_minus_close_peg_mm"] = {
            "median": [round(float(v), 2) for v in np.median(diff, axis=0)],
            "max_norm": round(float(np.linalg.norm(diff, axis=1).max()), 2),
        }
    return out, summary


def _view_episode_starts(view_root: Path) -> dict[int, tuple[int, int]]:
    files = sorted((view_root / "meta" / "episodes").glob("*/*.parquet"))
    meta = pd.concat(pd.read_parquet(f, columns=["episode_index", "dataset_from_index", "dataset_to_index"]) for f in files)
    return {int(r.episode_index): (int(r.dataset_from_index), int(r.dataset_to_index)) for r in meta.itertuples()}


def build_relabelled_view(
    view_root: Path,
    source_roots: list[Path],
    out_root: Path,
    *,
    horizon: int,
    max_height_mm: float = DEFAULT_MAX_HEIGHT_MM,
    holdout_source: str | None = None,
    holdout_episodes: int = 0,
    seed: int = 0,
    trial_rows: list[dict[str, Any]] | None = None,
    holdout_named: list[int] | None = None,
) -> dict[str, Any]:
    if out_root.exists():
        raise ValueError(f"{out_root} exists; pick a new name rather than overwrite a view")
    manifest = json.loads((view_root / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("action_mode") != "delta_ee_from_prev_cmd":
        raise ValueError(f"{view_root.name} is {manifest.get('action_mode')!r}; relabelling assumes delta_ee_from_prev_cmd")
    by_name = {root.name: root for root in source_roots}
    view_starts = _view_episode_starts(view_root)
    view_frames = _read_frames(view_root)
    view_state = np.stack(view_frames["observation.state"].to_numpy())

    indices, chunks, pads, summaries = [], [], [], []
    view_eps_by_source: dict[str, list[tuple[int, int]]] = {}
    per_source: dict[str, tuple[dict[int, dict[str, Any]], dict[str, Any]]] = {}
    for entry in manifest["episode_source_index"]:
        name = Path(entry["source_dataset_root"]).name
        view_eps_by_source.setdefault(name, []).append((int(entry["episode_index"]), int(entry["source_episode_index"])))

    def _add(name: str, pairs: list[tuple[int, int]]) -> None:
        relabelled, summary = per_source[name]
        summaries.append(summary)
        for view_ep, source_ep in pairs:
            if source_ep not in relabelled:
                continue
            rec = relabelled[source_ep]
            start, end = view_starts[view_ep]
            if end - start != len(rec["state"]):
                raise ValueError(f"view episode {view_ep} has {end - start} frames, source {name}:{source_ep} {len(rec['state'])}")
            if not np.allclose(view_state[start:end], rec["state"], atol=1e-6):
                raise ValueError(f"view episode {view_ep} is not {name}:{source_ep} frame for frame")
            indices.append(start + rec["frames"])
            chunks.append(rec["action"])
            pads.append(rec["is_pad"])

    # Sources without a funnel close wait for the others: they borrow the others' offset.
    deferred: list[tuple[str, Path, list[tuple[int, int]]]] = []
    for name, pairs in view_eps_by_source.items():
        root = by_name.get(name)
        if root is None:
            continue
        expert = json.loads((root / "meta" / "expert_source.json").read_text(encoding="utf-8")) if (
            root / "meta" / "expert_source.json"
        ).is_file() else {}
        if not expert.get("recordsPolicyFrames"):
            continue
        try:
            per_source[name] = relabel_source(root, horizon, max_height_mm, trial_rows)
        except NoFunnelCloseError:
            deferred.append((name, root, pairs))
            continue
        _add(name, pairs)
    if deferred:
        measured = [s["close_tracking_offset_mm"] for s in summaries if s["close_tracking_offset_from"] == "funnel closes"]
        if not measured:
            raise NoFunnelCloseError(
                f"no source here has a funnel close to measure the tracking offset on: {[d[0] for d in deferred]}"
            )
        funnel_offset = np.median(np.asarray(measured, dtype=np.float64), axis=0) / 1000.0
        for name, root, pairs in deferred:
            per_source[name] = relabel_source(root, horizon, max_height_mm, trial_rows, funnel_offset)
            _add(name, pairs)
    missing = [name for name in view_eps_by_source if name not in by_name]
    if not indices:
        raise ValueError(f"nothing to relabel; sources not given: {missing}")
    index = np.concatenate(indices)
    order = np.argsort(index)
    overrides = {
        "index": index[order].astype(np.int64),
        "action": np.concatenate(chunks)[order].astype(np.float32),
        "action_is_pad": np.concatenate(pads)[order],
    }

    holdout: list[int] = []
    if holdout_source and holdout_episodes:
        candidates = sorted(view for view, _ in view_eps_by_source.get(holdout_source, []))
        if len(candidates) < holdout_episodes:
            raise ValueError(f"{holdout_source} has {len(candidates)} view episodes, fewer than {holdout_episodes}")
        holdout = sorted(int(v) for v in np.random.default_rng(seed).choice(candidates, holdout_episodes, replace=False))
    # Named episodes are held out beside the drawn ones, which stay what the holdout probe reads.
    named = sorted(set(holdout_named or []))
    if unknown := [ep for ep in named if ep not in view_starts]:
        raise ValueError(f"{view_root.name} has no view episodes {unknown}")
    if both := sorted(set(named) & set(holdout)):
        raise ValueError(f"view episodes {both} are both drawn and named")

    out_root.mkdir(parents=True)
    shutil.copytree(view_root / "meta", out_root / "meta")
    shutil.copytree(view_root / "data", out_root / "data")
    (out_root / "videos").symlink_to((view_root / "videos").resolve(), target_is_directory=True)
    np.savez(out_root / ACTION_OVERRIDES_RELATIVE_PATH, **overrides)
    if holdout or named:
        store = {
            "annotations": {
                str(ep): {
                    "episode": ep,
                    "includeInTraining": False,
                    "notes": "funnel_relabel holdout: offline probe episodes, never trained on",
                    "source": "funnel_relabel",
                }
                for ep in holdout + named
            }
        }
        (out_root / ANNOTATION_STORE_RELATIVE_PATH).write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    report = {
        "base_view": str(view_root),
        "horizon": horizon,
        "overridden_frames": int(len(overrides["index"])),
        "total_frames": int(len(view_frames)),
        "sources": summaries,
        "sources_not_given": missing,
        "holdout_source": holdout_source,
        "holdout_view_episodes": holdout,
        "holdout_named_view_episodes": named,
        "close_wait_steps": EXPERT_CLOSE_WAIT_STEPS,
        "post_close_steps": EXPERT_POST_CLOSE_STEPS,
    }
    manifest["repo_id"] = f"local/{out_root.name}"
    manifest["action_overrides"] = report
    (out_root / "meta" / "il_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report


def _episode_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True,
                        help="a whole-trial recording the view was built from (data/ and meta/ are enough)")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50, help="the policy's chunk size")
    parser.add_argument("--max-height-mm", type=float, default=DEFAULT_MAX_HEIGHT_MM)
    parser.add_argument("--holdout-source", default=None, help="source root name to hold episodes out of")
    parser.add_argument("--holdout-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-view-episodes", type=_episode_list, default=[],
                        help="view episodes to hold out as well, e.g. 466-481,490 (any source; the "
                        "holdout probe still reads only the drawn ones)")
    parser.add_argument("--trial-rows", type=Path, action="append", default=[],
                        help="a grasp-loop run's JSONL (outputs/analysis/grasp_loop/grasp_*.jsonl); required "
                        "for sources with arm-A (policy-only) episodes, whose peg comes from the row, and "
                        "what marks a kept miss as one")
    args = parser.parse_args()
    report = build_relabelled_view(
        args.view_root,
        args.source_root,
        args.out_root,
        horizon=args.horizon,
        max_height_mm=args.max_height_mm,
        holdout_source=args.holdout_source,
        holdout_episodes=args.holdout_episodes,
        seed=args.seed,
        trial_rows=load_trial_rows(args.trial_rows),
        holdout_named=args.holdout_view_episodes,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
