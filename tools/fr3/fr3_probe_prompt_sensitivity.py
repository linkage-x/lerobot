#!/usr/bin/env python3
"""Does this checkpoint still read its prompt?

Every episode this rig has trained on carries the same instruction. A constant input produces
no discriminative gradient, so LoRA has had no reason to keep attending to the language tokens
-- whatever it does with them can be absorbed into a bias. That makes "the policy is language
conditioned" an architectural fact rather than a measured one, and multi-task training rests on
the measured version.

The measurement is a paired ablation on stored frames. For one frame we predict the action chunk
several times, changing exactly one thing each time:

    A  = chunk(recorded prompt, noise seed s0)     the reference
    A' = chunk(recorded prompt, noise seed s1)     the noise floor
    B  = chunk(other prompt,    noise seed s0)     the prompt effect
    N  = chunk at the next sampled frame, seed s0  a real, small observation change

The comparison that matters is B against A', not B against zero. pi0.5 is a flow-matching
policy: it integrates an action out of sampled noise, so two runs of the *same* prompt already
differ, and any non-zero B would otherwise read as "it listened". If swapping the entire task
instruction moves the action less than re-rolling the noise does, the prompt is not a control
input in any usable sense -- the policy's own run-to-run spread swamps it.

N is the upper reference and comes free: it is the distance between two chunks this policy
itself produced from genuinely different observations a fraction of a second apart. It says what
scale a real conditioning signal has on this rig.

Reads nothing but archived frames; moves no robot; trains nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The two stand-in instructions. One is a plausible manipulation task on this rig's own hardware,
# the other is a task the arm has never been asked to do; a policy that reads its prompt should
# be disturbed more by the second than the first, and a policy that ignores it by neither.
DEFAULT_ALTERNATE_PROMPTS: tuple[str, ...] = (
    "Pick up the cube and place it in the box",
    "Wipe the table with the sponge",
)

# Two seeds, not one. The floor has to be measured on the same frames as the effect, or the two
# numbers are not comparable.
DEFAULT_SEEDS: tuple[int, int] = (0, 1)

# The action contract these views are built on: three translation deltas, a yaw delta, a gripper
# command. Named here because every metric below reads specific columns of it.
TRANSLATION_DIMS = slice(0, 3)
YAW_DIM = 3
GRIPPER_DIM = 4


class ProbeError(RuntimeError):
    """A checkpoint or dataset that cannot answer this question."""


@dataclass
class ChunkDiff:
    """How far apart two predicted chunks are, in units the roadmap already argues in."""

    step_median_mm: float
    step_max_mm: float
    endpoint_mm: float
    yaw_deg: float
    gripper: float

    @property
    def summary(self) -> str:
        return (
            f"step {self.step_median_mm:.3f} mm (max {self.step_max_mm:.3f}) · "
            f"endpoint {self.endpoint_mm:.3f} mm · yaw {self.yaw_deg:.4f} deg · "
            f"grip {self.gripper:.4f}"
        )


def chunk_difference(left: np.ndarray, right: np.ndarray) -> ChunkDiff:
    """Compare two action chunks of shape (chunk, 5).

    Two views of the same difference, because they answer different questions. The per-step
    figure is the distance between two commands the robot would receive on the same tick. The
    endpoint figure integrates the chunk and says how far apart the arm ends up, which is the
    quantity this project argues in -- a difference that alternates sign is loud per step and
    goes nowhere, and only the second number can tell.

    Summing is the exact reconstruction here, not an approximation: `fr3_gui_replay_runtime.py`
    integrates this contract by adding the translation delta in the base frame, and every delta
    rotation is a pure z-rotation (drx/dry are dropped from the view), which commute.
    """
    if left.shape != right.shape:
        raise ProbeError(f"chunks disagree on shape: {left.shape} vs {right.shape}")
    if left.ndim != 2 or left.shape[1] <= GRIPPER_DIM:
        raise ProbeError(f"expected a (chunk, >={GRIPPER_DIM + 1}) chunk, got {left.shape}")

    delta = left - right
    step_mm = np.linalg.norm(delta[:, TRANSLATION_DIMS], axis=1) * 1000.0
    endpoint_mm = float(
        np.linalg.norm(delta[:, TRANSLATION_DIMS].sum(axis=0)) * 1000.0
    )
    return ChunkDiff(
        step_median_mm=float(np.median(step_mm)),
        step_max_mm=float(np.max(step_mm)),
        endpoint_mm=endpoint_mm,
        yaw_deg=float(abs(np.degrees(delta[:, YAW_DIM].sum()))),
        gripper=float(np.mean(np.abs(delta[:, GRIPPER_DIM]))),
    )


@dataclass
class ProbeSummary:
    """The three numbers the verdict is read off, plus what produced them."""

    frames: int
    noise_floor_mm: float
    prompt_effect_mm: dict[str, float]
    neighbour_mm: float
    prompt_over_floor: dict[str, float]
    frames_where_prompt_beats_floor: dict[str, str] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def summarise(
    noise: Sequence[ChunkDiff],
    prompts: dict[str, Sequence[ChunkDiff]],
    neighbours: Sequence[ChunkDiff],
) -> ProbeSummary:
    floor = _median([d.step_median_mm for d in noise])
    effects = {name: _median([d.step_median_mm for d in diffs]) for name, diffs in prompts.items()}
    ratios = {
        name: (effect / floor if floor else float("inf")) for name, effect in effects.items()
    }
    # A paired sign count next to the ratio of medians. Under "the prompt is ignored" the two
    # differences are drawn from the same spread, so this sits near half; a ratio can be pushed
    # around by a few frames, this cannot.
    beats: dict[str, str] = {}
    for name, diffs in prompts.items():
        wins = sum(
            1 for prompt_diff, noise_diff in zip(diffs, noise, strict=False)
            if prompt_diff.step_median_mm > noise_diff.step_median_mm
        )
        beats[name] = f"{wins}/{min(len(diffs), len(noise))}"
    return ProbeSummary(
        frames=len(noise),
        noise_floor_mm=floor,
        prompt_effect_mm=effects,
        neighbour_mm=_median([d.step_median_mm for d in neighbours]),
        prompt_over_floor=ratios,
        frames_where_prompt_beats_floor=beats,
        detail={
            "noise_floor": {
                "endpoint_mm": _median([d.endpoint_mm for d in noise]),
                "yaw_deg": _median([d.yaw_deg for d in noise]),
                "gripper": _median([d.gripper for d in noise]),
            },
            "prompt": {
                name: {
                    "endpoint_mm": _median([d.endpoint_mm for d in diffs]),
                    "yaw_deg": _median([d.yaw_deg for d in diffs]),
                    "gripper": _median([d.gripper for d in diffs]),
                }
                for name, diffs in prompts.items()
            },
        },
    )


def verdict(summary: ProbeSummary) -> str:
    """One sentence, stated so it can be wrong.

    The thresholds are not calibrated against anything -- there is no second checkpoint to
    calibrate against yet. They exist so the reading is written down before the number is,
    and the raw ratio is always printed next to the verdict.
    """
    if not summary.prompt_over_floor:
        return "no alternate prompt was run; nothing to compare"
    strongest = max(summary.prompt_over_floor.values())
    if strongest < 1.5:
        return (
            "the language channel is inert: swapping the whole instruction moves the action no "
            "more than re-rolling the flow-matching noise does"
        )
    if strongest < 3.0:
        return (
            "the language channel is weak: measurable, but of the same order as this policy's "
            "own sampling spread"
        )
    return "the language channel is live: the prompt moves the action well past the noise floor"


def parse_indices(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def sample_frames(length: int, count: int) -> list[int]:
    """Evenly spaced frames across an episode, endpoints included.

    Spread rather than the first N: the prompt's job, if it has one, is most likely to show at
    the phase boundaries -- approach, close, transfer -- and the opening frames of every episode
    on this rig look nearly alike.
    """
    if length <= 0:
        raise ProbeError("episode has no frames")
    if count <= 1:
        return [0]
    count = min(count, length)
    return sorted({int(round(i * (length - 1) / (count - 1))) for i in range(count)})


def _chunk(
    observation: dict[str, np.ndarray],
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
    task: str,
    seed: int,
    robot_type: str | None,
    use_amp: bool,
) -> np.ndarray:
    """One action chunk, with the flow-matching noise pinned to `seed`.

    The seed is set immediately before the policy call rather than once per frame: the
    preprocessor tokenizes the prompt in between, and a step that ever consumed the global RNG
    would otherwise silently make the "same seed" conditions differ -- which is exactly the
    comparison this script rests on.
    """
    from contextlib import nullcontext

    import fr3_act_infer_real_runtime as infer_runtime

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        prepared = infer_runtime.prepare_observation_for_inference(
            dict(observation), device, task=task, robot_type=robot_type
        )
        processed = preprocessor(prepared)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        actions = policy.predict_action_chunk(processed)
        return np.asarray(
            postprocessor(actions).squeeze(0).detach().cpu().to(torch.float32).numpy(),
            dtype=np.float64,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--episodes",
        default="22,40",
        help="Default is the pair held out of L4/L5 training. The prompt is constant across the "
             "whole view, so held-out and trained episodes answer the same question; the holdout "
             "is used only so the frames are ones no checkpoint has memorised.",
    )
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--prompt", action="append", default=None, help="Alternate prompt; repeatable.")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write the full result here.")
    args = parser.parse_args(argv)

    import fr3_act_infer_real_runtime as infer_runtime
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    seeds = parse_indices(args.seeds)
    if len(seeds) < 2:
        raise ProbeError("--seeds needs two values: without a second one there is no noise floor")
    alternates = list(args.prompt or DEFAULT_ALTERNATE_PROMPTS)

    pretrained_dir = infer_runtime.resolve_pretrained_model_dir(args.checkpoint)
    train_cfg = infer_runtime.load_train_config(pretrained_dir)
    dataset_root = infer_runtime.resolve_dataset_root(pretrained_dir, train_cfg, args.dataset_root)
    ds_meta = infer_runtime.load_dataset_metadata(dataset_root, train_cfg.dataset.repo_id)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, preprocessor, postprocessor = infer_runtime.load_policy_stack(
        pretrained_dir, ds_meta=ds_meta, device=device
    )
    use_amp = bool(policy.config.use_amp)
    input_feature_keys = list(policy.config.input_features)

    print(f"[INFO] checkpoint={pretrained_dir}")
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] device={device} use_amp={use_amp} chunk={policy.config.chunk_size}")
    print(f"[INFO] seeds={seeds} alternates={alternates}")

    noise_diffs: list[ChunkDiff] = []
    prompt_diffs: dict[str, list[ChunkDiff]] = {prompt: [] for prompt in alternates}
    neighbour_diffs: list[ChunkDiff] = []
    recorded_prompts: set[str] = set()

    for episode_idx in parse_indices(args.episodes):
        dataset = LeRobotDataset(
            train_cfg.dataset.repo_id, root=dataset_root, episodes=[episode_idx], image_transforms=None
        )
        previous_reference: np.ndarray | None = None
        for frame_idx in sample_frames(len(dataset), args.frames_per_episode):
            item = dataset[frame_idx]
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ProbeError(f"episode {episode_idx} frame {frame_idx} carries no task string")
            recorded_prompts.add(task)
            observation = {}
            for key in input_feature_keys:
                value = item[key].detach().cpu().numpy()
                if key.startswith("observation.images."):
                    value = np.moveaxis(value, 0, -1)
                    value = np.clip(np.rint(value * 255.0), 0.0, 255.0).astype(np.uint8)
                observation[key] = np.asarray(value)

            def run(prompt: str, seed: int) -> np.ndarray:
                return _chunk(
                    observation,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    task=prompt,
                    seed=seed,
                    robot_type=dataset.meta.robot_type,
                    use_amp=use_amp,
                )

            reference = run(task, seeds[0])
            noise = chunk_difference(reference, run(task, seeds[1]))
            noise_diffs.append(noise)
            line = [f"[PROBE] ep={episode_idx:3d} f={frame_idx:4d} floor {noise.step_median_mm:7.3f} mm"]
            for prompt in alternates:
                diff = chunk_difference(reference, run(prompt, seeds[0]))
                prompt_diffs[prompt].append(diff)
                line.append(f"| {prompt[:22]:22s} {diff.step_median_mm:7.3f} mm")
            if previous_reference is not None:
                neighbour_diffs.append(chunk_difference(reference, previous_reference))
            previous_reference = reference
            print(" ".join(line))

    summary = summarise(noise_diffs, prompt_diffs, neighbour_diffs)
    print()
    print(f"[SUMMARY] frames={summary.frames}  recorded prompt(s)={sorted(recorded_prompts)}")
    print(f"[SUMMARY] noise floor (same prompt, other seed)  {summary.noise_floor_mm:8.3f} mm / step")
    for prompt, effect in summary.prompt_effect_mm.items():
        print(
            f"[SUMMARY] prompt effect  {effect:8.3f} mm / step "
            f"= {summary.prompt_over_floor[prompt]:5.2f}x floor "
            f"(beats floor on {summary.frames_where_prompt_beats_floor[prompt]} frames)  <- {prompt!r}"
        )
    print(f"[SUMMARY] neighbouring frame (a real observation change)  {summary.neighbour_mm:8.3f} mm / step")
    print(f"[VERDICT] {verdict(summary)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "checkpoint": str(pretrained_dir),
                    "dataset_root": str(dataset_root),
                    "recorded_prompts": sorted(recorded_prompts),
                    "alternate_prompts": alternates,
                    "seeds": seeds,
                    "summary": asdict(summary),
                    "verdict": verdict(summary),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
