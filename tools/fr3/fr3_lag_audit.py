#!/usr/bin/env python3
"""Audit FR3 dataset obs-cmd lag from recorded parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EE_POSITION_NAMES = ('ee.x', 'ee.y', 'ee.z')
EE_QUAT_NAMES = ('ee.qx', 'ee.qy', 'ee.qz', 'ee.qw')
GRIPPER_NAME = 'gripper.pos'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit FR3 observation-command lag from local datasets.')
    parser.add_argument('datasets', nargs='+', help='Dataset roots or glob patterns.')
    parser.add_argument('--max-ee-lag', type=int, default=20, help='Maximum positive frame lag to search for EE axes.')
    parser.add_argument(
        '--max-gripper-lag',
        type=int,
        default=60,
        help='Maximum positive frame lag to search for gripper lag.',
    )
    parser.add_argument(
        '--gripper-transition-threshold',
        type=float,
        default=0.2,
        help='Minimum absolute action delta used to detect gripper command transitions.',
    )
    parser.add_argument(
        '--gripper-target-tolerance',
        type=float,
        default=0.02,
        help='Absolute observation tolerance used to mark a gripper transition as reached.',
    )
    return parser.parse_args(argv)


def expand_dataset_roots(specs: list[str]) -> list[Path]:
    dataset_roots: list[Path] = []
    for spec in specs:
        spec_path = Path(spec)
        if spec_path.is_absolute():
            matches = [path for path in spec_path.parent.glob(spec_path.name)]
        else:
            matches = sorted(Path().glob(spec))
        if matches:
            dataset_roots.extend(match.resolve() for match in matches)
            continue
        dataset_roots.append(Path(spec).resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for dataset_root in dataset_roots:
        if dataset_root in seen:
            continue
        seen.add(dataset_root)
        deduped.append(dataset_root)
    return deduped


def load_dataset_info(dataset_root: Path) -> dict:
    return json.loads((dataset_root / 'meta' / 'info.json').read_text(encoding='utf-8'))


def extract_state_indices(dataset_root: Path) -> dict[str, int]:
    info = load_dataset_info(dataset_root)
    state_names = info.get('features', {}).get('observation.state', {}).get('names')
    if not isinstance(state_names, list):
        raise KeyError(f'{dataset_root}: meta/info.json is missing observation.state names')
    required_names = [*EE_POSITION_NAMES, *EE_QUAT_NAMES, GRIPPER_NAME]
    missing_names = [name for name in required_names if name not in state_names]
    if missing_names:
        raise KeyError(f'{dataset_root}: observation.state is missing required names {missing_names}')
    return {name: state_names.index(name) for name in required_names}


def load_scalar_arrays(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquet_paths = sorted((dataset_root / 'data').rglob('*.parquet'))
    if not parquet_paths:
        raise FileNotFoundError(f'{dataset_root}: no parquet files found under data/')

    table = pa.concat_tables(
        [
            pq.read_table(
                str(parquet_path),
                columns=['observation.state', 'action', 'episode_index', 'frame_index'],
            )
            for parquet_path in parquet_paths
        ]
    )
    payload = table.to_pydict()
    return (
        np.asarray(payload['observation.state'], dtype=np.float64),
        np.asarray(payload['action'], dtype=np.float64),
        np.asarray(payload['episode_index'], dtype=np.int64),
        np.asarray(payload['frame_index'], dtype=np.int64),
    )


def compute_best_positive_lag(
    *,
    observation_values: np.ndarray,
    action_values: np.ndarray,
    episode_index: np.ndarray,
    max_lag: int,
) -> tuple[int, float]:
    best_lag = 0
    best_mse = float('inf')
    for lag in range(max_lag + 1):
        episode_errors: list[float] = []
        for episode in sorted(set(episode_index.tolist())):
            mask = episode_index == episode
            obs = observation_values[mask]
            act = action_values[mask]
            if len(obs) - lag <= 0:
                continue
            episode_errors.append(float(np.mean((obs[lag:] - act[:-lag or None]) ** 2)))
        if not episode_errors:
            continue
        mse = float(np.mean(episode_errors))
        if mse < best_mse:
            best_lag = lag
            best_mse = mse
    return best_lag, best_mse


def analyze_gripper_transitions(
    *,
    observation_values: np.ndarray,
    action_values: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    transition_threshold: float,
    target_tolerance: float,
) -> list[dict[str, int | float | None]]:
    rows: list[dict[str, int | float | None]] = []
    for episode in sorted(set(episode_index.tolist())):
        mask = episode_index == episode
        obs = observation_values[mask]
        act = action_values[mask]
        frames = frame_index[mask]
        transition_indices = np.where(np.abs(np.diff(act)) > transition_threshold)[0]
        for transition_index in transition_indices:
            start_value = float(obs[transition_index])
            target_value = float(act[transition_index + 1])
            if target_value > start_value:
                reached = np.where(obs[transition_index + 1 :] >= target_value - target_tolerance)[0]
            else:
                reached = np.where(obs[transition_index + 1 :] <= target_value + target_tolerance)[0]
            rows.append(
                {
                    'episode': int(episode),
                    'frame': int(frames[transition_index]),
                    'cmd0': float(act[transition_index]),
                    'cmd1': target_value,
                    'obs0': start_value,
                    'reach_delay_frames': None if len(reached) == 0 else int(reached[0] + 1),
                }
            )
    return rows


def audit_dataset(
    dataset_root: Path,
    *,
    max_ee_lag: int,
    max_gripper_lag: int,
    gripper_transition_threshold: float,
    gripper_target_tolerance: float,
) -> int:
    info = load_dataset_info(dataset_root)
    fps = float(info.get('fps', 30.0))
    state_indices = extract_state_indices(dataset_root)
    observation_state, action, episode_index, frame_index = load_scalar_arrays(dataset_root)

    print(f'DATASET {dataset_root}')
    for feature_name in EE_POSITION_NAMES:
        index = state_indices[feature_name]
        lag, mse = compute_best_positive_lag(
            observation_values=observation_state[:, index],
            action_values=action[:, index],
            episode_index=episode_index,
            max_lag=max_ee_lag,
        )
        print(f'  ee_lag {feature_name}: frames={lag} seconds={lag / fps:.3f} mse={mse:.9f}')

    gripper_index = state_indices[GRIPPER_NAME]
    gripper_lag, gripper_mse = compute_best_positive_lag(
        observation_values=observation_state[:, gripper_index],
        action_values=action[:, gripper_index],
        episode_index=episode_index,
        max_lag=max_gripper_lag,
    )
    print(f'  grip_lag: frames={gripper_lag} seconds={gripper_lag / fps:.3f} mse={gripper_mse:.9f}')

    transition_rows = analyze_gripper_transitions(
        observation_values=observation_state[:, gripper_index],
        action_values=action[:, gripper_index],
        episode_index=episode_index,
        frame_index=frame_index,
        transition_threshold=gripper_transition_threshold,
        target_tolerance=gripper_target_tolerance,
    )
    if not transition_rows:
        print('  grip_transitions: none')
    else:
        for row in transition_rows:
            print(
                '  grip_transition: '
                f"episode={row['episode']} frame={row['frame']} "
                f"cmd={row['cmd0']:.3f}->{row['cmd1']:.3f} "
                f"obs0={row['obs0']:.3f} "
                f"reach_delay_frames={row['reach_delay_frames']}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_roots = expand_dataset_roots(args.datasets)
    if not dataset_roots:
        print('No dataset roots matched.', file=sys.stderr)
        return 1

    exit_code = 0
    for dataset_root in dataset_roots:
        try:
            audit_dataset(
                dataset_root,
                max_ee_lag=args.max_ee_lag,
                max_gripper_lag=args.max_gripper_lag,
                gripper_transition_threshold=args.gripper_transition_threshold,
                gripper_target_tolerance=args.gripper_target_tolerance,
            )
        except Exception as exc:
            exit_code = 1
            print(f'DATASET {dataset_root}')
            print(f'  ERROR: {type(exc).__name__}: {exc}')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
