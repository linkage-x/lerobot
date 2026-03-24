#!/usr/bin/env python3
"""Compare the current FR3 EE pose against dataset episode start states."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / 'outputs/datasets/lerobotv3_0310_100ep'
_DEFAULT_ROBOT_IP = '192.168.1.208'
_DEFAULT_GRIPPER_PORT = '/dev/ttyUSB0'
_DEFAULT_GRIPPER_BACKEND = 'das'
_DAS_URDF = _REPO_ROOT / 'src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf'


@dataclass(frozen=True)
class EpisodeStartState:
    episode_index: int
    state: Any


@dataclass(frozen=True)
class EpisodeStartDelta:
    episode_index: int
    position_delta_m: Any
    position_distance_m: float
    rotation_delta_deg: float
    gripper_delta: float
    weighted_score: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare current FR3 EE pose to dataset episode start states.')
    parser.add_argument('--dataset', type=Path, default=_DEFAULT_DATASET, help='Dataset root containing data/ and meta/.')
    parser.add_argument('--limit-episodes', type=int, default=None, help='Only inspect the first N episode indices.')
    parser.add_argument('--top-k', type=int, default=10, help='Print the closest K episode starts.')
    parser.add_argument('--robot-ip', default=_DEFAULT_ROBOT_IP)
    parser.add_argument('--gripper-port', default=_DEFAULT_GRIPPER_PORT)
    parser.add_argument('--gripper-backend', choices=['pika', 'das'], default=_DEFAULT_GRIPPER_BACKEND)
    parser.add_argument(
        '--position-weight-mm',
        type=float,
        default=1.0,
        help='Weight applied to Euclidean xyz distance in millimeters for the combined score.',
    )
    parser.add_argument(
        '--rotation-weight-deg',
        type=float,
        default=1.0,
        help='Weight applied to orientation angle distance in degrees for the combined score.',
    )
    parser.add_argument(
        '--gripper-weight',
        type=float,
        default=50.0,
        help='Weight applied to absolute gripper delta for the combined score.',
    )
    parser.add_argument(
        '--dump-json',
        type=Path,
        default=None,
        help='Optional JSON path for saving the current state, summary stats, and top matches.',
    )
    return parser.parse_args(argv)


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError('numpy is required to run this script. Use the project container or environment with dependencies installed.') from exc
    return np


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError('pyarrow is required to read the dataset parquet files.') from exc
    return pq


def _require_rotation():
    from lerobot.utils.rotation import Rotation

    return Rotation


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def load_episode_start_states(dataset_root: str | Path, *, limit_episodes: int | None = None) -> list[EpisodeStartState]:
    np = _require_numpy()
    pq = _require_pyarrow_parquet()

    dataset_root = _resolve_repo_path(dataset_root)
    meta_dir = dataset_root / 'meta' / 'episodes'
    meta_files = sorted(meta_dir.rglob('*.parquet'))
    if not meta_files:
        raise FileNotFoundError(f'No episode metadata parquet files found in {meta_dir}')

    episode_rows: list[tuple[int, int, int]] = []
    for meta_file in meta_files:
        table = pq.read_table(str(meta_file)).to_pydict()
        episode_indices = table['episode_index']
        chunk_indices = table['data/chunk_index']
        file_indices = table['data/file_index']
        for episode_index, chunk_index, file_index in zip(episode_indices, chunk_indices, file_indices, strict=True):
            episode_rows.append((int(episode_index), int(chunk_index), int(file_index)))

    episode_rows.sort(key=lambda item: item[0])
    if limit_episodes is not None:
        episode_rows = episode_rows[: max(limit_episodes, 0)]

    start_states: list[EpisodeStartState] = []
    for episode_index, chunk_index, file_index in episode_rows:
        data_file = dataset_root / 'data' / f'chunk-{chunk_index:03d}' / f'file-{file_index:06d}.parquet'
        table = pq.read_table(str(data_file), columns=['episode_index', 'observation.state']).to_pydict()
        for row_episode_index, state in zip(table['episode_index'], table['observation.state'], strict=True):
            if int(row_episode_index) != episode_index:
                continue
            start_states.append(EpisodeStartState(episode_index=episode_index, state=np.asarray(state, dtype=np.float64)))
            break
        else:
            raise ValueError(f'Episode {episode_index} metadata found, but no rows matched in {data_file}')

    if not start_states:
        raise ValueError(f'No episode starts resolved from {dataset_root}')
    return start_states


def get_current_robot_state(args: argparse.Namespace):
    np = _require_numpy()
    from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config
    from lerobot.robots.franka_research3.processor_franka_research3 import KeepAbsoluteEEObservation

    robot_cfg = FrankaResearch3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_backend=args.gripper_backend,
        allow_mock_gripper=False,
        urdf_path=str(_DAS_URDF),
        target_frame_name='das_gripper_ee',
        cameras={},
    )
    robot = FrankaResearch3(robot_cfg)
    state_processor = KeepAbsoluteEEObservation()
    robot.connect()
    try:
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    processed = state_processor.observation(dict(observation))
    return np.asarray(
        [
            processed['ee.x'],
            processed['ee.y'],
            processed['ee.z'],
            processed['ee.qx'],
            processed['ee.qy'],
            processed['ee.qz'],
            processed['ee.qw'],
            processed['gripper.pos'],
        ],
        dtype=np.float64,
    )


def quaternion_angle_deg(quaternion_xyzw_a, quaternion_xyzw_b) -> float:
    np = _require_numpy()
    Rotation = _require_rotation()

    relative = Rotation.from_quat(quaternion_xyzw_a).inv() * Rotation.from_quat(quaternion_xyzw_b)
    return float(np.degrees(np.linalg.norm(relative.as_rotvec())))


def compute_episode_deltas(
    current_state,
    episode_start_states: Iterable[EpisodeStartState],
    *,
    position_weight_mm: float,
    rotation_weight_deg: float,
    gripper_weight: float,
) -> list[EpisodeStartDelta]:
    np = _require_numpy()

    current_position = current_state[:3]
    current_quaternion = current_state[3:7]
    current_gripper = float(current_state[7])

    deltas: list[EpisodeStartDelta] = []
    for episode_start in episode_start_states:
        target_state = episode_start.state
        target_position = target_state[:3]
        target_quaternion = target_state[3:7]
        target_gripper = float(target_state[7])
        position_delta_m = target_position - current_position
        position_distance_m = float(np.linalg.norm(position_delta_m))
        rotation_delta_deg = quaternion_angle_deg(current_quaternion, target_quaternion)
        gripper_delta = abs(target_gripper - current_gripper)
        weighted_score = (
            position_distance_m * 1000.0 * position_weight_mm
            + rotation_delta_deg * rotation_weight_deg
            + gripper_delta * gripper_weight
        )
        deltas.append(
            EpisodeStartDelta(
                episode_index=episode_start.episode_index,
                position_delta_m=position_delta_m,
                position_distance_m=position_distance_m,
                rotation_delta_deg=rotation_delta_deg,
                gripper_delta=gripper_delta,
                weighted_score=weighted_score,
            )
        )
    deltas.sort(key=lambda item: item.weighted_score)
    return deltas


def _format_xyz_mm(values_m) -> str:
    return ', '.join(f'{value * 1000.0:.1f}' for value in values_m)


def summarize(values) -> tuple[float, float, float]:
    np = _require_numpy()

    return float(np.min(values)), float(np.median(values)), float(np.max(values))


def _to_serializable_float_list(values) -> list[float]:
    return [float(value) for value in values]


def dump_snapshot_json(
    dump_path: str | Path,
    *,
    dataset_path: Path,
    current_state,
    episodes_compared: int,
    position_distance_mm_stats: tuple[float, float, float],
    rotation_distance_deg_stats: tuple[float, float, float],
    gripper_delta_stats: tuple[float, float, float],
    top_matches: list[EpisodeStartDelta],
) -> Path:
    dump_path = _resolve_repo_path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset': str(dataset_path),
        'episodes_compared': int(episodes_compared),
        'current_state': {
            'x': float(current_state[0]),
            'y': float(current_state[1]),
            'z': float(current_state[2]),
            'qx': float(current_state[3]),
            'qy': float(current_state[4]),
            'qz': float(current_state[5]),
            'qw': float(current_state[6]),
            'gripper': float(current_state[7]),
        },
        'position_distance_mm': {
            'min': float(position_distance_mm_stats[0]),
            'median': float(position_distance_mm_stats[1]),
            'max': float(position_distance_mm_stats[2]),
        },
        'rotation_distance_deg': {
            'min': float(rotation_distance_deg_stats[0]),
            'median': float(rotation_distance_deg_stats[1]),
            'max': float(rotation_distance_deg_stats[2]),
        },
        'gripper_delta': {
            'min': float(gripper_delta_stats[0]),
            'median': float(gripper_delta_stats[1]),
            'max': float(gripper_delta_stats[2]),
        },
        'top_matches': [
            {
                'rank': int(rank),
                'episode_index': int(item.episode_index),
                'weighted_score': float(item.weighted_score),
                'position_delta_m': _to_serializable_float_list(item.position_delta_m),
                'position_distance_m': float(item.position_distance_m),
                'rotation_delta_deg': float(item.rotation_delta_deg),
                'gripper_delta': float(item.gripper_delta),
            }
            for rank, item in enumerate(top_matches, start=1)
        ],
    }
    dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dump_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np = _require_numpy()

    episode_start_states = load_episode_start_states(args.dataset, limit_episodes=args.limit_episodes)
    current_state = get_current_robot_state(args)
    deltas = compute_episode_deltas(
        current_state,
        episode_start_states,
        position_weight_mm=args.position_weight_mm,
        rotation_weight_deg=args.rotation_weight_deg,
        gripper_weight=args.gripper_weight,
    )

    print(f'[INFO] dataset={_resolve_repo_path(args.dataset)}')
    print(f'[INFO] episodes_compared={len(deltas)}')
    print(
        '[INFO] current_state='
        f"xyz=({current_state[0]:.4f}, {current_state[1]:.4f}, {current_state[2]:.4f}) "
        f"quat=({current_state[3]:.4f}, {current_state[4]:.4f}, {current_state[5]:.4f}, {current_state[6]:.4f}) "
        f"gripper={current_state[7]:.3f}"
    )

    position_distances_mm = np.asarray([item.position_distance_m * 1000.0 for item in deltas], dtype=np.float64)
    rotation_distances_deg = np.asarray([item.rotation_delta_deg for item in deltas], dtype=np.float64)
    gripper_deltas = np.asarray([item.gripper_delta for item in deltas], dtype=np.float64)
    pos_min, pos_median, pos_max = summarize(position_distances_mm)
    rot_min, rot_median, rot_max = summarize(rotation_distances_deg)
    grip_min, grip_median, grip_max = summarize(gripper_deltas)
    print(f'[INFO] position_distance_mm min/median/max = {pos_min:.1f} / {pos_median:.1f} / {pos_max:.1f}')
    print(f'[INFO] rotation_distance_deg min/median/max = {rot_min:.1f} / {rot_median:.1f} / {rot_max:.1f}')
    print(f'[INFO] gripper_delta min/median/max = {grip_min:.3f} / {grip_median:.3f} / {grip_max:.3f}')
    top_matches = deltas[: max(args.top_k, 0)]
    print(f'[INFO] top_k={min(args.top_k, len(deltas))}')

    if args.dump_json is not None:
        dump_path = dump_snapshot_json(
            args.dump_json,
            dataset_path=_resolve_repo_path(args.dataset),
            current_state=current_state,
            episodes_compared=len(deltas),
            position_distance_mm_stats=(pos_min, pos_median, pos_max),
            rotation_distance_deg_stats=(rot_min, rot_median, rot_max),
            gripper_delta_stats=(grip_min, grip_median, grip_max),
            top_matches=top_matches,
        )
        print(f'[INFO] dumped_current_state={dump_path}')

    for rank, item in enumerate(top_matches, start=1):
        print(
            f'[MATCH] rank={rank} episode={item.episode_index} '
            f'score={item.weighted_score:.2f} '
            f'pos_mm=({_format_xyz_mm(item.position_delta_m)}) '
            f'pos_norm_mm={item.position_distance_m * 1000.0:.1f} '
            f'rot_deg={item.rotation_delta_deg:.1f} '
            f'gripper_delta={item.gripper_delta:.3f}'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
