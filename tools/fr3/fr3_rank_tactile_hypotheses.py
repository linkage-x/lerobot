#!/usr/bin/env python3
"""Capture multiple DAS tactile frames and rank reconstruction hypotheses.

This script samples a sequence of raw 448-byte tactile payloads from the DAS serial link,
reconstructs each frame under three hypotheses, and ranks the hypotheses against stored
tactile profile statistics plus an idle baseline frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from statistics import mean, median

import numpy as np

try:
    from tools.fr3.fr3_capture_tactile_frame import (
        _DEFAULT_BAUDRATE,
        _DEFAULT_BASELINE_PATH,
        _DEFAULT_ENCODER_FREQ,
        _DEFAULT_MASK_PATH,
        _DEFAULT_SCALE,
        _DEFAULT_TACTILE_FREQ,
        _DEFAULT_TIMEOUT_S,
        _DEFAULT_TTY_PORT,
        _COMPRESSED_SIDE_VALID_COUNT,
        _EXPECTED_VALID_COUNT,
        _build_horizontal_mirror_pairs,
        _build_row_major_pairs,
        _build_vertical_priority_pairs,
        _compute_baseline_abs_diff_stats,
        _decode_direct_spatial_split_expand,
        _expand_pair_values_to_dense,
        _load_baseline_side,
        _load_mask,
        _pairwise_reduce_adjacent_bytes,
        _require_sdk_databus_cls,
        _resolve_path,
        _save_pngs,
        _scatter_row_major,
    )
except ImportError:
    from fr3_capture_tactile_frame import (
        _DEFAULT_BAUDRATE,
        _DEFAULT_BASELINE_PATH,
        _DEFAULT_ENCODER_FREQ,
        _DEFAULT_MASK_PATH,
        _DEFAULT_SCALE,
        _DEFAULT_TACTILE_FREQ,
        _DEFAULT_TIMEOUT_S,
        _DEFAULT_TTY_PORT,
        _COMPRESSED_SIDE_VALID_COUNT,
        _EXPECTED_VALID_COUNT,
        _build_horizontal_mirror_pairs,
        _build_row_major_pairs,
        _build_vertical_priority_pairs,
        _compute_baseline_abs_diff_stats,
        _decode_direct_spatial_split_expand,
        _expand_pair_values_to_dense,
        _load_baseline_side,
        _load_mask,
        _pairwise_reduce_adjacent_bytes,
        _require_sdk_databus_cls,
        _resolve_path,
        _save_pngs,
        _scatter_row_major,
    )

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILE_STATS_PATH = _REPO_ROOT / 'docs/tactile/profile_stats.json'
_DEFAULT_OUTPUT_ANALYSIS_ROOT = _REPO_ROOT / 'outputs/tactile_sequence_analysis'
_DEFAULT_NUM_FRAMES = 12
_DEFAULT_FRAME_INTERVAL_S = 1.5
_ACTIVE_FRAME_MAX_THRESHOLD = 10.0
_ACTIVE_RELATIVE_THRESHOLD = 0.35
_COMPONENT_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Capture and rank tactile reconstruction hypotheses over multiple frames.')
    parser.add_argument('--tty-port', default=_DEFAULT_TTY_PORT)
    parser.add_argument('--baudrate', type=int, default=_DEFAULT_BAUDRATE)
    parser.add_argument('--tactile-freq', type=float, default=_DEFAULT_TACTILE_FREQ)
    parser.add_argument('--encoder-freq', type=float, default=_DEFAULT_ENCODER_FREQ)
    parser.add_argument('--timeout-s', type=float, default=_DEFAULT_TIMEOUT_S)
    parser.add_argument('--mask-path', type=Path, default=_DEFAULT_MASK_PATH)
    parser.add_argument('--idle-baseline-path', type=Path, default=_DEFAULT_BASELINE_PATH)
    parser.add_argument('--profile-stats-path', type=Path, default=_DEFAULT_PROFILE_STATS_PATH)
    parser.add_argument('--baseline-path', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--sequence-name', default='sequence_rank')
    parser.add_argument('--num-frames', type=int, default=_DEFAULT_NUM_FRAMES)
    parser.add_argument('--frame-interval-s', type=float, default=_DEFAULT_FRAME_INTERVAL_S)
    parser.add_argument('--scale', type=int, default=_DEFAULT_SCALE)
    parser.add_argument('--image-baseline-side', choices=('left', 'right'), default='left')
    return parser.parse_args(argv)


def _default_output_dir() -> Path:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return (_DEFAULT_OUTPUT_ANALYSIS_ROOT / timestamp).resolve()


def _load_baseline_records(path: str | Path) -> dict:
    return json.loads(_resolve_path(path).read_text(encoding='utf-8'))


def _load_dataset_profiles(profile_stats_path: str | Path, valid_mask: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    payload = _load_baseline_records(profile_stats_path)
    profiles_payload = payload.get('profiles')
    if profiles_payload is None:
        return _build_dataset_profiles(payload, valid_mask)

    profiles: dict[str, dict[str, dict[str, float]]] = {}
    for side in ('left', 'right'):
        side_payload = profiles_payload[side]
        metrics_payload = side_payload['metrics']
        profiles[side] = {
            metric_name: {
                'mean': float(metric_stats['mean']),
                'std': float(metric_stats['std']),
            }
            for metric_name, metric_stats in metrics_payload.items()
        }
        profiles[side]['frame_count'] = {'mean': float(side_payload['frame_count']), 'std': 1.0}
    return profiles


def _valid_values(frame: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask.astype(bool)
    return np.asarray(frame[valid], dtype=np.float32)


def _anisotropy_metrics(frame: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask.astype(bool)
    vertical: list[float] = []
    horizontal: list[float] = []
    rows, cols = frame.shape
    for r in range(rows - 1):
        for c in range(cols):
            if valid[r, c] and valid[r + 1, c]:
                vertical.append(abs(float(frame[r, c]) - float(frame[r + 1, c])))
    for r in range(rows):
        for c in range(cols - 1):
            if valid[r, c] and valid[r, c + 1]:
                horizontal.append(abs(float(frame[r, c]) - float(frame[r, c + 1])))
    vertical_mean = float(mean(vertical)) if vertical else 0.0
    horizontal_mean = float(mean(horizontal)) if horizontal else 0.0
    ratio = float(vertical_mean / horizontal_mean) if horizontal_mean > 0.0 else 0.0
    return {
        'vertical_abs_mean': vertical_mean,
        'horizontal_abs_mean': horizontal_mean,
        'anisotropy_ratio': ratio,
    }


def _hot_threshold(frame: np.ndarray, valid_mask: np.ndarray) -> float:
    valid_values = _valid_values(frame, valid_mask)
    max_value = float(valid_values.max()) if valid_values.size else 0.0
    return max(_ACTIVE_FRAME_MAX_THRESHOLD, max_value * _ACTIVE_RELATIVE_THRESHOLD)


def _connectivity_metrics(frame: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask.astype(bool)
    threshold = _hot_threshold(frame, valid_mask)
    hot = (frame >= threshold) & valid
    hot_count = int(hot.sum())
    if hot_count == 0:
        return {
            'hot_threshold': threshold,
            'hot_count': 0.0,
            'hot_fraction': 0.0,
            'component_count': 0.0,
            'largest_component_fraction': 0.0,
        }

    rows, cols = frame.shape
    seen = np.zeros_like(hot, dtype=bool)
    component_sizes: list[int] = []
    for r in range(rows):
        for c in range(cols):
            if not hot[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in _COMPONENT_NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if seen[nr, nc] or not hot[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            component_sizes.append(size)

    largest = max(component_sizes) if component_sizes else 0
    valid_count = int(valid.sum())
    return {
        'hot_threshold': threshold,
        'hot_count': float(hot_count),
        'hot_fraction': float(hot_count / valid_count) if valid_count > 0 else 0.0,
        'component_count': float(len(component_sizes)),
        'largest_component_fraction': float(largest / hot_count) if hot_count > 0 else 0.0,
    }


def _frame_metrics(frame: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid_values = _valid_values(frame, valid_mask)
    metrics = {
        'valid_max': float(valid_values.max()) if valid_values.size else 0.0,
        'valid_mean': float(valid_values.mean()) if valid_values.size else 0.0,
        'valid_sum': float(valid_values.sum()) if valid_values.size else 0.0,
        'nonzero_count': float((valid_values > 0).sum()) if valid_values.size else 0.0,
    }
    metrics.update(_anisotropy_metrics(frame, valid_mask))
    metrics.update(_connectivity_metrics(frame, valid_mask))
    return metrics


def _is_active_record(frame: np.ndarray, valid_mask: np.ndarray) -> bool:
    valid_values = _valid_values(frame, valid_mask)
    return bool(valid_values.size) and float(valid_values.max()) >= _ACTIVE_FRAME_MAX_THRESHOLD


def _mean_and_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': 0.0, 'std': 1.0}
    mu = float(mean(values))
    variance = float(sum((x - mu) ** 2 for x in values) / len(values))
    std = variance ** 0.5
    return {'mean': mu, 'std': std if std > 1e-6 else 1.0}


def _build_dataset_profiles(baseline_payload: dict, valid_mask: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
    rows, cols = valid_mask.shape
    profiles: dict[str, dict[str, dict[str, float]]] = {}
    for side in ('left', 'right'):
        metrics_list: list[dict[str, float]] = []
        for record in baseline_payload['data']:
            values = np.asarray(record['tactiles'][side], dtype=np.float32).reshape(rows, cols)
            if not _is_active_record(values, valid_mask):
                continue
            metrics_list.append(_frame_metrics(values, valid_mask))
        profiles[side] = {
            metric_name: _mean_and_std([item[metric_name] for item in metrics_list])
            for metric_name in (
                'anisotropy_ratio',
                'component_count',
                'largest_component_fraction',
                'hot_fraction',
            )
        }
        profiles[side]['frame_count'] = {'mean': float(len(metrics_list)), 'std': 1.0}
    return profiles


def _score_against_side(metrics: dict[str, float], profile: dict[str, dict[str, float]]) -> dict[str, float]:
    anisotropy = abs(metrics['anisotropy_ratio'] - profile['anisotropy_ratio']['mean']) / profile['anisotropy_ratio']['std']
    connectivity = 0.0
    for key in ('component_count', 'largest_component_fraction', 'hot_fraction'):
        connectivity += abs(metrics[key] - profile[key]['mean']) / profile[key]['std']
    connectivity /= 3.0
    return {
        'anisotropy_distance': float(anisotropy),
        'connectivity_distance': float(connectivity),
        'combined_distance': float((anisotropy + connectivity) / 2.0),
    }


def _rank_frame_hypothesis(
    frame: np.ndarray,
    valid_mask: np.ndarray,
    left_baseline: np.ndarray,
    right_baseline: np.ndarray,
    dataset_profiles: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    metrics = _frame_metrics(frame, valid_mask)
    left_diff = _compute_baseline_abs_diff_stats(frame, left_baseline, valid_mask)
    right_diff = _compute_baseline_abs_diff_stats(frame, right_baseline, valid_mask)
    left_score = _score_against_side(metrics, dataset_profiles['left'])
    right_score = _score_against_side(metrics, dataset_profiles['right'])
    preferred_side = 'left' if left_score['combined_distance'] <= right_score['combined_distance'] else 'right'
    side_score = left_score if preferred_side == 'left' else right_score

    baseline_identical = bool(np.array_equal(left_baseline, right_baseline))
    baseline_distance = min(left_diff['mean'], right_diff['mean'])
    total_score = side_score['combined_distance']
    return {
        'metrics': metrics,
        'baseline_diff_left': left_diff,
        'baseline_diff_right': right_diff,
        'dataset_distance_left': left_score,
        'dataset_distance_right': right_score,
        'preferred_side': preferred_side,
        'assignment': preferred_side,
        'baseline_identical': baseline_identical,
        'baseline_distance': float(baseline_distance),
        'total_score': float(total_score),
    }


def _rank_known_side_frame(
    frame: np.ndarray,
    valid_mask: np.ndarray,
    baseline: np.ndarray,
    dataset_profile: dict[str, dict[str, float]],
) -> dict[str, object]:
    metrics = _frame_metrics(frame, valid_mask)
    baseline_diff = _compute_baseline_abs_diff_stats(frame, baseline, valid_mask)
    dataset_distance = _score_against_side(metrics, dataset_profile)
    return {
        'metrics': metrics,
        'baseline_diff': baseline_diff,
        'dataset_distance': dataset_distance,
        'total_score': float(dataset_distance['combined_distance']),
    }


def _rank_bilateral_hypothesis(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    valid_mask: np.ndarray,
    left_baseline: np.ndarray,
    right_baseline: np.ndarray,
    dataset_profiles: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    left_score = _rank_known_side_frame(left_frame, valid_mask, left_baseline, dataset_profiles['left'])
    right_score = _rank_known_side_frame(right_frame, valid_mask, right_baseline, dataset_profiles['right'])
    metrics = {
        'anisotropy_ratio': float((left_score['metrics']['anisotropy_ratio'] + right_score['metrics']['anisotropy_ratio']) / 2.0),
        'component_count': float((left_score['metrics']['component_count'] + right_score['metrics']['component_count']) / 2.0),
        'hot_fraction': float((left_score['metrics']['hot_fraction'] + right_score['metrics']['hot_fraction']) / 2.0),
    }
    return {
        'assignment': 'fixed_left_right',
        'metrics': metrics,
        'left': left_score,
        'right': right_score,
        'baseline_identical': bool(np.array_equal(left_baseline, right_baseline)),
        'baseline_distance': float((left_score['baseline_diff']['mean'] + right_score['baseline_diff']['mean']) / 2.0),
        'total_score': float((left_score['total_score'] + right_score['total_score']) / 2.0),
    }


def _capture_sequence(args: argparse.Namespace) -> list[bytes]:
    DataBus = _require_sdk_databus_cls()
    payload_lock = threading.Lock()
    latest_payload: bytes | None = None
    latest_seq = -1
    frames_event = threading.Event()
    state = {'seq': -1}

    def tactile_callback(record_data: bytes) -> None:
        nonlocal latest_payload, latest_seq
        payload = bytes(record_data)
        if len(payload) != _EXPECTED_VALID_COUNT:
            return
        with payload_lock:
            state['seq'] += 1
            latest_seq = state['seq']
            latest_payload = payload
            frames_event.set()

    def encoder_callback(record_data: bytes) -> None:
        del record_data

    db = DataBus(
        tty_port=args.tty_port,
        baudrate=args.baudrate,
        encoder_freq=args.encoder_freq,
        tactile_freq=args.tactile_freq,
        tactile_callback=tactile_callback,
        encoder_callback=encoder_callback,
    )

    captured: list[bytes] = []
    seen_seq = -1
    try:
        if not frames_event.wait(timeout=args.timeout_s):
            raise TimeoutError(f'No tactile frame received within {args.timeout_s:.1f}s on {args.tty_port}.')
        next_capture_time = time.perf_counter()
        while len(captured) < args.num_frames:
            now = time.perf_counter()
            if now < next_capture_time:
                time.sleep(min(0.01, next_capture_time - now))
                continue
            with payload_lock:
                payload = latest_payload
                seq = latest_seq
            if payload is not None and seq != seen_seq:
                captured.append(payload)
                seen_seq = seq
                next_capture_time = time.perf_counter() + args.frame_interval_s
            else:
                time.sleep(0.01)
    finally:
        db.stop()
    return captured


def _save_frame_outputs(
    output_dir: Path,
    frame_index: int,
    hypothesis_name: str,
    frame: np.ndarray,
    valid_mask: np.ndarray,
    baseline_for_images: np.ndarray,
    scale: int,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    frame_dir = output_dir / f'frame_{frame_index:03d}'
    side_name = f'{hypothesis_name}'
    return _save_pngs(
        frame,
        valid_mask=valid_mask,
        baseline_50x10=baseline_for_images,
        output_dir=frame_dir,
        side_name=side_name,
        scale=scale,
    )


def _summarize_hypothesis(frame_results: list[dict[str, object]]) -> dict[str, object]:
    totals = [float(item['score']['total_score']) for item in frame_results]
    anisotropy = [float(item['score']['metrics']['anisotropy_ratio']) for item in frame_results]
    components = [float(item['score']['metrics']['component_count']) for item in frame_results]
    hot_fraction = [float(item['score']['metrics']['hot_fraction']) for item in frame_results]
    assignments = [str(item['score'].get('assignment', item['score'].get('preferred_side', 'unknown'))) for item in frame_results]
    assignment_counts = {assignment: assignments.count(assignment) for assignment in sorted(set(assignments))}
    return {
        'frames': len(frame_results),
        'total_score_mean': float(mean(totals)) if totals else 0.0,
        'total_score_median': float(median(totals)) if totals else 0.0,
        'anisotropy_ratio_mean': float(mean(anisotropy)) if anisotropy else 0.0,
        'component_count_mean': float(mean(components)) if components else 0.0,
        'hot_fraction_mean': float(mean(hot_fraction)) if hot_fraction else 0.0,
        'assignment_counts': assignment_counts,
    }


def run(args: argparse.Namespace) -> int:
    if args.num_frames <= 0:
        raise ValueError('--num-frames must be positive')
    if args.frame_interval_s <= 0:
        raise ValueError('--frame-interval-s must be positive')

    output_dir = _resolve_path(args.output_dir) if args.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_path is not None:
        args.idle_baseline_path = args.baseline_path
        args.profile_stats_path = args.baseline_path

    valid_mask = _load_mask(args.mask_path)
    left_baseline = _load_baseline_side(args.idle_baseline_path, 'left', valid_mask)
    right_baseline = _load_baseline_side(args.idle_baseline_path, 'right', valid_mask)
    baseline_for_images = left_baseline if args.image_baseline_side == 'left' else right_baseline
    row_major_pairs = _build_row_major_pairs(valid_mask)
    vertical_pairs, vertical_leftovers = _build_vertical_priority_pairs(valid_mask)
    horizontal_mirror_pairs = _build_horizontal_mirror_pairs(valid_mask)
    dataset_profiles = _load_dataset_profiles(args.profile_stats_path, valid_mask)

    sequence = _capture_sequence(args)

    all_results: dict[str, list[dict[str, object]]] = {
        '448_direct': [],
        '224_flat_pair_expand': [],
        '224_vertical_pair_expand': [],
        '448_direct_spatial_split_expand': [],
    }

    for frame_index, payload in enumerate(sequence):
        frame_dir = output_dir / f'frame_{frame_index:03d}'
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / f'{args.sequence_name}_raw_payload.bin').write_bytes(payload)

        valid_values = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
        reduced_224 = _pairwise_reduce_adjacent_bytes(payload)
        reconstructed = {
            '448_direct': _scatter_row_major(valid_values, valid_mask),
            '224_flat_pair_expand': _expand_pair_values_to_dense(reduced_224, row_major_pairs, valid_mask),
            '224_vertical_pair_expand': _expand_pair_values_to_dense(reduced_224, vertical_pairs, valid_mask),
        }
        bilateral_left, bilateral_right = _decode_direct_spatial_split_expand(reconstructed['448_direct'], valid_mask)

        for hypothesis_name, frame in reconstructed.items():
            paths = _save_frame_outputs(
                output_dir=output_dir,
                frame_index=frame_index,
                hypothesis_name=hypothesis_name,
                frame=frame,
                valid_mask=valid_mask,
                baseline_for_images=baseline_for_images,
                scale=args.scale,
            )
            score = _rank_frame_hypothesis(frame, valid_mask, left_baseline, right_baseline, dataset_profiles)
            all_results[hypothesis_name].append(
                {
                    'frame_index': frame_index,
                    'saved_raw_png': str(paths[0]),
                    'saved_preview_png': str(paths[1]),
                    'saved_baseline_png': str(paths[2]),
                    'saved_baseline_preview_png': str(paths[3]),
                    'saved_mask_png': str(paths[4]),
                    'saved_value_heatmap_png': str(paths[5]),
                    'saved_baseline_diff_png': str(paths[6]),
                    'score': score,
                }
            )

        bilateral_left_paths = _save_frame_outputs(
            output_dir=output_dir,
            frame_index=frame_index,
            hypothesis_name='448_direct_spatial_split_expand_left',
            frame=bilateral_left,
            valid_mask=valid_mask,
            baseline_for_images=left_baseline,
            scale=args.scale,
        )
        bilateral_right_paths = _save_frame_outputs(
            output_dir=output_dir,
            frame_index=frame_index,
            hypothesis_name='448_direct_spatial_split_expand_right',
            frame=bilateral_right,
            valid_mask=valid_mask,
            baseline_for_images=right_baseline,
            scale=args.scale,
        )
        bilateral_score = _rank_bilateral_hypothesis(
            bilateral_left,
            bilateral_right,
            valid_mask,
            left_baseline,
            right_baseline,
            dataset_profiles,
        )
        all_results['448_direct_spatial_split_expand'].append(
            {
                'frame_index': frame_index,
                'left': {
                    'saved_raw_png': str(bilateral_left_paths[0]),
                    'saved_preview_png': str(bilateral_left_paths[1]),
                    'saved_baseline_png': str(bilateral_left_paths[2]),
                    'saved_baseline_preview_png': str(bilateral_left_paths[3]),
                    'saved_mask_png': str(bilateral_left_paths[4]),
                    'saved_value_heatmap_png': str(bilateral_left_paths[5]),
                    'saved_baseline_diff_png': str(bilateral_left_paths[6]),
                },
                'right': {
                    'saved_raw_png': str(bilateral_right_paths[0]),
                    'saved_preview_png': str(bilateral_right_paths[1]),
                    'saved_baseline_png': str(bilateral_right_paths[2]),
                    'saved_baseline_preview_png': str(bilateral_right_paths[3]),
                    'saved_mask_png': str(bilateral_right_paths[4]),
                    'saved_value_heatmap_png': str(bilateral_right_paths[5]),
                    'saved_baseline_diff_png': str(bilateral_right_paths[6]),
                },
                'score': bilateral_score,
            }
        )

    ranking = [
        {
            'hypothesis': name,
            **_summarize_hypothesis(results),
        }
        for name, results in all_results.items()
    ]
    ranking.sort(key=lambda item: item['total_score_mean'])

    summary = {
        'sequence_name': args.sequence_name,
        'output_dir': str(output_dir),
        'num_frames': len(sequence),
        'frame_interval_s': args.frame_interval_s,
        'payload_len_bytes': _EXPECTED_VALID_COUNT,
        'pairing': {
            'row_major_pair_count': len(row_major_pairs),
            'vertical_pair_count': len(vertical_pairs),
            'vertical_leftovers': [list(item) for item in vertical_leftovers],
            'horizontal_mirror_pair_count': len(horizontal_mirror_pairs),
        },
        'baseline_identical': bool(np.array_equal(left_baseline, right_baseline)),
        'dataset_profiles': dataset_profiles,
        'ranking': ranking,
        'frame_results': all_results,
    }

    summary_json_path = output_dir / f'{args.sequence_name}_summary.json'
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        f'# Tactile Hypothesis Ranking: {args.sequence_name}',
        '',
        f'- output_dir: `{output_dir}`',
        f'- num_frames: `{len(sequence)}`',
        f'- frame_interval_s: `{args.frame_interval_s}`',
        f'- baseline_identical: `{summary["baseline_identical"]}`',
        '',
        '## Ranking',
        '',
    ]
    for index, item in enumerate(ranking, start=1):
        lines.extend(
            [
                f'{index}. `{item["hypothesis"]}`',
                f'   - total_score_mean: `{item["total_score_mean"]:.4f}`',
                f'   - total_score_median: `{item["total_score_median"]:.4f}`',
                f'   - anisotropy_ratio_mean: `{item["anisotropy_ratio_mean"]:.4f}`',
                f'   - component_count_mean: `{item["component_count_mean"]:.4f}`',
                f'   - hot_fraction_mean: `{item["hot_fraction_mean"]:.4f}`',
                f'   - assignment_counts: `{item["assignment_counts"]}`',
            ]
        )
    summary_md_path = output_dir / f'{args.sequence_name}_summary.md'
    summary_md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'[INFO] output_dir={output_dir}')
    print(f'[INFO] summary_json={summary_json_path}')
    print(f'[INFO] summary_md={summary_md_path}')
    print(f'[INFO] num_frames={len(sequence)}')
    print(f'[INFO] baseline_identical={summary["baseline_identical"]}')
    for index, item in enumerate(ranking, start=1):
        print(
            f'[RANK {index}] hypothesis={item["hypothesis"]} '
            f'total_score_mean={item["total_score_mean"]:.4f} '
            f'anisotropy_ratio_mean={item["anisotropy_ratio_mean"]:.4f} '
            f'component_count_mean={item["component_count_mean"]:.4f} '
            f'hot_fraction_mean={item["hot_fraction_mean"]:.4f} '
            f'assignment_counts={item["assignment_counts"]}'
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
