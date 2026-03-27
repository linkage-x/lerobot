#!/usr/bin/env python3
"""Capture one DAS tactile frame and compare multiple reconstruction hypotheses.

This script uses the installed gen_con_sdk_python_release SDK inside the infer container.
It captures one raw 448-byte tactile payload and emits comparison visualizations for
multiple single-side reconstruction hypotheses from the same blob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MASK_PATH = _REPO_ROOT / 'docs/tactile/tactile_valid_mask_50x10.json'
_DEFAULT_BASELINE_PATH = _REPO_ROOT / 'docs/tactile/idle_baseline.json'
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / 'outputs/tactile_capture'
_DEFAULT_TTY_PORT = '/dev/ttyUSB0'
_DEFAULT_BAUDRATE = 921600
_DEFAULT_TACTILE_FREQ = 5.0
_DEFAULT_ENCODER_FREQ = 5.0
_DEFAULT_TIMEOUT_S = 8.0
_DEFAULT_SCALE = 24
_INVALID_VALUE = 255.0
_EXPECTED_VALID_COUNT = 448
_HYPOTHESIS_224_COUNT = 224
_COMPRESSED_SIDE_VALID_COUNT = _EXPECTED_VALID_COUNT // 2
_IMAGE_SHAPE = (50, 10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Capture one tactile frame via SDK and save multi-hypothesis PNGs.')
    parser.add_argument('--tty-port', default=_DEFAULT_TTY_PORT)
    parser.add_argument('--baudrate', type=int, default=_DEFAULT_BAUDRATE)
    parser.add_argument('--tactile-freq', type=float, default=_DEFAULT_TACTILE_FREQ)
    parser.add_argument(
        '--encoder-freq',
        type=float,
        default=_DEFAULT_ENCODER_FREQ,
        help='Keep encoder polling enabled because some devices only return tactile when both loops run.',
    )
    parser.add_argument('--timeout-s', type=float, default=_DEFAULT_TIMEOUT_S)
    parser.add_argument('--mask-path', type=Path, default=_DEFAULT_MASK_PATH)
    parser.add_argument('--baseline-path', type=Path, default=_DEFAULT_BASELINE_PATH)
    parser.add_argument('--baseline-side', choices=('left', 'right'), default='left')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--side-name', default='single_side', help='Only used for output file names.')
    parser.add_argument('--scale', type=int, default=_DEFAULT_SCALE, help='Nearest-neighbor preview upscaling factor.')
    return parser.parse_args(argv)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _load_mask(mask_path: str | Path) -> np.ndarray:
    path = _resolve_path(mask_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    mask = np.asarray(payload['mask'], dtype=np.uint8)
    if tuple(mask.shape) != _IMAGE_SHAPE:
        raise ValueError(f'mask must have shape {_IMAGE_SHAPE}, got {tuple(mask.shape)} from {path}')
    valid_count = int(mask.astype(bool).sum())
    if valid_count != _EXPECTED_VALID_COUNT:
        raise ValueError(f'mask valid count must be {_EXPECTED_VALID_COUNT}, got {valid_count} from {path}')
    return mask


def _scatter_row_major(valid_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = np.asarray(valid_values, dtype=np.float32).reshape(-1)
    flat_mask = valid_mask.astype(bool).reshape(-1)
    if valid_values.size != int(flat_mask.sum()):
        raise ValueError(f'expected {int(flat_mask.sum())} valid values, got {valid_values.size}')
    dense = np.full(flat_mask.shape, _INVALID_VALUE, dtype=np.float32)
    dense[flat_mask] = valid_values
    return dense.reshape(valid_mask.shape)


def _build_row_major_pairs(valid_mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    positions = [tuple(int(v) for v in item) for item in np.argwhere(valid_mask.astype(bool))]
    if len(positions) != _EXPECTED_VALID_COUNT:
        raise ValueError(f'expected {_EXPECTED_VALID_COUNT} valid positions, got {len(positions)}')
    return [(positions[i], positions[i + 1]) for i in range(0, len(positions), 2)]


def _build_vertical_priority_pairs(
    valid_mask: np.ndarray,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    valid = valid_mask.astype(bool)
    used = np.zeros_like(valid, dtype=bool)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    rows, cols = valid.shape
    for r in range(rows - 1):
        for c in range(cols):
            if valid[r, c] and valid[r + 1, c] and not used[r, c] and not used[r + 1, c]:
                a = (int(r), int(c))
                b = (int(r + 1), int(c))
                pairs.append((a, b))
                used[r, c] = True
                used[r + 1, c] = True

    leftovers = [(int(r), int(c)) for r in range(rows) for c in range(cols) if valid[r, c] and not used[r, c]]
    if len(leftovers) % 2 != 0:
        raise ValueError(f'expected an even number of vertical leftovers, got {len(leftovers)}')
    for i in range(0, len(leftovers), 2):
        pairs.append((leftovers[i], leftovers[i + 1]))

    if len(pairs) != _HYPOTHESIS_224_COUNT:
        raise ValueError(f'expected {_HYPOTHESIS_224_COUNT} vertical-priority pairs, got {len(pairs)}')
    return pairs, leftovers


def _build_horizontal_mirror_pairs(valid_mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    valid = valid_mask.astype(bool)
    rows, cols = valid.shape
    if cols % 2 != 0:
        raise ValueError(f'expected an even tactile width, got {cols}')

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols // 2):
            mirror_c = cols - 1 - c
            if not valid[r, c]:
                if valid[r, mirror_c]:
                    raise ValueError('valid mask must be horizontally symmetric for mirror expansion')
                continue
            if not valid[r, mirror_c]:
                raise ValueError('valid mask must be horizontally symmetric for mirror expansion')
            pairs.append(((int(r), int(c)), (int(r), int(mirror_c))))

    if len(pairs) != _COMPRESSED_SIDE_VALID_COUNT:
        raise ValueError(f'expected {_COMPRESSED_SIDE_VALID_COUNT} horizontal-mirror pairs, got {len(pairs)}')
    return pairs


def _decode_direct_spatial_split_expand(
    direct_frame: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pairs = _build_horizontal_mirror_pairs(valid_mask)
    left = np.full(valid_mask.shape, _INVALID_VALUE, dtype=np.float32)
    right = np.full(valid_mask.shape, _INVALID_VALUE, dtype=np.float32)
    for left_pos, right_pos in pairs:
        left_value = float(direct_frame[left_pos])
        right_value = float(direct_frame[right_pos])
        left[left_pos] = left_value
        left[right_pos] = left_value
        right[left_pos] = right_value
        right[right_pos] = right_value
    return left, right


def _pairwise_reduce_adjacent_bytes(record_data: bytes) -> np.ndarray:
    values = np.frombuffer(record_data, dtype=np.uint8).astype(np.float32)
    if values.size != _EXPECTED_VALID_COUNT:
        raise ValueError(f'expected {_EXPECTED_VALID_COUNT} payload bytes, got {values.size}')
    return values.reshape(-1, 2).mean(axis=1)


def _expand_pair_values_to_dense(
    pair_values: np.ndarray,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    valid_mask: np.ndarray,
) -> np.ndarray:
    pair_values = np.asarray(pair_values, dtype=np.float32).reshape(-1)
    if pair_values.size != len(pairs):
        raise ValueError(f'expected {len(pairs)} pair values, got {pair_values.size}')

    dense = np.full(valid_mask.shape, _INVALID_VALUE, dtype=np.float32)
    for value, (a, b) in zip(pair_values, pairs, strict=True):
        dense[a] = value
        dense[b] = value
    return dense


def _load_baseline_side(baseline_path: str | Path, side: str, valid_mask: np.ndarray | None = None) -> np.ndarray:
    path = _resolve_path(baseline_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('encoding') == 'mask_fill':
        if valid_mask is None:
            raise ValueError(f'valid_mask is required to decode mask_fill tactile baseline from {path}')
        try:
            side_payload = payload['sides'][side]
            valid_value = float(side_payload['valid_value'])
            invalid_value = float(side_payload['invalid_value'])
        except Exception as exc:
            raise ValueError(f'Could not load tactile baseline side={side!r} from {path}') from exc
        baseline = np.full(valid_mask.shape, invalid_value, dtype=np.float32)
        baseline[valid_mask.astype(bool)] = valid_value
        return baseline
    try:
        values = payload['data'][0]['tactiles'][side]
    except Exception as exc:
        raise ValueError(f'Could not load tactile baseline side={side!r} from {path}') from exc
    baseline = np.asarray(values, dtype=np.float32)
    if baseline.size != _IMAGE_SHAPE[0] * _IMAGE_SHAPE[1]:
        raise ValueError(
            f'baseline side={side!r} must have {_IMAGE_SHAPE[0] * _IMAGE_SHAPE[1]} values, got {baseline.size} from {path}'
        )
    return baseline.reshape(_IMAGE_SHAPE)


def _make_mask_only_image(valid_mask: np.ndarray) -> np.ndarray:
    return np.where(valid_mask.astype(bool), 255, 0).astype(np.uint8)


def _make_value_heatmap(frame_50x10: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    valid = valid_mask.astype(bool)
    normalized = np.zeros_like(frame_50x10, dtype=np.uint8)
    if np.any(valid):
        valid_values = frame_50x10[valid]
        max_value = float(valid_values.max())
        scale = 255.0 / max_value if max_value > 0.0 else 0.0
        if scale > 0.0:
            normalized[valid] = np.clip(valid_values * scale, 0.0, 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    return heatmap


def _make_baseline_diff_heatmap(frame_50x10: np.ndarray, baseline_50x10: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    valid = valid_mask.astype(bool)
    diff = np.zeros_like(frame_50x10, dtype=np.float32)
    diff[valid] = np.abs(frame_50x10[valid] - baseline_50x10[valid])
    if np.any(valid):
        valid_diff = diff[valid]
        max_diff = float(valid_diff.max())
        scale = 255.0 / max_diff if max_diff > 0.0 else 0.0
        normalized = np.zeros_like(diff, dtype=np.uint8)
        if scale > 0.0:
            normalized[valid] = np.clip(diff[valid] * scale, 0.0, 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(diff, dtype=np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    return heatmap


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError('OpenCV (cv2) is required to save PNG outputs.') from exc
    return cv2


def _require_sdk_databus_cls():
    try:
        from lerobot.robots.franka_research3.backends import _resolve_das_databus_cls
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError('Could not import the FR3 DAS SDK resolver. Run this in the project container.') from exc
    return _resolve_das_databus_cls(None)


def _save_pngs(
    frame_50x10: np.ndarray,
    *,
    valid_mask: np.ndarray,
    baseline_50x10: np.ndarray,
    output_dir: Path,
    side_name: str,
    scale: int,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    cv2 = _require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_uint8 = np.clip(frame_50x10, 0.0, 255.0).astype(np.uint8)
    raw_png_path = output_dir / f'{side_name}_50x10.png'
    preview_png_path = output_dir / f'{side_name}_preview_x{scale}.png'
    baseline_png_path = output_dir / f'{side_name}_baseline_50x10.png'
    baseline_preview_png_path = output_dir / f'{side_name}_baseline_preview_x{scale}.png'
    mask_png_path = output_dir / f'{side_name}_mask_only.png'
    value_heatmap_png_path = output_dir / f'{side_name}_value_heatmap.png'
    diff_png_path = output_dir / f'{side_name}_baseline_diff_heatmap.png'

    if not cv2.imwrite(str(raw_png_path), raw_uint8):
        raise RuntimeError(f'Failed to write {raw_png_path}')

    preview = cv2.resize(raw_uint8, (raw_uint8.shape[1] * scale, raw_uint8.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(preview_png_path), preview):
        raise RuntimeError(f'Failed to write {preview_png_path}')

    baseline_uint8 = np.clip(baseline_50x10, 0.0, 255.0).astype(np.uint8)
    if not cv2.imwrite(str(baseline_png_path), baseline_uint8):
        raise RuntimeError(f'Failed to write {baseline_png_path}')

    baseline_preview = cv2.resize(
        baseline_uint8,
        (baseline_uint8.shape[1] * scale, baseline_uint8.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    if not cv2.imwrite(str(baseline_preview_png_path), baseline_preview):
        raise RuntimeError(f'Failed to write {baseline_preview_png_path}')

    mask_only = _make_mask_only_image(valid_mask)
    if not cv2.imwrite(str(mask_png_path), mask_only):
        raise RuntimeError(f'Failed to write {mask_png_path}')

    value_heatmap = _make_value_heatmap(frame_50x10, valid_mask)
    if not cv2.imwrite(str(value_heatmap_png_path), value_heatmap):
        raise RuntimeError(f'Failed to write {value_heatmap_png_path}')

    diff_heatmap = _make_baseline_diff_heatmap(frame_50x10, baseline_50x10, valid_mask)
    if not cv2.imwrite(str(diff_png_path), diff_heatmap):
        raise RuntimeError(f'Failed to write {diff_png_path}')

    return raw_png_path, preview_png_path, baseline_png_path, baseline_preview_png_path, mask_png_path, value_heatmap_png_path, diff_png_path


def _default_output_dir() -> Path:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def _compute_baseline_abs_diff_stats(frame_50x10: np.ndarray, baseline_50x10: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask.astype(bool)
    baseline_abs_diff = np.zeros_like(frame_50x10, dtype=np.float32)
    baseline_abs_diff[valid] = np.abs(frame_50x10[valid] - baseline_50x10[valid])
    valid_diff = baseline_abs_diff[valid]
    return {
        'min': float(valid_diff.min() if valid_diff.size else 0.0),
        'max': float(valid_diff.max() if valid_diff.size else 0.0),
        'mean': float(valid_diff.mean() if valid_diff.size else 0.0),
    }


def capture_one_frame(args: argparse.Namespace) -> int:
    mask = _load_mask(args.mask_path)
    left_baseline_50x10 = _load_baseline_side(args.baseline_path, 'left', mask)
    right_baseline_50x10 = _load_baseline_side(args.baseline_path, 'right', mask)
    baseline_50x10 = left_baseline_50x10 if args.baseline_side == 'left' else right_baseline_50x10
    output_dir = (_resolve_path(args.output_dir) if args.output_dir is not None else _default_output_dir())
    row_major_pairs = _build_row_major_pairs(mask)
    vertical_priority_pairs, vertical_leftovers = _build_vertical_priority_pairs(mask)
    horizontal_mirror_pairs = _build_horizontal_mirror_pairs(mask)

    DataBus = _require_sdk_databus_cls()
    frame_event = threading.Event()
    payload_holder: dict[str, bytes] = {}

    def tactile_callback(record_data: bytes) -> None:
        if frame_event.is_set():
            return
        payload_holder['record_data'] = bytes(record_data)
        frame_event.set()

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

    try:
        if not frame_event.wait(timeout=args.timeout_s):
            raise TimeoutError(
                f'No tactile frame received within {args.timeout_s:.1f}s on {args.tty_port}. '
                'Try increasing --timeout-s or verifying the device state.'
            )
    finally:
        db.stop()

    record_data = payload_holder['record_data']
    if len(record_data) != _EXPECTED_VALID_COUNT:
        raise ValueError(
            f'This script currently expects a single-side tactile payload of {_EXPECTED_VALID_COUNT} bytes, '
            f'got {len(record_data)} bytes.'
        )

    valid_values = np.frombuffer(record_data, dtype=np.uint8).astype(np.float32)
    reduced_224_values = _pairwise_reduce_adjacent_bytes(record_data)

    frame_50x10_direct = _scatter_row_major(valid_values, mask)
    frame_50x10_flat224 = _expand_pair_values_to_dense(reduced_224_values, row_major_pairs, mask)
    frame_50x10_vertical224 = _expand_pair_values_to_dense(reduced_224_values, vertical_priority_pairs, mask)
    frame_50x10_bilateral_left, frame_50x10_bilateral_right = _decode_direct_spatial_split_expand(frame_50x10_direct, mask)

    direct_paths = _save_pngs(
        frame_50x10_direct,
        valid_mask=mask,
        baseline_50x10=baseline_50x10,
        output_dir=output_dir,
        side_name=f'{args.side_name}_hyp_448_direct',
        scale=args.scale,
    )
    flat_paths = _save_pngs(
        frame_50x10_flat224,
        valid_mask=mask,
        baseline_50x10=baseline_50x10,
        output_dir=output_dir,
        side_name=f'{args.side_name}_hyp_224_flat_pair_expand',
        scale=args.scale,
    )
    vertical_paths = _save_pngs(
        frame_50x10_vertical224,
        valid_mask=mask,
        baseline_50x10=baseline_50x10,
        output_dir=output_dir,
        side_name=f'{args.side_name}_hyp_224_vertical_pair_expand',
        scale=args.scale,
    )
    bilateral_left_paths = _save_pngs(
        frame_50x10_bilateral_left,
        valid_mask=mask,
        baseline_50x10=left_baseline_50x10,
        output_dir=output_dir,
        side_name=f'{args.side_name}_hyp_448_direct_spatial_split_left',
        scale=args.scale,
    )
    bilateral_right_paths = _save_pngs(
        frame_50x10_bilateral_right,
        valid_mask=mask,
        baseline_50x10=right_baseline_50x10,
        output_dir=output_dir,
        side_name=f'{args.side_name}_hyp_448_direct_spatial_split_right',
        scale=args.scale,
    )

    raw_payload_path = output_dir / f'{args.side_name}_raw_payload.bin'
    raw_payload_path.write_bytes(record_data)

    metadata_path = output_dir / f'{args.side_name}_hypothesis_metadata.json'
    metadata = {
        'payload_len_bytes': len(record_data),
        'payload_reduce_for_224': 'mean_of_adjacent_byte_pairs',
        'compressed_side_valid_count': _COMPRESSED_SIDE_VALID_COUNT,
        'row_major_pair_count': len(row_major_pairs),
        'vertical_priority_pair_count': len(vertical_priority_pairs),
        'vertical_priority_leftovers': [list(item) for item in vertical_leftovers],
        'horizontal_mirror_pair_count': len(horizontal_mirror_pairs),
        'hypotheses': {
            '448_direct': {
                'saved_raw_png': str(direct_paths[0]),
                'saved_preview_png': str(direct_paths[1]),
                'saved_baseline_png': str(direct_paths[2]),
                'saved_baseline_preview_png': str(direct_paths[3]),
                'saved_mask_png': str(direct_paths[4]),
                'saved_value_heatmap_png': str(direct_paths[5]),
                'saved_baseline_diff_png': str(direct_paths[6]),
                'baseline_abs_diff_stats': _compute_baseline_abs_diff_stats(frame_50x10_direct, baseline_50x10, mask),
            },
            '224_flat_pair_expand': {
                'saved_raw_png': str(flat_paths[0]),
                'saved_preview_png': str(flat_paths[1]),
                'saved_baseline_png': str(flat_paths[2]),
                'saved_baseline_preview_png': str(flat_paths[3]),
                'saved_mask_png': str(flat_paths[4]),
                'saved_value_heatmap_png': str(flat_paths[5]),
                'saved_baseline_diff_png': str(flat_paths[6]),
                'baseline_abs_diff_stats': _compute_baseline_abs_diff_stats(frame_50x10_flat224, baseline_50x10, mask),
            },
            '224_vertical_pair_expand': {
                'saved_raw_png': str(vertical_paths[0]),
                'saved_preview_png': str(vertical_paths[1]),
                'saved_baseline_png': str(vertical_paths[2]),
                'saved_baseline_preview_png': str(vertical_paths[3]),
                'saved_mask_png': str(vertical_paths[4]),
                'saved_value_heatmap_png': str(vertical_paths[5]),
                'saved_baseline_diff_png': str(vertical_paths[6]),
                'baseline_abs_diff_stats': _compute_baseline_abs_diff_stats(frame_50x10_vertical224, baseline_50x10, mask),
            },
            '448_direct_spatial_split_expand': {
                'left': {
                    'saved_raw_png': str(bilateral_left_paths[0]),
                    'saved_preview_png': str(bilateral_left_paths[1]),
                    'saved_baseline_png': str(bilateral_left_paths[2]),
                    'saved_baseline_preview_png': str(bilateral_left_paths[3]),
                    'saved_mask_png': str(bilateral_left_paths[4]),
                    'saved_value_heatmap_png': str(bilateral_left_paths[5]),
                    'saved_baseline_diff_png': str(bilateral_left_paths[6]),
                    'baseline_abs_diff_stats': _compute_baseline_abs_diff_stats(frame_50x10_bilateral_left, left_baseline_50x10, mask),
                },
                'right': {
                    'saved_raw_png': str(bilateral_right_paths[0]),
                    'saved_preview_png': str(bilateral_right_paths[1]),
                    'saved_baseline_png': str(bilateral_right_paths[2]),
                    'saved_baseline_preview_png': str(bilateral_right_paths[3]),
                    'saved_mask_png': str(bilateral_right_paths[4]),
                    'saved_value_heatmap_png': str(bilateral_right_paths[5]),
                    'saved_baseline_diff_png': str(bilateral_right_paths[6]),
                    'baseline_abs_diff_stats': _compute_baseline_abs_diff_stats(frame_50x10_bilateral_right, right_baseline_50x10, mask),
                },
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    direct_diff_stats = metadata['hypotheses']['448_direct']['baseline_abs_diff_stats']
    flat_diff_stats = metadata['hypotheses']['224_flat_pair_expand']['baseline_abs_diff_stats']
    vertical_diff_stats = metadata['hypotheses']['224_vertical_pair_expand']['baseline_abs_diff_stats']
    bilateral_left_diff_stats = metadata['hypotheses']['448_direct_spatial_split_expand']['left']['baseline_abs_diff_stats']
    bilateral_right_diff_stats = metadata['hypotheses']['448_direct_spatial_split_expand']['right']['baseline_abs_diff_stats']

    print(f'[INFO] tty_port={args.tty_port}')
    print(f'[INFO] payload_len={len(record_data)}')
    print(f'[INFO] baseline_side={args.baseline_side}')
    print(f'[INFO] raw_payload_path={raw_payload_path}')
    print(f'[INFO] metadata_path={metadata_path}')
    print(f'[INFO] hyp_448_direct_preview={direct_paths[1]}')
    print(f'[INFO] hyp_448_direct_value_heatmap={direct_paths[5]}')
    print(f'[INFO] hyp_224_flat_pair_expand_preview={flat_paths[1]}')
    print(f'[INFO] hyp_224_vertical_pair_expand_preview={vertical_paths[1]}')
    print(f'[INFO] hyp_448_direct_spatial_split_left_preview={bilateral_left_paths[1]}')
    print(f'[INFO] hyp_448_direct_spatial_split_right_preview={bilateral_right_paths[1]}')
    print(f'[INFO] vertical_priority_leftovers={vertical_leftovers}')
    print(
        '[INFO] hyp_448_direct_baseline_abs_diff='
        f"min={direct_diff_stats['min']:.1f} max={direct_diff_stats['max']:.1f} mean={direct_diff_stats['mean']:.2f}"
    )
    print(
        '[INFO] hyp_224_flat_pair_expand_baseline_abs_diff='
        f"min={flat_diff_stats['min']:.1f} max={flat_diff_stats['max']:.1f} mean={flat_diff_stats['mean']:.2f}"
    )
    print(
        '[INFO] hyp_224_vertical_pair_expand_baseline_abs_diff='
        f"min={vertical_diff_stats['min']:.1f} max={vertical_diff_stats['max']:.1f} mean={vertical_diff_stats['mean']:.2f}"
    )
    print(
        '[INFO] hyp_448_direct_spatial_split_left_baseline_abs_diff='
        f"min={bilateral_left_diff_stats['min']:.1f} max={bilateral_left_diff_stats['max']:.1f} mean={bilateral_left_diff_stats['mean']:.2f}"
    )
    print(
        '[INFO] hyp_448_direct_spatial_split_right_baseline_abs_diff='
        f"min={bilateral_right_diff_stats['min']:.1f} max={bilateral_right_diff_stats['max']:.1f} mean={bilateral_right_diff_stats['mean']:.2f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return capture_one_frame(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
