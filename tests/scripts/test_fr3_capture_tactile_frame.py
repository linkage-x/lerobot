#!/usr/bin/env python

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.fr3 import fr3_capture_tactile_frame


def test_capture_one_frame_passes_loaded_mask_into_idle_baseline_loader(monkeypatch, tmp_path: Path):
    mask = np.ones((50, 10), dtype=np.uint8)
    seen_masks: list[np.ndarray] = []

    monkeypatch.setattr(fr3_capture_tactile_frame, "_load_mask", lambda path: mask)

    def fake_load_baseline_side(path, side, valid_mask=None):
        del path, side
        seen_masks.append(np.asarray(valid_mask))
        return np.zeros((50, 10), dtype=np.float32)

    monkeypatch.setattr(fr3_capture_tactile_frame, "_load_baseline_side", fake_load_baseline_side)
    monkeypatch.setattr(fr3_capture_tactile_frame, "_resolve_path", lambda path: Path(path))
    monkeypatch.setattr(fr3_capture_tactile_frame, "_build_row_major_pairs", lambda valid_mask: [])
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_build_vertical_priority_pairs",
        lambda valid_mask: ([], []),
    )
    monkeypatch.setattr(fr3_capture_tactile_frame, "_build_horizontal_mirror_pairs", lambda valid_mask: [])
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_pairwise_reduce_adjacent_bytes",
        lambda record_data: np.zeros((0,), dtype=np.float32),
    )
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_scatter_row_major",
        lambda valid_values, valid_mask: np.zeros((50, 10), dtype=np.float32),
    )
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_expand_pair_values_to_dense",
        lambda pair_values, pairs, valid_mask: np.zeros((50, 10), dtype=np.float32),
    )
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_decode_direct_spatial_split_expand",
        lambda direct_frame, valid_mask: (
            np.zeros((50, 10), dtype=np.float32),
            np.zeros((50, 10), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        fr3_capture_tactile_frame,
        "_save_pngs",
        lambda *args, **kwargs: tuple(tmp_path / f"artifact_{idx}.png" for idx in range(7)),
    )

    class FakeDataBus:
        def __init__(self, *, tactile_callback, **kwargs):
            tactile_callback(bytes([0] * fr3_capture_tactile_frame._EXPECTED_VALID_COUNT))

        def stop(self):
            return None

    monkeypatch.setattr(fr3_capture_tactile_frame, "_require_sdk_databus_cls", lambda: FakeDataBus)

    args = Namespace(
        mask_path=tmp_path / "mask.json",
        baseline_path=tmp_path / "idle_baseline.json",
        baseline_side="left",
        output_dir=tmp_path / "out",
        tty_port="/dev/null",
        baudrate=921600,
        encoder_freq=5.0,
        tactile_freq=5.0,
        timeout_s=0.1,
        side_name="demo",
        scale=8,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = fr3_capture_tactile_frame.capture_one_frame(args)

    assert exit_code == 0
    assert len(seen_masks) == 2
    assert np.array_equal(seen_masks[0], mask)
    assert np.array_equal(seen_masks[1], mask)
