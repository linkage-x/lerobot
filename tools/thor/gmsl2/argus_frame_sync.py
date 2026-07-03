"""Per-frame Argus timestamp alignment for Thor GMSL2 recordings.

This module is intentionally independent from the capture backend.  A
production Libargus recorder should write one sidecar row per encoded frame,
then call this module to decide which frames form a synchronized episode.
"""

from __future__ import annotations

import bisect
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


SIDECAR_BASENAME = "argus_frame_metadata.csv"
DEFAULT_SOF_TOLERANCE_NS = 1_000_000  # 1 ms


@dataclass(frozen=True)
class ArgusFrameMetadata:
    """Metadata for one encoded frame from one camera."""

    camera: str
    encoded_frame_index: int
    local_frame_number: int
    sensor_timestamp_ns: int
    sof_tsc_ns: int
    eof_tsc_ns: int = 0
    internal_frame_count: int = 0


@dataclass(frozen=True)
class FrameMatch:
    """One camera frame matched to one reference SOF timestamp."""

    camera: str
    reference_index: int
    reference_sof_tsc_ns: int
    encoded_frame_index: int
    local_frame_number: int
    sof_tsc_ns: int
    delta_ns: int


@dataclass
class CameraAlignment:
    camera: str
    matches: list[FrameMatch] = field(default_factory=list)
    max_abs_delta_ns: int = 0
    missing_reference_indices: list[int] = field(default_factory=list)
    out_of_tolerance: list[FrameMatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_reference_indices and not self.out_of_tolerance


@dataclass
class EpisodeAlignment:
    reference_camera: str
    tolerance_ns: int
    reference_frame_count: int
    cameras: dict[str, CameraAlignment]
    failures: list[str] = field(default_factory=list)
    accepted_reference_indices: list[int] = field(default_factory=list)
    dropped_reference_indices: list[int] = field(default_factory=list)
    drop_reasons: dict[int, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def frame_count_by_camera(self) -> dict[str, int]:
        return {name: len(camera.matches) for name, camera in self.cameras.items()}


@dataclass(frozen=True)
class CameraFrameWindow:
    """Contiguous encoded-frame window to keep for one camera."""

    camera: str
    start_frame_index: int
    stop_frame_index: int
    frame_count: int


def frame_metadata_sidecar_path(episode_dir: Path, camera: str) -> Path:
    return episode_dir / f"{camera}.{SIDECAR_BASENAME}"


def write_frame_metadata_csv(path: Path, rows: Iterable[ArgusFrameMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "camera",
        "encoded_frame_index",
        "local_frame_number",
        "sensor_timestamp_ns",
        "sof_tsc_ns",
        "eof_tsc_ns",
        "internal_frame_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_frame_metadata_csv(path: Path, *, camera: str | None = None) -> list[ArgusFrameMetadata]:
    rows: list[ArgusFrameMetadata] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row_camera = str(raw.get("camera") or camera or path.name.split(".", 1)[0])
            rows.append(
                ArgusFrameMetadata(
                    camera=row_camera,
                    encoded_frame_index=int(raw["encoded_frame_index"]),
                    local_frame_number=int(raw["local_frame_number"]),
                    sensor_timestamp_ns=int(raw.get("sensor_timestamp_ns") or 0),
                    sof_tsc_ns=int(raw["sof_tsc_ns"]),
                    eof_tsc_ns=int(raw.get("eof_tsc_ns") or 0),
                    internal_frame_count=int(raw.get("internal_frame_count") or 0),
                )
            )
    return sorted(rows, key=lambda row: (row.sof_tsc_ns, row.encoded_frame_index))


def load_episode_sidecars(episode_dir: Path, cameras: Iterable[str]) -> dict[str, list[ArgusFrameMetadata]]:
    loaded: dict[str, list[ArgusFrameMetadata]] = {}
    for camera in cameras:
        path = frame_metadata_sidecar_path(episode_dir, camera)
        loaded[camera] = read_frame_metadata_csv(path, camera=camera)
    return loaded


def _reference_rows(
    rows: list[ArgusFrameMetadata],
    *,
    start_sof_tsc_ns: int | None,
    stop_sof_tsc_ns: int | None,
) -> list[ArgusFrameMetadata]:
    selected = rows
    if start_sof_tsc_ns is not None:
        selected = [row for row in selected if row.sof_tsc_ns >= int(start_sof_tsc_ns)]
    if stop_sof_tsc_ns is not None:
        selected = [row for row in selected if row.sof_tsc_ns < int(stop_sof_tsc_ns)]
    return selected


def _nearest_after_previous(
    rows: list[ArgusFrameMetadata],
    sof_values: list[int],
    target_sof_tsc_ns: int,
    previous_row_index: int,
) -> tuple[int, ArgusFrameMetadata] | None:
    """Return nearest row whose index is greater than the previous match."""

    start = previous_row_index + 1
    if start >= len(rows):
        return None

    insert_at = bisect.bisect_left(sof_values, target_sof_tsc_ns, lo=start)
    candidate_indices: list[int] = []
    if insert_at < len(rows):
        candidate_indices.append(insert_at)
    if insert_at - 1 >= start:
        candidate_indices.append(insert_at - 1)
    if not candidate_indices:
        return None
    best_index = min(
        candidate_indices,
        key=lambda idx: (abs(rows[idx].sof_tsc_ns - target_sof_tsc_ns), idx),
    )
    return best_index, rows[best_index]


def align_episode_frames(
    frames_by_camera: dict[str, list[ArgusFrameMetadata]],
    *,
    reference_camera: str | None = None,
    tolerance_ns: int = DEFAULT_SOF_TOLERANCE_NS,
    start_sof_tsc_ns: int | None = None,
    stop_sof_tsc_ns: int | None = None,
) -> EpisodeAlignment:
    """Align camera frames to a reference camera by nearest SOF TSC.

    The returned match sequence is strictly monotonic per camera: one camera
    frame can only be used once, and every next match must appear after the
    previous matched row in that camera's metadata sidecar.
    """

    if not frames_by_camera:
        return EpisodeAlignment(
            reference_camera="",
            tolerance_ns=int(tolerance_ns),
            reference_frame_count=0,
            cameras={},
            failures=["no camera metadata sidecars"],
        )

    if reference_camera is None:
        reference_camera = sorted(frames_by_camera)[0]
    if reference_camera not in frames_by_camera:
        return EpisodeAlignment(
            reference_camera=reference_camera,
            tolerance_ns=int(tolerance_ns),
            reference_frame_count=0,
            cameras={},
            failures=[f"reference camera {reference_camera!r} not present"],
        )

    sorted_by_camera = {
        camera: sorted(rows, key=lambda row: (row.sof_tsc_ns, row.encoded_frame_index))
        for camera, rows in frames_by_camera.items()
    }
    reference = _reference_rows(
        sorted_by_camera[reference_camera],
        start_sof_tsc_ns=start_sof_tsc_ns,
        stop_sof_tsc_ns=stop_sof_tsc_ns,
    )
    failures: list[str] = []
    if not reference:
        failures.append(f"reference camera {reference_camera} has no frames in episode window")

    alignments = {
        camera: CameraAlignment(camera=camera)
        for camera in sorted(sorted_by_camera)
    }
    last_indices = {camera: -1 for camera in sorted_by_camera}
    sof_values_by_camera = {
        camera: [row.sof_tsc_ns for row in rows]
        for camera, rows in sorted_by_camera.items()
    }

    reference_row_indices = {
        id(row): idx for idx, row in enumerate(sorted_by_camera[reference_camera])
    }

    raw_drop_reasons_by_index: dict[int, list[str]] = {}

    for ref_index, ref_row in enumerate(reference):
        target_sof = ref_row.sof_tsc_ns
        tentative: dict[str, tuple[int, FrameMatch]] = {}
        drop_reasons: list[str] = []
        for camera, rows in sorted_by_camera.items():
            if camera == reference_camera:
                row_index = reference_row_indices[id(ref_row)]
                row = ref_row
            else:
                nearest = _nearest_after_previous(
                    rows,
                    sof_values_by_camera[camera],
                    target_sof,
                    last_indices[camera],
                )
                if nearest is None:
                    drop_reasons.append(f"{camera}:missing")
                    continue
                row_index, row = nearest

            delta_ns = row.sof_tsc_ns - target_sof
            match = FrameMatch(
                camera=camera,
                reference_index=ref_index,
                reference_sof_tsc_ns=target_sof,
                encoded_frame_index=row.encoded_frame_index,
                local_frame_number=row.local_frame_number,
                sof_tsc_ns=row.sof_tsc_ns,
                delta_ns=delta_ns,
            )
            if camera != reference_camera and abs(delta_ns) > int(tolerance_ns):
                drop_reasons.append(f"{camera}:delta_ns={delta_ns}")
                continue
            tentative[camera] = (row_index, match)

        if drop_reasons:
            raw_drop_reasons_by_index[ref_index] = drop_reasons
            for reason in drop_reasons:
                camera = reason.split(":", 1)[0]
                if camera in alignments:
                    if reason.endswith(":missing"):
                        alignments[camera].missing_reference_indices.append(ref_index)
                    else:
                        match = tentative.get(camera, (None, None))[1]
                        if match is not None:
                            alignments[camera].out_of_tolerance.append(match)
            continue

        for camera, (row_index, match) in tentative.items():
            alignment = alignments[camera]
            last_indices[camera] = row_index
            alignment.matches.append(match)
            alignment.max_abs_delta_ns = max(alignment.max_abs_delta_ns, abs(match.delta_ns))

    accepted_reference_indices = [
        match.reference_index
        for match in alignments[reference_camera].matches
    ]
    accepted_reference_index_set = set(accepted_reference_indices)
    dropped_reference_indices = [
        idx for idx in range(len(reference)) if idx not in accepted_reference_index_set
    ]
    drop_reasons_by_index: dict[int, list[str]] = {
        idx: list(raw_drop_reasons_by_index.get(idx, []))
        for idx in dropped_reference_indices
    }
    for camera, alignment in alignments.items():
        for idx in alignment.missing_reference_indices:
            drop_reasons_by_index.setdefault(idx, []).append(f"{camera}:missing")
        for match in alignment.out_of_tolerance:
            drop_reasons_by_index.setdefault(match.reference_index, []).append(
                f"{camera}:delta_ns={match.delta_ns}"
            )

    if not accepted_reference_indices:
        failures.append("no synchronized frame set within tolerance")
    else:
        first = min(accepted_reference_indices)
        last = max(accepted_reference_indices)
        interior_drops = [
            idx for idx in dropped_reference_indices
            if first < idx < last
        ]
        if interior_drops:
            failures.append(
                f"dropped {len(interior_drops)} reference frames inside synchronized window"
            )

    frame_counts = {camera: len(alignment.matches) for camera, alignment in alignments.items()}
    if len(set(frame_counts.values())) > 1:
        failures.append(f"matched frame count mismatch: {frame_counts}")

    return EpisodeAlignment(
        reference_camera=reference_camera,
        tolerance_ns=int(tolerance_ns),
        reference_frame_count=len(reference),
        cameras=alignments,
        failures=failures,
        accepted_reference_indices=accepted_reference_indices,
        dropped_reference_indices=dropped_reference_indices,
        drop_reasons=drop_reasons_by_index,
    )


def camera_frame_windows(alignment: EpisodeAlignment) -> dict[str, CameraFrameWindow]:
    """Return the contiguous encoded-frame window selected for each camera.

    ``align_episode_frames`` rejects interior reference drops, so the accepted
    matches should form one contiguous encoded-frame range in every camera. This
    helper makes that contract explicit before any video file is materialized.
    """

    if not alignment.ok:
        raise ValueError("cannot compute frame windows for a failed alignment")
    expected_count: int | None = None
    windows: dict[str, CameraFrameWindow] = {}
    for camera, camera_alignment in alignment.cameras.items():
        indices = [match.encoded_frame_index for match in camera_alignment.matches]
        if not indices:
            raise ValueError(f"{camera} has no accepted matches")
        frame_count = len(indices)
        if expected_count is None:
            expected_count = frame_count
        elif frame_count != expected_count:
            raise ValueError(
                f"{camera} accepted frame count {frame_count} != {expected_count}"
            )
        start = indices[0]
        stop = indices[-1] + 1
        if indices != list(range(start, stop)):
            raise ValueError(f"{camera} accepted encoded frames are not contiguous: {indices}")
        windows[camera] = CameraFrameWindow(
            camera=camera,
            start_frame_index=start,
            stop_frame_index=stop,
            frame_count=frame_count,
        )
    return windows


def write_alignment_report_json(path: Path, alignment: EpisodeAlignment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        windows = {
            camera: asdict(window)
            for camera, window in camera_frame_windows(alignment).items()
        }
    except ValueError:
        windows = {}
    payload = {
        "ok": alignment.ok,
        "reference_camera": alignment.reference_camera,
        "tolerance_ns": alignment.tolerance_ns,
        "reference_frame_count": alignment.reference_frame_count,
        "accepted_reference_indices": alignment.accepted_reference_indices,
        "dropped_reference_indices": alignment.dropped_reference_indices,
        "drop_reasons": alignment.drop_reasons,
        "frame_windows": windows,
        "frame_count_by_camera": alignment.frame_count_by_camera(),
        "failures": alignment.failures,
        "cameras": {
            camera: {
                "ok": camera_alignment.ok,
                "match_count": len(camera_alignment.matches),
                "max_abs_delta_ns": camera_alignment.max_abs_delta_ns,
                "missing_reference_indices": camera_alignment.missing_reference_indices,
                "out_of_tolerance": [
                    {
                        "reference_index": match.reference_index,
                        "encoded_frame_index": match.encoded_frame_index,
                        "local_frame_number": match.local_frame_number,
                        "sof_tsc_ns": match.sof_tsc_ns,
                        "reference_sof_tsc_ns": match.reference_sof_tsc_ns,
                        "delta_ns": match.delta_ns,
                    }
                    for match in camera_alignment.out_of_tolerance
                ],
                "matches": [asdict(match) for match in camera_alignment.matches],
            }
            for camera, camera_alignment in alignment.cameras.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
