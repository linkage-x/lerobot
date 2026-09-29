"""The capture-time previews must agree with what the solves gate on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.thor.gmsl2 import tracker_live_geometry as tlg


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def test_eigenvalues_match_numpy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.normal(size=(3, 3))
        a = m @ m.T
        assert np.allclose(tlg.sym3_eigvals(a.tolist()), np.linalg.eigvalsh(a), atol=1e-9)
    assert tlg.sym3_eigvals([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 3.0]]) == (1.0, 2.0, 3.0)


def test_station_geometry_matches_assess_geometry():
    """``metrology.laser_tracker_registration.assess_geometry``: extent is the
    largest pairwise distance, planarity sigma_3 / sigma_1."""
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=[150.0, 90.0, 30.0], size=(19, 3))
    g = tlg.station_geometry([tuple(p) for p in pts])
    sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    ext = max(np.linalg.norm(a - b) for a in pts for b in pts)
    assert g["planarity"] == pytest.approx(sv[2] / sv[0], rel=1e-9)
    assert g["extent_m"] == pytest.approx(ext * 1e-3, rel=1e-12)
    assert g["ok"] == (g["extent_m"] >= 0.3 and g["planarity"] >= 0.1)
    assert tlg.station_geometry([(0.0, 0.0, 0.0)] * 2)["ok"] is False


def test_pivot_gain_separates_a_tilt_only_sweep_from_a_rolled_one():
    """09-24's pivot: gain 0.024 from a narrow cap. Rolling about the beam
    lifts the weak direction; tilting about one axis leaves it near zero."""
    socket = np.array([600.0, 870.0, -93.0])
    v = np.array([0.0, 60.0, 72.0])  # ~94 mm, the real radius

    def sweep(roll, tilt):
        ts = np.linspace(0, 30, 3000)
        return [tuple(socket + _rot([1, 0, 0], roll * math.sin(2 * math.pi * t / 7)) @ _rot(
            [0, 1, 0], tilt * math.sin(2 * math.pi * t / 4.3)) @ v) for t in ts]

    narrow = tlg.pivot_geometry(sweep(3.0, 25.0))
    wide = tlg.pivot_geometry(sweep(55.0, 22.0))
    assert narrow["radius_mm"] == pytest.approx(np.linalg.norm(v), abs=1e-6)
    assert narrow["gain_min"] < 0.02 and not narrow["ok"]
    assert wide["gain_min"] > tlg.PIVOT_MIN_GAIN and wide["ok"]
    assert wide["span_deg"] > narrow["span_deg"]
    one_axis = tlg.pivot_geometry(sweep(0.0, 25.0))
    assert one_axis["degenerate"] and one_axis["gain_min"] == 0.0 and not one_axis["ok"]
    assert tlg.pivot_geometry([(0.0, 0.0, 0.0)] * 5)["ok"] is False


def test_the_recorder_emits_a_segment_line_only_for_tracker_mount_captures(capsys):
    from tools.thor.gmsl2 import thor_record

    class FakeTracker:
        def __init__(self):
            self.asked = []

        def segment_points(self, *, last_rows):
            self.asked.append(last_rows)
            return [(100.0, 200.0, 300.0), (100.2, 200.0, 300.0), (99.8, 200.0, 300.0)]

    tr = FakeTracker()
    rec = {"t_start_wall_s": 10.0, "t_end_wall_s": 14.0}
    thor_record._emit_tracker_segment_geometry(tr, rec, {"protocol": "smr_parked_pose_dwell"}, 7)
    thor_record._emit_tracker_segment_geometry(tr, rec, {"protocol": "task"}, 8)
    thor_record._emit_tracker_segment_geometry(tr, rec, None, 9)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("LT_SEGMENT ")]
    assert tr.asked == [4000]
    assert len(lines) == 1
    import json

    seg = json.loads(lines[0].removeprefix("LT_SEGMENT "))
    assert seg["episode"] == 7 and seg["kind"] == "dwell"
    assert seg["point_mm"] == [100.0, 200.0, 300.0] and seg["spread_mm"] == pytest.approx(0.2)
