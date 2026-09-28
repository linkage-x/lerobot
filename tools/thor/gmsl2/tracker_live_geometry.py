"""Capture-time geometry of a tracker-mount capture, in pure Python.

The solves refuse a thin capture -- a station whose parked points span less
than 0.3 m, a pivot that did not roll far enough about the beam -- but they run
after Disconnect, when the rig has been taken down. 2026-09-24 lost a whole
station and a pivot's certification that way. These numbers are the same
quantities the solves gate on, computed from the tracker samples of each segment
as it is saved, so the operator can see them while still standing at the rig.

Pure Python on purpose: the gateway runs on the system interpreter without
numpy, and the recorder should not grow a dependency for a dozen 3x3 sums.
They are previews, not the solve: the station counts every saved segment (the
solve drops a segment without a dwell), and the pivot gain uses every beam-valid
sample (the solve drops lifted ones first). Definitions follow
``metrology.laser_tracker_registration.assess_geometry`` and
``metrology.tracker_pivot._direction_gains``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float, float]

STATION_MIN_EXTENT_M = 0.3
STATION_MIN_PLANARITY = 0.1
PIVOT_MIN_GAIN = 0.05


def _mean(points: Sequence[Point]) -> Point:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, sum(p[2] for p in points) / n)


def _cov(points: Sequence[Point]) -> list[list[float]]:
    m = _mean(points)
    c = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = (p[0] - m[0], p[1] - m[1], p[2] - m[2])
        for i in range(3):
            for j in range(3):
                c[i][j] += d[i] * d[j]
    n = len(points)
    return [[v / n for v in row] for row in c]


def sym3_eigvals(a: list[list[float]]) -> tuple[float, float, float]:
    """Eigenvalues of a symmetric 3x3 matrix, ascending (closed form)."""
    p1 = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2
    q = (a[0][0] + a[1][1] + a[2][2]) / 3.0
    if p1 <= 1e-30 * max(1.0, q * q):
        return tuple(sorted((a[0][0], a[1][1], a[2][2])))  # type: ignore[return-value]
    p2 = (a[0][0] - q) ** 2 + (a[1][1] - q) ** 2 + (a[2][2] - q) ** 2 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    b = [[(a[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
    det_b = (
        b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
    )
    r = max(-1.0, min(1.0, det_b / 2.0))
    phi = math.acos(r) / 3.0
    e_hi = q + 2.0 * p * math.cos(phi)
    e_lo = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    e_mid = 3.0 * q - e_hi - e_lo
    return (e_lo, e_mid, e_hi)


def median_point(points: Sequence[Point]) -> Point:
    def med(v: list[float]) -> float:
        v = sorted(v)
        k = len(v) // 2
        return v[k] if len(v) % 2 else 0.5 * (v[k - 1] + v[k])

    return (med([p[0] for p in points]), med([p[1] for p in points]), med([p[2] for p in points]))


def station_geometry(points_mm: Sequence[Point]) -> dict[str, float | int | bool | None]:
    """Extent (largest pairwise distance) and planarity (sigma_3 / sigma_1) of the
    parked points so far, against the station's refusal thresholds."""
    n = len(points_mm)
    if n < 3:
        return {"n": n, "extent_m": None, "planarity": None, "ok": False}
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            extent = max(extent, math.dist(points_mm[i], points_mm[j]))
    lo, _mid, hi = sym3_eigvals(_cov(points_mm))
    planarity = math.sqrt(max(lo, 0.0) / hi) if hi > 0.0 else 0.0
    extent_m = extent * 1e-3
    return {
        "n": n,
        "extent_m": extent_m,
        "planarity": planarity,
        "ok": extent_m >= STATION_MIN_EXTENT_M and planarity >= STATION_MIN_PLANARITY,
    }


def _solve4(m: list[list[float]], v: list[float]) -> list[float] | None:
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for c in range(4):
        piv = max(range(c, 4), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-18:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(4):
            if r != c:
                f = a[r][c] / a[c][c]
                for k in range(c, 5):
                    a[r][k] -= f * a[c][k]
    return [a[i][4] / a[i][i] for i in range(4)]


def fit_sphere(points: Sequence[Point]) -> tuple[Point, float] | None:
    """Algebraic sphere fit ``|p|^2 = 2 p.s + k``: centre and radius."""
    ata = [[0.0] * 4 for _ in range(4)]
    atb = [0.0] * 4
    for p in points:
        row = (2.0 * p[0], 2.0 * p[1], 2.0 * p[2], 1.0)
        b = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        for i in range(4):
            atb[i] += row[i] * b
            for j in range(4):
                ata[i][j] += row[i] * row[j]
    x = _solve4(ata, atb)
    if x is None:
        return None
    s = (x[0], x[1], x[2])
    r2 = x[3] + s[0] ** 2 + s[1] ** 2 + s[2] ** 2
    return (s, math.sqrt(r2)) if r2 > 0.0 else None


def pivot_geometry(points_mm: Sequence[Point], *, cell_deg: float = 1.0) -> dict[str, float | int | bool | None]:
    """How well a pivot sweep pins the socket: the weak-direction gain the E1p
    solve refuses to certify below, plus the cap's angular span."""
    n = len(points_mm)
    out: dict[str, float | int | bool | None] = {
        "n": n, "radius_mm": None, "rms_mm": None, "gain_min": None, "gain_max": None,
        "span_deg": None, "ok": False,
    }
    if n < 20:
        return out
    lo_p, _m, hi_p = sym3_eigvals(_cov(points_mm))
    fit = fit_sphere(points_mm) if hi_p > 0.0 and math.sqrt(max(lo_p, 0.0) / hi_p) > 1e-4 else None
    if fit is None:
        # Every point in one plane -- on one circle: a turn about a single axis
        # (a real cap of +/-15 deg still has planarity ~0.03). The socket is
        # free along that axis; the solve refuses it too.
        out.update({"gain_min": 0.0, "degenerate": True})
        return out
    s, r = fit
    # One unit vector per ~1 deg cell, as the solve pools, so lingering does not
    # weight the gain.
    cells: dict[tuple[int, int, int], Point] = {}
    step = math.radians(cell_deg)
    res2 = 0.0
    for p in points_mm:
        d = (p[0] - s[0], p[1] - s[1], p[2] - s[2])
        norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        res2 += (norm - r) ** 2
        u = (d[0] / norm, d[1] / norm, d[2] / norm)
        cells.setdefault((round(u[0] / step), round(u[1] / step), round(u[2] / step)), u)
    units = list(cells.values())
    if len(units) < 3:
        return out
    lo, _mid, hi = sym3_eigvals(_cov(units))
    m = _mean(units)
    mn = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2) or 1.0
    span = max(math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * m[0] + u[1] * m[1] + u[2] * m[2]) / mn))))
               for u in units)
    gain_min = math.sqrt(max(lo, 0.0))
    out.update({
        "radius_mm": r,
        "rms_mm": math.sqrt(res2 / n),
        "gain_min": gain_min,
        "gain_max": math.sqrt(max(hi, 0.0)),
        "span_deg": span,
        "cells": len(units),
        "ok": gain_min >= PIVOT_MIN_GAIN,
    })
    return out
