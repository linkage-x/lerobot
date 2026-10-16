# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pivot calibration with the FR3's own FK: where a point fixed to the tool sits in the EE frame.

The arm holds a tool whose point ``s`` (EE frame) stays on a fixed point ``c`` (base frame) while the
arm turns about it -- the pivot insert's socket seated on a ball on the table. Every FK pose gives
``R_i s + t_i = c``, linear in ``(s, c)``.

Use: the camera trajectory's TCP is the socket centre of pivot insert v2; the FR3 drives
``corenetic_gripper_ee`` (the vendor's closed-fingertip point). ``s`` is the offset between the two, which
the 10-12 end-to-end replay saw as a constant ~(−0.6, −2.9, −10.4) mm in the TCP frame.

Pure numpy, no robot imports.
"""

# ruff: noqa: N803, N806  (matrix names: R, T, A follow the maths)
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PivotFit:
    s_ee_m: np.ndarray  # pivot point in the EE frame
    c_base_m: np.ndarray  # pivot point in the base frame
    s_sigma_m: (
        np.ndarray
    )  # 1-sigma of s, from time-block jackknife (dwells and slow drift are not independent)
    c_sigma_m: np.ndarray  # 1-sigma of c, same jackknife
    residual_rms_m: float
    residual_p95_m: float
    n_used: int
    n_total: int
    span_deg: float  # largest rotation between any two used orientations (a 0..30° sweep reads 30°)
    weak_singular: float  # smallest singular value of the orientation spread; ~0 when s is not observable

    def as_dict(self) -> dict:
        return {
            "s_ee_mm": np.round(self.s_ee_m * 1e3, 3).tolist(),
            "s_sigma_mm": np.round(self.s_sigma_m * 1e3, 3).tolist(),
            "c_base_mm": np.round(self.c_base_m * 1e3, 3).tolist(),
            "c_sigma_mm": np.round(self.c_sigma_m * 1e3, 3).tolist(),
            "residual_rms_mm": round(self.residual_rms_m * 1e3, 4),
            "residual_p95_mm": round(self.residual_p95_m * 1e3, 4),
            "n_used": self.n_used,
            "n_total": self.n_total,
            "span_deg": round(self.span_deg, 2),
            "weak_singular": round(self.weak_singular, 4),
        }


def _solve(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(R)
    A = np.zeros((3 * n, 6))
    A[:, :3] = R.reshape(3 * n, 3)
    A[:, 3:] = -np.tile(np.eye(3), (n, 1))
    x, *_ = np.linalg.lstsq(A, -t.reshape(-1), rcond=None)
    return x[:3], x[3:]


def _orientation_bins(R: np.ndarray, bin_deg: float) -> np.ndarray:
    """One representative sample per orientation cell, so a long dwell does not outvote the sweep."""
    z = R[:, :, 2]
    x = R[:, :, 0]
    key = np.round(np.c_[z, x] / math.radians(bin_deg)).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


def _span_deg(R: np.ndarray) -> tuple[float, float]:
    # span: the largest angle between any two samples (the whole sweep, not its half about the mean)
    tr_min = min(float(np.einsum("aij,bij->ab", R[i : i + 512], R).min()) for i in range(0, len(R), 512))
    span = float(np.degrees(np.arccos(np.clip((tr_min - 1) / 2, -1, 1))))
    # mean rotation by SVD of the summed matrices
    U, _, Vt = np.linalg.svd(R.sum(axis=0))
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    # observability of s: the spread of R_i around their mean (rank 3 needs turns about two axes)
    D = (R - Rm).reshape(-1, 3)
    weak = float(np.linalg.svd(D, compute_uv=False)[-1] / math.sqrt(len(R)))
    return span, weak


def fit_pivot(
    R: np.ndarray,
    t: np.ndarray,
    *,
    bin_deg: float = 1.0,
    reject_mm: float = 0.5,
    reject_k: float = 3.0,
    blocks: int = 10,
) -> PivotFit:
    """Fit ``(s, c)`` from EE rotations ``R`` (n,3,3) and positions ``t`` (n,3), metres.

    Samples off the sphere by more than ``max(reject_mm, reject_k * median residual)`` (the insert lifted
    off the ball) are dropped, refitting until the kept set settles. ``s_sigma`` is a jackknife over
    ``blocks`` consecutive time blocks.
    """
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 3)
    bins = keep = _orientation_bins(R, bin_deg)
    s, c = _solve(R[keep], t[keep])
    for _ in range(10):
        r = np.linalg.norm(np.einsum("nij,j->ni", R, s) + t - c, axis=1)
        good = r <= max(reject_mm * 1e-3, reject_k * np.median(r[keep]))
        new = np.intersect1d(bins, np.nonzero(good)[0])
        if len(new) < 6:
            raise ValueError(f"only {len(new)} usable orientations; sweep the tool through more angles")
        if np.array_equal(new, keep):
            break
        keep = new
        s, c = _solve(R[keep], t[keep])
    r = np.linalg.norm(np.einsum("nij,j->ni", R[keep], s) + t[keep] - c, axis=1)
    # time-block jackknife on the kept samples
    edges = np.linspace(0, len(keep), blocks + 1).astype(int)
    loo = []
    for b in range(blocks):
        sub = np.r_[keep[: edges[b]], keep[edges[b + 1] :]]
        if len(sub) >= 6:
            loo.append(np.concatenate(_solve(R[sub], t[sub])))
    loo = np.asarray(loo)
    sigma = (
        np.sqrt((len(loo) - 1) / len(loo) * ((loo - loo.mean(0)) ** 2).sum(0))
        if len(loo) > 1
        else np.full(6, np.nan)
    )
    span, weak = _span_deg(R[keep])
    return PivotFit(
        s_ee_m=s,
        c_base_m=c,
        s_sigma_m=sigma[:3],
        c_sigma_m=sigma[3:],
        residual_rms_m=float(np.sqrt(np.mean(r**2))),
        residual_p95_m=float(np.percentile(r, 95)),
        n_used=int(len(keep)),
        n_total=int(len(R)),
        span_deg=span,
        weak_singular=weak,
    )


@dataclass
class JointPivotFit:
    """One ``s`` from several runs: each run turns the tool about the ball from its own heading, which
    together pins the components of ``s`` a single ±20° sweep leaves loose (along the tool axis, 10-14:
    3.0-6.7 mm run to run). Headings differ by a turn about the vertical, so they say nothing more about
    the component of ``s`` along the vertical (the cup axis): only tilting the cup does. The ball is one
    point for all runs, or one per run if it may have moved."""

    s_ee_m: np.ndarray
    s_sigma_m: np.ndarray  # block jackknife, blocks within each run
    c_base_m: np.ndarray  # (k, 3): one row if shared, else one per run
    # 1-sigma of c_base_m, same shape: the larger of the block jackknife and, with 3+ runs, the
    # leave-one-run-out jackknife (runs from different headings disagree by more than blocks do)
    c_sigma_m: np.ndarray
    shared_centre: bool
    run_rms_m: np.ndarray  # per run
    run_n_used: np.ndarray
    # Each run's own ball centre (a fit with a centre per run, whatever ``shared_centre``), minus their
    # mean: a run off by mm had the ball moved -- with a shared centre it bends s instead.
    run_centre_offset_m: np.ndarray
    s_leave_one_run_out_m: np.ndarray  # (runs, 3); empty for fewer than 3 runs
    residual_rms_m: float
    residual_p95_m: float
    n_used: int
    n_total: int
    span_deg: float
    weak_singular: float

    def as_dict(self) -> dict:
        return {
            "s_ee_mm": np.round(self.s_ee_m * 1e3, 3).tolist(),
            "s_sigma_mm": np.round(self.s_sigma_m * 1e3, 3).tolist(),
            "c_base_mm": np.round(self.c_base_m * 1e3, 3).tolist(),
            "c_sigma_mm": np.round(self.c_sigma_m * 1e3, 3).tolist(),
            "shared_centre": self.shared_centre,
            "run_rms_mm": np.round(self.run_rms_m * 1e3, 4).tolist(),
            "run_n_used": self.run_n_used.tolist(),
            "run_centre_offset_mm": np.round(self.run_centre_offset_m * 1e3, 3).tolist(),
            "s_leave_one_run_out_mm": np.round(self.s_leave_one_run_out_m * 1e3, 3).tolist(),
            "residual_rms_mm": round(self.residual_rms_m * 1e3, 4),
            "residual_p95_mm": round(self.residual_p95_m * 1e3, 4),
            "n_used": self.n_used,
            "n_total": self.n_total,
            "span_deg": round(self.span_deg, 2),
            "weak_singular": round(self.weak_singular, 4),
        }


def _solve_joint(Rs, ts, keeps, shared: bool) -> tuple[np.ndarray, np.ndarray]:
    k = len(Rs)
    nc = 1 if shared else k
    rows, rhs = [], []
    for j, (R, t, keep) in enumerate(zip(Rs, ts, keeps, strict=True)):
        n = len(keep)
        A = np.zeros((3 * n, 3 + 3 * nc))
        A[:, :3] = R[keep].reshape(3 * n, 3)
        col = 3 if shared else 3 + 3 * j
        A[:, col : col + 3] = -np.tile(np.eye(3), (n, 1))
        rows.append(A)
        rhs.append(-t[keep].reshape(-1))
    x, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return x[:3], x[3:].reshape(nc, 3)


def fit_pivot_joint(
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    *,
    shared_centre: bool = True,
    bin_deg: float = 1.0,
    reject_mm: float = 0.5,
    reject_k: float = 3.0,
    per_run_median: bool = False,
    blocks_per_run: int = 5,
) -> JointPivotFit:
    """``fit_pivot`` over several runs at once: one ``s``, and one ``c`` (``shared_centre``) or one per run.

    Orientation binning and the off-the-sphere rejection are per run, as in ``fit_pivot``: a sample farther
    than ``max(reject_mm, reject_k * median residual)`` is dropped, refitting until the kept sets settle.
    The median is over all runs, or each run's own with ``per_run_median`` -- for runs whose noise differs
    (camera labels: how many cameras saw the tool changes from one ball position to the next).
    """
    Rs = [np.asarray(R, dtype=np.float64).reshape(-1, 3, 3) for R in Rs]
    ts = [np.asarray(t, dtype=np.float64).reshape(-1, 3) for t in ts]
    if len(Rs) < 2:
        raise ValueError("a joint fit needs at least two runs")
    bins = [_orientation_bins(R, bin_deg) for R in Rs]
    keeps = list(bins)
    s, C = _solve_joint(Rs, ts, keeps, shared_centre)

    def centre(j):
        return C[0] if shared_centre else C[j]

    for _ in range(10):
        res = [np.linalg.norm(np.einsum("nij,j->ni", R, s) + t - centre(j), axis=1) for j, (R, t) in enumerate(zip(Rs, ts, strict=True))]
        if per_run_median:
            meds = [np.median(r[k]) for r, k in zip(res, keeps, strict=True)]
        else:
            meds = [np.median(np.concatenate([r[k] for r, k in zip(res, keeps, strict=True)]))] * len(Rs)
        new = [
            np.intersect1d(b, np.nonzero(r <= max(reject_mm * 1e-3, reject_k * med))[0])
            for r, b, med in zip(res, bins, meds, strict=True)
        ]
        short = [j for j, k in enumerate(new) if len(k) < 6]
        if short:
            raise ValueError(f"run(s) {short} keep fewer than 6 orientations; sweep them through more angles")
        if all(np.array_equal(a, b) for a, b in zip(new, keeps, strict=True)):
            break
        keeps = new
        s, C = _solve_joint(Rs, ts, keeps, shared_centre)
    res = [np.linalg.norm(np.einsum("nij,j->ni", R[k], s) + t[k] - centre(j), axis=1) for j, (R, t, k) in enumerate(zip(Rs, ts, keeps, strict=True))]
    r_all = np.concatenate(res)
    # block jackknife: drop one time block of one run at a time
    def jackknife(x):
        x = np.asarray(x)
        return np.sqrt((len(x) - 1) / len(x) * ((x - x.mean(0)) ** 2).sum(0)) if len(x) > 1 else np.full(x.shape[1:], np.nan)

    loo, loo_c = [], []
    for j, k in enumerate(keeps):
        edges = np.linspace(0, len(k), blocks_per_run + 1).astype(int)
        for b in range(blocks_per_run):
            sub = [kk if i != j else np.r_[k[: edges[b]], k[edges[b + 1] :]] for i, kk in enumerate(keeps)]
            if len(sub[j]) >= 6:
                s_b, C_b = _solve_joint(Rs, ts, sub, shared_centre)
                loo.append(s_b)
                loo_c.append(C_b)
    sigma = jackknife(loo)
    c_sigma = jackknife(loo_c)
    own = C if not shared_centre else _solve_joint(Rs, ts, keeps, False)[1]
    run_loo = np.zeros((0, 3))
    if len(Rs) >= 3:
        drop = [[i for i in range(len(Rs)) if i != j] for j in range(len(Rs))]
        solved = [_solve_joint([Rs[i] for i in d], [ts[i] for i in d], [keeps[i] for i in d], shared_centre) for d in drop]
        run_loo = np.asarray([x[0] for x in solved])
        if shared_centre:  # with a centre per run, dropping a run drops its centre
            c_sigma = np.maximum(c_sigma, jackknife([x[1] for x in solved]))
    span, weak = _span_deg(np.concatenate([R[k] for R, k in zip(Rs, keeps, strict=True)]))
    return JointPivotFit(
        s_ee_m=s,
        s_sigma_m=sigma,
        c_base_m=C,
        c_sigma_m=c_sigma,
        shared_centre=shared_centre,
        run_rms_m=np.asarray([float(np.sqrt(np.mean(r**2))) for r in res]),
        run_n_used=np.asarray([len(k) for k in keeps]),
        run_centre_offset_m=own - own.mean(0),
        s_leave_one_run_out_m=run_loo,
        residual_rms_m=float(np.sqrt(np.mean(r_all**2))),
        residual_p95_m=float(np.percentile(r_all, 95)),
        n_used=int(sum(len(k) for k in keeps)),
        n_total=int(sum(len(R) for R in Rs)),
        span_deg=span,
        weak_singular=weak,
    )
