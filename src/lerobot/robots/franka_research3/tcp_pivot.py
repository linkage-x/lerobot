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
    residual_rms_m: float
    residual_p95_m: float
    n_used: int
    n_total: int
    span_deg: float  # largest rotation away from the mean orientation
    weak_singular: float  # smallest singular value of the orientation spread; ~0 when s is not observable

    def as_dict(self) -> dict:
        return {
            "s_ee_mm": np.round(self.s_ee_m * 1e3, 3).tolist(),
            "s_sigma_mm": np.round(self.s_sigma_m * 1e3, 3).tolist(),
            "c_base_mm": np.round(self.c_base_m * 1e3, 3).tolist(),
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
    # mean rotation by SVD of the summed matrices, then the angle of each sample from it
    U, _, Vt = np.linalg.svd(R.sum(axis=0))
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    cosang = np.clip((np.einsum("nij,ij->n", R, Rm) - 1) / 2, -1, 1)
    span = float(np.degrees(np.arccos(cosang)).max())
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
    blocks: int = 10,
) -> PivotFit:
    """Fit ``(s, c)`` from EE rotations ``R`` (n,3,3) and positions ``t`` (n,3), metres.

    Samples off the sphere by more than ``reject_mm`` (the insert lifted off the ball) are dropped after
    a first fit. ``s_sigma`` is a jackknife over ``blocks`` consecutive time blocks.
    """
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 3)
    keep = _orientation_bins(R, bin_deg)
    s, c = _solve(R[keep], t[keep])
    for _ in range(3):
        r = np.linalg.norm(np.einsum("nij,j->ni", R, s) + t - c, axis=1)
        good = r <= max(reject_mm * 1e-3, 3 * np.median(r))
        keep = np.intersect1d(_orientation_bins(R, bin_deg), np.nonzero(good)[0])
        if len(keep) < 6:
            raise ValueError(f"only {len(keep)} usable orientations; sweep the tool through more angles")
        s, c = _solve(R[keep], t[keep])
    r = np.linalg.norm(np.einsum("nij,j->ni", R[keep], s) + t[keep] - c, axis=1)
    # time-block jackknife on the kept samples
    edges = np.linspace(0, len(keep), blocks + 1).astype(int)
    loo = []
    for b in range(blocks):
        sub = np.r_[keep[: edges[b]], keep[edges[b + 1] :]]
        if len(sub) >= 6:
            loo.append(_solve(R[sub], t[sub])[0])
    loo = np.asarray(loo)
    sigma = (
        np.sqrt((len(loo) - 1) / len(loo) * ((loo - loo.mean(0)) ** 2).sum(0))
        if len(loo) > 1
        else np.full(3, np.nan)
    )
    span, weak = _span_deg(R[keep])
    return PivotFit(
        s_ee_m=s,
        c_base_m=c,
        s_sigma_m=sigma,
        residual_rms_m=float(np.sqrt(np.mean(r**2))),
        residual_p95_m=float(np.percentile(r, 95)),
        n_used=int(len(keep)),
        n_total=int(len(R)),
        span_deg=span,
        weak_singular=weak,
    )
