from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SMCErgodicConfig:
    dim: int
    low: tuple[float, ...]
    high: tuple[float, ...]
    num_k_per_dim: int = 5
    dt: float = 0.05
    speed: float = 0.05
    boundary: str = "reflect"
    seed: int = 0


def make_index_vectors(dim: int, num_k_per_dim: int) -> np.ndarray:
    axes = [np.arange(num_k_per_dim, dtype=int) for _ in range(dim)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=1)


def cosine_basis_norms(ks: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    per_dim_norm_sq = np.where(ks == 0, lengths, lengths / 2.0)
    return np.sqrt(np.prod(per_dim_norm_sq, axis=1))


def uniform_coefficients(ks: np.ndarray, volume: float) -> np.ndarray:
    phik = np.zeros(ks.shape[0], dtype=float)
    zero_idx = np.where(np.all(ks == 0, axis=1))[0]
    if zero_idx.size != 1:
        raise ValueError("Expected exactly one zero-frequency Fourier mode.")
    phik[zero_idx[0]] = 1.0 / np.sqrt(volume)
    return phik


def basis_and_gradient(
    x: np.ndarray,
    ks: np.ndarray,
    low: np.ndarray,
    lengths: np.ndarray,
    hk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_x = x - low
    angles = np.pi * ks * shifted_x / lengths
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    fk = np.prod(cos_vals, axis=1) / hk
    grad = np.empty((x.size, ks.shape[0]), dtype=float)
    for axis in range(x.size):
        if x.size == 1:
            prod_other = np.ones(ks.shape[0], dtype=float)
        else:
            prod_other = np.prod(np.delete(cos_vals, axis, axis=1), axis=1)
        grad[axis] = -np.pi * ks[:, axis] / lengths[axis] * sin_vals[:, axis] * prod_other / hk
    return fk, grad


def reflect_into_box(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    reflected = x.copy()
    lengths = high - low
    for axis in range(x.size):
        if lengths[axis] <= 0.0:
            raise ValueError("Each high bound must be larger than low bound.")
        while reflected[axis] < low[axis] or reflected[axis] > high[axis]:
            if reflected[axis] < low[axis]:
                reflected[axis] = low[axis] + (low[axis] - reflected[axis])
            if reflected[axis] > high[axis]:
                reflected[axis] = high[axis] - (reflected[axis] - high[axis])
        reflected[axis] = np.clip(reflected[axis], low[axis], high[axis])
    return reflected


def apply_boundary(x: np.ndarray, low: np.ndarray, high: np.ndarray, boundary: str) -> np.ndarray:
    if boundary == "reflect":
        return reflect_into_box(x, low, high)
    if boundary == "clip":
        return np.clip(x, low, high)
    if boundary == "none":
        return x
    raise ValueError(f"Unsupported boundary mode: {boundary}")


def _run_uniform_box_smc_cuda(
    cfg: SMCErgodicConfig,
    *,
    tsteps: int,
    x0: np.ndarray,
) -> dict[str, Any]:
    import torch

    device = torch.device("cuda")
    dtype = torch.float32
    low = torch.as_tensor(cfg.low, dtype=dtype, device=device).reshape(-1)
    high = torch.as_tensor(cfg.high, dtype=dtype, device=device).reshape(-1)
    x = torch.as_tensor(x0, dtype=dtype, device=device).reshape(cfg.dim)
    x = torch.clamp(x, low, high)
    lengths = high - low
    volume = torch.prod(lengths)

    ks_np = make_index_vectors(cfg.dim, cfg.num_k_per_dim)
    ks = torch.as_tensor(ks_np, dtype=dtype, device=device)
    per_dim_norm_sq = torch.where(ks == 0, lengths, lengths / 2.0)
    hk = torch.sqrt(torch.prod(per_dim_norm_sq, dim=1))
    phik = torch.zeros(ks.shape[0], dtype=dtype, device=device)
    zero_idx = torch.nonzero(torch.all(ks == 0, dim=1), as_tuple=False).reshape(-1)
    if int(zero_idx.numel()) != 1:
        raise ValueError("Expected exactly one zero-frequency Fourier mode.")
    phik[zero_idx[0]] = 1.0 / torch.sqrt(volume)
    lamk = torch.pow(1.0 + torch.linalg.norm(ks, dim=1), -float(cfg.dim + 1) / 2.0)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    ck_integral = torch.zeros(ks.shape[0], dtype=dtype, device=device)
    x_traj = torch.zeros((tsteps, cfg.dim), dtype=dtype, device=device)
    u_traj = torch.zeros((tsteps, cfg.dim), dtype=dtype, device=device)
    metric_log = torch.zeros(tsteps, dtype=dtype, device=device)

    for step_idx in range(tsteps):
        shifted_x = x - low
        angles = torch.pi * ks * shifted_x / lengths
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        fk = torch.prod(cos_vals, dim=1) / hk
        grad = torch.empty((cfg.dim, ks.shape[0]), dtype=dtype, device=device)
        for axis in range(cfg.dim):
            if cfg.dim == 1:
                prod_other = torch.ones(ks.shape[0], dtype=dtype, device=device)
            else:
                other_axes = [idx for idx in range(cfg.dim) if idx != axis]
                prod_other = torch.prod(cos_vals[:, other_axes], dim=1)
            grad[axis] = -torch.pi * ks[:, axis] / lengths[axis] * sin_vals[:, axis] * prod_other / hk

        ck_integral += fk * float(cfg.dt)
        ck = ck_integral / (float(step_idx + 1) * float(cfg.dt))
        bt = torch.sum(lamk * (ck - phik) * grad, dim=1)
        bt_norm = torch.linalg.norm(bt)
        if float(bt_norm.item()) < 1e-12:
            direction = torch.randn(cfg.dim, dtype=dtype, device=device, generator=generator)
            direction = direction / torch.clamp(torch.linalg.norm(direction), min=1e-12)
            u = float(cfg.speed) * direction
        else:
            u = -float(cfg.speed) * bt / bt_norm

        x = x + float(cfg.dt) * u
        if cfg.boundary == "reflect":
            period = 2.0 * lengths
            y = torch.remainder(x - low, period)
            x = torch.where(y <= lengths, low + y, high - (y - lengths))
        elif cfg.boundary == "clip":
            x = torch.clamp(x, low, high)
        elif cfg.boundary != "none":
            raise ValueError(f"Unsupported boundary mode: {cfg.boundary}")

        x_traj[step_idx] = x
        u_traj[step_idx] = u
        metric_log[step_idx] = torch.sum(lamk * torch.square(phik - ck))

    return {
        "x_traj": x_traj.cpu().numpy().astype(np.float64),
        "u_traj": u_traj.cpu().numpy().astype(np.float64),
        "metric_log": metric_log.cpu().numpy().astype(np.float64),
        "ks": ks_np,
        "low": low.cpu().numpy().astype(np.float64),
        "high": high.cpu().numpy().astype(np.float64),
        "device": "cuda",
    }


def run_uniform_box_smc(
    cfg: SMCErgodicConfig,
    *,
    tsteps: int,
    x0: np.ndarray,
    device: str = "cpu",
) -> dict[str, Any]:
    if cfg.dim <= 0:
        raise ValueError("SMC dim must be positive.")
    if cfg.num_k_per_dim <= 0:
        raise ValueError("num_k_per_dim must be positive.")
    if cfg.dt <= 0.0:
        raise ValueError("SMC dt must be positive.")
    if cfg.speed <= 0.0:
        raise ValueError("SMC speed must be positive.")
    if tsteps <= 0:
        raise ValueError("tsteps must be positive.")
    if device == "cuda":
        return _run_uniform_box_smc_cuda(cfg, tsteps=tsteps, x0=x0)

    low = np.asarray(cfg.low, dtype=np.float64).reshape(-1)
    high = np.asarray(cfg.high, dtype=np.float64).reshape(-1)
    if low.size != cfg.dim or high.size != cfg.dim:
        raise ValueError(f"low/high must have {cfg.dim} values.")
    if np.any(high <= low):
        raise ValueError("Each high bound must be larger than low bound.")
    x = np.asarray(x0, dtype=np.float64).reshape(cfg.dim)
    x = np.clip(x, low, high)

    rng = np.random.default_rng(int(cfg.seed))
    lengths = high - low
    volume = float(np.prod(lengths))
    ks = make_index_vectors(cfg.dim, cfg.num_k_per_dim)
    hk = cosine_basis_norms(ks, lengths)
    phik = uniform_coefficients(ks, volume)
    lamk = np.power(1.0 + np.linalg.norm(ks, axis=1), -(cfg.dim + 1) / 2.0)

    ck_integral = np.zeros(ks.shape[0], dtype=np.float64)
    x_traj = np.zeros((tsteps, cfg.dim), dtype=np.float64)
    u_traj = np.zeros((tsteps, cfg.dim), dtype=np.float64)
    metric_log = np.zeros(tsteps, dtype=np.float64)

    for step_idx in range(tsteps):
        fk, dfk = basis_and_gradient(x, ks, low, lengths, hk)
        ck_integral += fk * cfg.dt
        ck = ck_integral / ((step_idx + 1) * cfg.dt)
        bt = np.sum(lamk * (ck - phik) * dfk, axis=1)
        bt_norm = float(np.linalg.norm(bt))
        if bt_norm < 1e-12:
            direction = rng.normal(size=cfg.dim)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            u = cfg.speed * direction
        else:
            u = -cfg.speed * bt / bt_norm

        x = apply_boundary(x + cfg.dt * u, low, high, cfg.boundary)
        x_traj[step_idx] = x
        u_traj[step_idx] = u
        metric_log[step_idx] = np.sum(lamk * np.square(phik - ck))

    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "metric_log": metric_log,
        "ks": ks,
        "low": low,
        "high": high,
        "device": "cpu",
    }