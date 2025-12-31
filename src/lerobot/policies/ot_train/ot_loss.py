#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
#
# Generic Optimal Transport (OT) loss utilities for LeRobot.
#
# This module provides:
#   - Configuration dataclasses for describing OT features and cost terms.
#   - A lightweight Sinkhorn-based OT loss implementation built on top of POT.
#   - (Deprecated) A simple OTEmbeddingHead that was previously used for learned costs.
#
# Design notes:
#   - The public surface is intentionally small and policy-agnostic.
#   - Policies own any learnable embedding heads (e.g. via `policy.ot_heads`).
#   - This file must NOT import from `lerobot.configs` or `lerobot.policies.*`
#     beyond standard torch / numpy, to avoid circular imports.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor, nn

try:  # Optional dependency – only required when OT is enabled.
    import ot as _ot
except Exception:  # pragma: no cover - optional dependency
    _ot = None


@dataclass
class OTTermConfig:
    """Configuration for a single cost term in the OT objective.

    Attributes:
        name: Human-readable name for this term (used in logs / metrics).
        weight: Scalar weight applied to this term when composing the final
            ground cost matrix M = Σ weight_i * M_i.
    """

    name: str
    weight: float = 1.0


@dataclass
class OTCostConfig:
    """Configuration for aggregating multiple cost terms into a single OT cost.

    Attributes:
        terms: List of cost term configs. The keys in the cost dictionary passed
            to `compute_ot_loss_from_terms` must match these `name` fields.
        reg: Entropic regularization strength for Sinkhorn (epsilon). Larger
            values yield smoother, more diffuse transport plans; smaller values
            produce sharper plans but can be numerically less stable.

        tau_src / tau_tgt: Optional unbalanced-OT marginal relaxation strengths.
            When either is not None we switch from standard balanced Sinkhorn to
            the unbalanced variant:

                ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m)

            where reg_m is either a scalar or a (tau_src, tau_tgt) tuple.
            By default both are None and we fall back to classic balanced OT.

        heuristic: Optional flag for adding a simple diagonal prior matrix `c`
            (as used in ot-sim2real / diffusion_policy_ot). When enabled and
            unbalanced OT is selected, we build:

                c[i, j] = 1 / N  if i == j < N
                        = 0      otherwise

            where N = min(n_src, n_tgt). This slightly biases the transport
            plan toward near-diagonal alignments. For balanced OT this flag is
            ignored.
    """

    terms: List[OTTermConfig] = field(default_factory=list)
    reg: float = 0.01
    tau_src: float | None = None
    tau_tgt: float | None = None
    heuristic: bool = False


@dataclass
class OTFeatureSpec:
    """Specification for how to build cost from a single source/target feature pair.

    This describes:
      - which observation keys to use on the src / tgt side;
      - how to slice the feature dimension;
      - whether to use a learnable embedding head in addition to label cost;
      - how to weight the embedding vs. label costs.
    """

    # Observation dict keys in src / tgt batches, e.g. "observation.state".
    src_key: str
    tgt_key: str

    # Optional slice along the last dimension (e.g. joints+gripper subset).
    dim_slice: slice | None = None

    # Learnable embedding configuration.
    use_learned_embed: bool = False
    embed_name: str | None = None

    # Weights for this feature's internal costs.
    weight_embed: float = 1.0
    weight_label: float = 1.0

    # Optional explicit term name; if None, derive from src_key / dim_slice.
    term_name: str | None = None
    # Optional top-level term weight when aggregating multiple terms (Σ w_i * M_i).
    # This mirrors ot-sim2real里单term的 emb_scale / cost_scale 之外的“外层”权重，用于多term场景。
    term_weight: float = 1.0


@dataclass
class OTLossConfig:
    """High-level configuration for building an OT loss from feature pairs."""

    features: List[OTFeatureSpec] = field(default_factory=list)
    reg: float = 0.0005
    # Optional unbalanced-OT parameters propagated down to OTCostConfig.
    # When both are None we compute a standard balanced OT plan.
    tau_src: float = 0.01
    tau_tgt: float = 0.01
    # Heuristic diagonal prior toggle (see OTCostConfig.heuristic for details).
    heuristic: bool = False


class OTEmbeddingHead(nn.Module):  # pragma: no cover
    """Deprecated: kept for import compatibility only.

    This class is no longer used by the OT loss path. OT embeddings must come
    from the policy's own encoders via `encode_feature_for_ot`.
    """

    def __init__(self, *args, **kwargs) -> None:  # ignore signature
        super().__init__()
        self._stub = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self._stub(x)


def _compute_sinkhorn_plan(
    cost: Tensor,
    reg: float,
    *,
    tau_src: float | None = None,
    tau_tgt: float | None = None,
    heuristic: bool = False,
) -> Tensor:
    """Compute a Sinkhorn OT plan for a given ground cost matrix.

    Args:
        cost: (B_src, B_tgt) ground cost matrix.
        reg: Entropic regularization strength (epsilon).
        tau_src / tau_tgt: If either is set, use the unbalanced Sinkhorn
            variant `ot.unbalanced.sinkhorn_knopp_unbalanced` with
            reg_m = (tau_src, tau_tgt) when both are provided or a scalar
            otherwise.
        heuristic: When True and using unbalanced OT, additionally pass a
            diagonal prior matrix `c` to POT, following ot-sim2real's
            diffusion_policy_ot implementation.

    Returns:
        plan: (B_src, B_tgt) transport plan tensor on the same device / dtype.
    """
    if _ot is None:
        raise RuntimeError(
            "OT loss requested but the 'ot' package (POT) is not available. "
            "Install it with `pip install pot` to enable OT training."
        )

    if cost.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape {tuple(cost.shape)}")

    n_src, n_tgt = cost.shape
    if n_src == 0 or n_tgt == 0:
        raise ValueError(f"Cannot compute OT for empty cost matrix of shape {tuple(cost.shape)}")

    # Build uniform marginals.
    a = torch.full((n_src,), 1.0 / n_src, dtype=cost.dtype, device=cost.device)
    b = torch.full((n_tgt,), 1.0 / n_tgt, dtype=cost.dtype, device=cost.device)

    # Run Sinkhorn in numpy to leverage POT, then convert back to torch.
    # Use float64 for numerical stability — small couplings can underflow in fp32
    # when the ground cost has a much larger magnitude than `reg`.
    cost_np = cost.detach().cpu().to(dtype=torch.float64).numpy()
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    unbalanced_mod = getattr(_ot, "unbalanced", None)
    if unbalanced_mod is None or not hasattr(unbalanced_mod, "sinkhorn_knopp_unbalanced"):
        raise RuntimeError(
            "Unbalanced OT requested (tau_src/tau_tgt set) but "
            "'ot.unbalanced.sinkhorn_knopp_unbalanced' is not available in the "
            "installed 'ot' (POT) package."
        )

    # reg_m can be a scalar or a (tau_src, tau_tgt) tuple.
    if tau_src is not None and tau_tgt is not None:
        reg_m = (float(tau_src), float(tau_tgt))
    else:
        reg_m = float(tau_src if tau_src is not None else tau_tgt)  # type: ignore[arg-type]

    # Optional heuristic diagonal prior matrix c, matching the ot-sim2real /
    # diffusion_policy_ot convention.
    if heuristic:
        src_n, tgt_n = cost_np.shape
        diag_n = min(src_n, tgt_n)
        c = np.zeros_like(cost_np)
        if diag_n > 0:
            idx = np.arange(diag_n)
            c[idx, idx] = 1.0 / float(diag_n)
    else:
        c = None

    if c is None:
        pi_np = unbalanced_mod.sinkhorn_knopp_unbalanced(  # type: ignore[union-attr]
            a_np,
            b_np,
            cost_np,
            reg,
            reg_m,
        )
    else:
        pi_np = unbalanced_mod.sinkhorn_knopp_unbalanced(  # type: ignore[union-attr]
            a_np,
            b_np,
            cost_np,
            reg,
            reg_m,
            c=c,
        )

    # Fallback: if numerical issues occurred, revert to uniform coupling so
    # training can continue without NaNs.
    if not np.isfinite(pi_np).all():
        n_src, n_tgt = cost_np.shape
        pi_np = np.full_like(cost_np, 1.0 / (n_src * n_tgt))

    # Keep high precision to avoid underflow in tiny couplings; downstream ops
    # can safely cast as needed by PyTorch's type promotion rules.
    pi = torch.from_numpy(pi_np).to(device=cost.device, dtype=torch.float64)
    return pi


def compute_ot_loss_from_terms(
    term_costs: Mapping[str, Tensor],
    cfg: OTCostConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compose multiple term costs into a single OT loss.

    Args:
        term_costs: Mapping from term name to cost matrix tensor (B_src, B_tgt).
        cfg: OTCostConfig describing how to weight each term and the Sinkhorn
            regularization parameter.

    Returns:
        ot_loss: Scalar OT loss tensor.
        metrics: Dictionary of scalar metrics (float) including:
            - "ot_loss"
            - "ot_pi_sum"
            - "ot_pi_diag" (if B_src == B_tgt)
            - Optional per-term average costs: "ot_cost/<term_name>"
    """
    if len(cfg.terms) == 0:
        raise ValueError("OTCostConfig.terms is empty – nothing to optimize.")

    # Build aggregate ground cost matrix M = Σ w_i * M_i.
    M: Tensor | None = None
    avg_term_costs: Dict[str, float] = {}

    for term in cfg.terms:
        if term.name not in term_costs:
            raise KeyError(
                f"Cost term '{term.name}' missing from term_costs. "
                f"Available terms: {list(term_costs.keys())}"
            )
        C_i = term_costs[term.name]
        if M is None:
            M = term.weight * C_i
        else:
            M = M + term.weight * C_i
        avg_term_costs[term.name] = float(C_i.mean().item())

    assert M is not None  # for type checkers

    # Compute Sinkhorn OT plan and final loss.
    pi = _compute_sinkhorn_plan(
        M,
        reg=cfg.reg,
        tau_src=cfg.tau_src,
        tau_tgt=cfg.tau_tgt,
        heuristic=cfg.heuristic,
    )
    ot_loss = torch.sum(pi * M)

    metrics: Dict[str, float] = {
        "ot_loss": float(ot_loss.item()),
        "ot_pi_sum": float(pi.sum().item()),
    }
    if M.shape[0] == M.shape[1]:
        metrics["ot_pi_diag"] = float(torch.trace(pi).item())

    # Log per-term average cost for monitoring.
    for name, v in avg_term_costs.items():
        metrics[f"ot_cost/{name}"] = v

    return ot_loss, metrics


def compute_ot_loss_for_policy(
    policy: nn.Module,
    src_obs: Mapping[str, Tensor],
    tgt_obs: Mapping[str, Tensor],
    cfg: OTLossConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """High-level helper to build OT loss from policy features and a config.

    This function builds the embedding part of the OT cost strictly from the
    policy's own observation encoders when `OTFeatureSpec.use_learned_embed`
    is True. We do NOT support any separate OT-specific heads here.

    If the policy implements `encode_feature_for_ot(key: str, x: Tensor) -> Tensor`,
    that hook is used. Otherwise, we apply light heuristics for known policies
    to reuse their internal encoders (e.g., ACT input projections, diffusion
    RGB encoder). If no policy-native encoder is available for a requested
    feature, an error is raised.

    Args:
        policy: Policy module. Optionally has an `ot_heads` attribute with
            embedding modules.
        src_obs: Observation dict for the source domain (e.g. from OT src dataset).
        tgt_obs: Observation dict for the target domain (e.g. from OT tgt dataset).
        cfg: OTLossConfig describing features and regularization.

    Returns:
        ot_loss: Scalar OT loss tensor.
        metrics: Dictionary of scalar metrics.
    """
    if len(cfg.features) == 0:
        raise ValueError("OTLossConfig.features is empty – no features specified for OT.")

    term_costs: Dict[str, Tensor] = {}

    # Try to infer the target device from the policy parameters so that all OT
    # computations run on the same device
    # as the model. This avoids CPU/GPU mismatches when dataloaders yield CPU
    # tensors but the policy is on CUDA.
    try:
        first_param = next(policy.parameters())
        target_device = first_param.device
    except StopIteration:  # pragma: no cover - policies without parameters are rare
        target_device = None

    # Optional fast path: some common policies don't expose an explicit
    # `encode_feature_for_ot` but do have well-known encoders we can reuse.
    # We wrap them here so the rest of the function can just call
    # `_encode_with_policy(key, x)` and get back an embedding or None.
    def _encode_with_policy(key: str, x: Tensor) -> Tensor | None:
        # 1) Preferred explicit hook
        if hasattr(policy, "encode_feature_for_ot"):
            try:
                enc = getattr(policy, "encode_feature_for_ot")(key, x)
                if isinstance(enc, Tensor):
                    return enc
            except Exception:
                # Fall through to heuristics
                pass

        # 2) Heuristics for known policies
        # ACT: project robot/env states with the policy's input projections
        try:
            from lerobot.utils.constants import OBS_STATE as _OBS_STATE, OBS_ENV_STATE as _OBS_ENV_STATE
        except Exception:
            _OBS_STATE = "observation.state"  # type: ignore[assignment]
            _OBS_ENV_STATE = "observation.environment_state"  # type: ignore[assignment]

        # ACTPolicy has `.model` with these projections
        model = getattr(policy, "model", None)
        if model is not None:
            if key == _OBS_STATE and hasattr(model, "encoder_robot_state_input_proj"):
                proj = getattr(model, "encoder_robot_state_input_proj")
                if isinstance(proj, nn.Module):
                    return proj(x)
            if key == _OBS_ENV_STATE and hasattr(model, "encoder_env_state_input_proj"):
                proj = getattr(model, "encoder_env_state_input_proj")
                if isinstance(proj, nn.Module):
                    return proj(x)

        # DiffusionPolicy: use identity for state, and rgb_encoder for images if available
        if key == _OBS_STATE:
            return x  # diffusion treats state as part of global cond without extra MLP
        # Image keys: try to locate a diffusion rgb encoder
        if isinstance(key, str) and (key.startswith("observation.images") or key.startswith("observation.image")):
            # Detect a nested diffusion model with rgb_encoder
            diff = getattr(policy, "diffusion", None)
            if diff is not None and hasattr(diff, "rgb_encoder"):
                rgb_enc = getattr(diff, "rgb_encoder")
                # Either a shared encoder or a list of per-camera encoders; both
                # implement a forward(x)->(B, D) on single images.
                if isinstance(rgb_enc, nn.Module):
                    return rgb_enc(x)
        return None

    def _resolve_obs_key(requested_key: str, obs: Mapping[str, Tensor]) -> str | None:
        if requested_key in obs:
            return requested_key
        if isinstance(requested_key, str) and (
            requested_key == "observation.images"
            or requested_key.startswith("observation.images")
            or requested_key == "observation.image"
            or requested_key.startswith("observation.image")
        ):
            # Prefer image-like tensors (>=3 dims)
            for k, v in obs.items():
                if isinstance(k, str) and (
                    k.startswith("observation.images.") or k.startswith("observation.image")
                ):
                    if torch.is_tensor(v) and v.ndim >= 3:
                        return k
        return None

    for feat in cfg.features:
        # Resolve possibly-generic keys (e.g., 'observation.images') to concrete keys
        src_key = _resolve_obs_key(feat.src_key, src_obs) or feat.src_key
        tgt_key = _resolve_obs_key(feat.tgt_key, tgt_obs) or feat.tgt_key
        if src_key not in src_obs:
            raise KeyError(f"src_obs missing key '{feat.src_key}' required by OT feature.")
        if tgt_key not in tgt_obs:
            raise KeyError(f"tgt_obs missing key '{feat.tgt_key}' required by OT feature.")

        src = src_obs[src_key]
        tgt = tgt_obs[tgt_key]
        # 保留原始张量，用于 label 路径做“时间展平”以对齐 ot-sim2real（flatten across time）
        src_raw = src
        tgt_raw = tgt

        # Some datasets (e.g., diffusion training) provide a temporal stack for
        # observations like state/env_state with shape (B, S, D). OT operates on
        # per-step features (B, D). To be robust, collapse any temporal dimension
        # by selecting the latest step along axis 1 for known keys.
        try:
            from lerobot.utils.constants import (
                ACTION as _ACTION,
                OBS_STATE as _OBS_STATE,
                OBS_ENV_STATE as _OBS_ENV_STATE,
                OBS_IMAGES as _OBS_IMAGES,
                OBS_IMAGE as _OBS_IMAGE,
            )
        except Exception:  # pragma: no cover - constants import is best-effort
            _ACTION = "action"  # type: ignore[assignment]
            _OBS_STATE = "observation.state"  # type: ignore[assignment]
            _OBS_ENV_STATE = "observation.environment_state"  # type: ignore[assignment]
            _OBS_IMAGES = "observation.images"  # type: ignore[assignment]
            _OBS_IMAGE = "observation.image"  # type: ignore[assignment]

        def _collapse_time_if_needed(key: str, x: Tensor) -> Tensor:
            # For state-like features, accept (B, S, D) and take the last S.
            if x.ndim == 3 and (key == _OBS_STATE or key == _OBS_ENV_STATE):
                return x[:, -1, :]
            # For actions shaped (B, T, D), mimic diffusion_policy_ot: select first timestep
            if x.ndim == 3 and (key == _ACTION or key == "action"):
                return x[:, 0, :]
            # For images possibly shaped (B, S, ...), take the last S to get (B, ...)
            if isinstance(key, str) and (key.startswith(_OBS_IMAGES) or key.startswith(_OBS_IMAGE)) and x.ndim >= 5:
                return x[:, -1]
            return x

        src_for_embed = _collapse_time_if_needed(feat.src_key, src)
        tgt_for_embed = _collapse_time_if_needed(feat.tgt_key, tgt)

        # Move features to the policy's device if necessary so that they match
        # any OT embedding head parameters.
        if target_device is not None:
            if src_for_embed.device != target_device:
                src_for_embed = src_for_embed.to(target_device)
            if tgt_for_embed.device != target_device:
                tgt_for_embed = tgt_for_embed.to(target_device)

        # Build label cost if requested; requires 2D tensors (B, D)
        build_label = float(getattr(feat, "weight_label", 0.0)) != 0.0

        # Label 路径：以 ot-sim2real 为准，若存在时间维（B,S,D），先在最后一维做 dim_slice，再在 S 维展平到 2D。
        if build_label:
            s_raw = src_raw
            t_raw = tgt_raw
            # 先做 dim_slice（按最后一维）
            if feat.dim_slice is not None and s_raw.ndim >= 2:
                if s_raw.ndim == 2:
                    s_raw = s_raw[:, feat.dim_slice]
                else:
                    s_raw = s_raw[..., feat.dim_slice]
            if feat.dim_slice is not None and t_raw.ndim >= 2:
                if t_raw.ndim == 2:
                    t_raw = t_raw[:, feat.dim_slice]
                else:
                    t_raw = t_raw[..., feat.dim_slice]
            # 再在 batch 之外全部展平，得到 (B, D_flat)
            if s_raw.ndim >= 3:
                s_lab = torch.flatten(s_raw, start_dim=1)
            else:
                s_lab = s_raw
            if t_raw.ndim >= 3:
                t_lab = torch.flatten(t_raw, start_dim=1)
            else:
                t_lab = t_raw
            # 移到设备
            if target_device is not None:
                if s_lab.device != target_device:
                    s_lab = s_lab.to(target_device)
                if t_lab.device != target_device:
                    t_lab = t_lab.to(target_device)
            src_slice = s_lab
            tgt_slice = t_lab
        else:
            # 不需要 label cost 时，embed 路径仍使用处理过的 2D 表征
            src_slice = src_for_embed
            tgt_slice = tgt_for_embed

        # Label cost in the original feature space. To match ot-sim2real's
        # behavior, label terms are treated as pure targets and do not receive
        # gradients – we explicitly detach them here.
        if build_label:
            src_label = src_slice.detach()
            tgt_label = tgt_slice.detach()
            M_label = torch.cdist(src_label, tgt_label) ** 2
        else:
            # Zero label cost if not requested.
            # Allocate a placeholder zero matrix with proper device/dtype once we know sizes.
            # We'll initialize it after we compute the embedding cost dims, or fallback to src/tgt dims if 2D.
            M_label = None  # type: ignore[assignment]

        # Optional embedding cost.
        if feat.use_learned_embed:
            # Require a policy-native encoder for learned embeddings.
            src_encoded = _encode_with_policy(feat.src_key, src_for_embed)
            tgt_encoded = _encode_with_policy(feat.tgt_key, tgt_for_embed)
            if src_encoded is None or tgt_encoded is None:
                raise ValueError(
                    f"No policy-native encoder available for OT embedding of keys "
                    f"'{feat.src_key}' / '{feat.tgt_key}'. Implement 'encode_feature_for_ot' on the policy "
                    f"or ensure the policy exposes appropriate encoders."
                )
            # 若编码结果仍包含时间维（B,S,D），按 ot-sim2real 展平为 (B, S*D)
            if src_encoded.ndim >= 3:
                src_encoded = torch.flatten(src_encoded, start_dim=1)
            if tgt_encoded.ndim >= 3:
                tgt_encoded = torch.flatten(tgt_encoded, start_dim=1)
            # Keep full encoded reps; don't reuse dim_slice here to avoid mismatch with policy dims.
            M_embed = torch.cdist(src_encoded, tgt_encoded) ** 2
        else:
            # Zero embedding cost if disabled.
            # If label is disabled too, we'll fill M_embed lazily below.
            M_embed = None  # type: ignore[assignment]

        # Lazily create zero matrices if needed
        if M_embed is None and M_label is None:
            # Edge case: both disabled
            raise ValueError("Both embedding and label costs are disabled for an OT feature.")
        if M_embed is None:
            # match label shape
            assert M_label is not None
            M_embed = torch.zeros_like(M_label)
        if M_label is None:
            # match embed shape
            assert M_embed is not None
            M_label = torch.zeros_like(M_embed)

        # Combine into a single term cost.
        M_term = feat.weight_embed * M_embed + feat.weight_label * M_label

        # Term name for metrics.
        name = feat.term_name or feat.src_key
        term_costs[name] = M_term

    # Build a corresponding OTCostConfig from feature specs.
    terms_cfg = [
        OTTermConfig(name=(feat.term_name or feat.src_key), weight=float(getattr(feat, "term_weight", 1.0)))
        for feat in cfg.features
    ]
    cost_cfg = OTCostConfig(
        terms=terms_cfg,
        reg=cfg.reg,
        tau_src=cfg.tau_src,
        tau_tgt=cfg.tau_tgt,
        heuristic=cfg.heuristic,
    )
    return compute_ot_loss_from_terms(term_costs, cost_cfg)
