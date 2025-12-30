"""
Batch concatenation utilities shared by training and OT pipelines.

This module centralizes robust concatenation logic to avoid drift between
training scripts. It is careful about:
- Device alignment when mixing tensors coming from different DataLoaders
- Restricting rank alignment (e.g., (B,T,D) vs (B,D)) to a whitelist of
  vector-like keys to avoid broadcasting images by mistake
- Padding known masks such as 'action_is_pad' when a key is present only on
  one side
"""
from __future__ import annotations

from typing import Dict, Iterable, Set

import torch


def _infer_batch_size(d: Dict) -> int | None:
    """Infer batch size by checking common tensor leaves.

    Prefer action-like keys; otherwise return the first tensor leaf's batch.
    """
    for k in ("action", "actions"):
        v = d.get(k, None)
        if hasattr(v, "shape") and v.__class__.__name__ == "Tensor":
            return int(v.shape[0])
    for _v in d.values():
        if isinstance(_v, dict):
            bs = _infer_batch_size(_v)
            if bs is not None:
                return bs
        elif hasattr(_v, "shape") and _v.__class__.__name__ == "Tensor":
            return int(_v.shape[0])
    return None


def concat_two_batches(
    b1: Dict,
    b2: Dict,
    *,
    rank_align_keys: Iterable[str] | None = None,
) -> Dict:
    """Concatenate two nested-batch dicts along batch dim for matching tensor leaves.

    - If both sides carry a tensor leaf under the same key, they are cat'ed on dim 0.
    - If dims differ (e.g., (B,T,D) vs (B,D)), rank alignment is only applied for
      known vector-like keys to avoid broadcasting images/videos.
    - If a tensor key exists only on one side and is 'action_is_pad', the other
      side is padded with zeros.
    - Non-tensor leaves are kept from the left side when present, otherwise from
      the right side (never cat'ed).
    """
    if rank_align_keys is None:
        # Safe defaults; include both 'action' and 'actions' for BC and OT.
        rank_align_keys = {"action", "actions", "observation.state", "observation.environment_state"}
    else:
        rank_align_keys = set(rank_align_keys)

    keys: Set[str] = set(b1.keys()) | set(b2.keys())
    out: Dict = {}

    for k in keys:
        in1 = k in b1
        in2 = k in b2
        if in1 and in2:
            v1, v2 = b1[k], b2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                out[k] = concat_two_batches(v1, v2, rank_align_keys=rank_align_keys)
            elif (
                hasattr(v1, "shape")
                and v1.__class__.__name__ == "Tensor"
                and hasattr(v2, "shape")
                and v2.__class__.__name__ == "Tensor"
            ):
                # Align device (prefer non-CPU)
                if v1.device != v2.device:
                    if v1.device.type != "cpu":
                        v2 = v2.to(v1.device, non_blocking=(v1.device.type == "cuda"))
                    elif v2.device.type != "cpu":
                        v1 = v1.to(v2.device, non_blocking=(v2.device.type == "cuda"))

                # Optional rank alignment for whitelisted keys
                if v1.dim() != v2.dim() and k in rank_align_keys:
                    if v1.dim() == 3 and v2.dim() == 2 and v1.size(-1) == v2.size(-1):
                        v2 = v2.unsqueeze(1).repeat(1, v1.size(1), 1)
                    elif v1.dim() == 2 and v2.dim() == 3 and v1.size(-1) == v2.size(-1):
                        v1 = v1.unsqueeze(1).repeat(1, v2.size(1), 1)

                out[k] = torch.cat([v1, v2], dim=0)
            else:
                # Non-tensors: prefer left side to keep metadata stable
                out[k] = v1
        elif in1:
            v1 = b1[k]
            if hasattr(v1, "shape") and v1.__class__.__name__ == "Tensor":
                if k == "action_is_pad" and v1.dim() >= 1:
                    bs2 = _infer_batch_size(b2) or 0
                    pad_shape = (bs2,) + tuple(v1.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=v1.dtype, device=v1.device)
                    out[k] = torch.cat([v1, pad], dim=0)
                else:
                    out[k] = v1
            else:
                out[k] = v1
        else:
            v2 = b2[k]
            if hasattr(v2, "shape") and v2.__class__.__name__ == "Tensor":
                if k == "action_is_pad" and v2.dim() >= 1:
                    bs1 = _infer_batch_size(b1) or 0
                    pad_shape = (bs1,) + tuple(v2.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=v2.dtype, device=v2.device)
                    out[k] = torch.cat([pad, v2], dim=0)
                else:
                    out[k] = v2
            else:
                out[k] = v2

    return out

