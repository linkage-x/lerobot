import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def concat_two_batch(batch1: dict, batch2: dict) -> dict:
    keys = set(batch1.keys()) | set(batch2.keys())
    out = {}
    for k in keys:
        if k in batch1 and k in batch2:
            if isinstance(batch1[k], dict):
                out[k] = concat_two_batch(batch1[k], batch2[k])
            else:
                v1, v2 = batch1[k], batch2[k]
                # Ensure both tensors are on the same device before concatenation.
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    if v1.device != v2.device:
                        if v1.device.type != "cpu":
                            v2 = v2.to(v1.device, non_blocking=(v1.device.type == "cuda"))
                        elif v2.device.type != "cpu":
                            v1 = v1.to(v2.device, non_blocking=(v2.device.type == "cuda"))
                    # Align ranks if needed (e.g., (B, T, D) with (B, D)).
                    if v1.dim() != v2.dim():
                        if v1.dim() == 3 and v2.dim() == 2 and v1.size(-1) == v2.size(-1):
                            v2 = v2.unsqueeze(1).repeat(1, v1.size(1), 1)
                        elif v1.dim() == 2 and v2.dim() == 3 and v1.size(-1) == v2.size(-1):
                            v1 = v1.unsqueeze(1).repeat(1, v2.size(1), 1)
                out[k] = torch.cat([v1, v2], dim=0)
        elif k in batch1:
            v1 = batch1[k]
            if isinstance(v1, torch.Tensor):
                # Pad missing side with zeros if it's the boolean pad mask.
                if k == "action_is_pad" and v1.dim() >= 1:
                    # Infer batch size from other batch
                    def _infer_bs(d: dict) -> int | None:
                        if "action" in d and isinstance(d["action"], torch.Tensor):
                            return int(d["action"].shape[0])
                        for _v in d.values():
                            if isinstance(_v, dict):
                                bs = _infer_bs(_v)
                                if bs is not None:
                                    return bs
                            elif isinstance(_v, torch.Tensor):
                                return int(_v.shape[0])
                        return None
                    bs2 = _infer_bs(batch2) or 0
                    pad = torch.zeros((bs2,) + tuple(v1.shape[1:]), dtype=v1.dtype, device=v1.device)
                    out[k] = torch.cat([v1, pad], dim=0)
                else:
                    out[k] = v1
            else:
                out[k] = v1
        else:
            v2 = batch2[k]
            if isinstance(v2, torch.Tensor):
                if k == "action_is_pad" and v2.dim() >= 1:
                    def _infer_bs(d: dict) -> int | None:
                        if "action" in d and isinstance(d["action"], torch.Tensor):
                            return int(d["action"].shape[0])
                        for _v in d.values():
                            if isinstance(_v, dict):
                                bs = _infer_bs(_v)
                                if bs is not None:
                                    return bs
                            elif isinstance(_v, torch.Tensor):
                                return int(_v.shape[0])
                        return None
                    bs1 = _infer_bs(batch1) or 0
                    pad = torch.zeros((bs1,) + tuple(v2.shape[1:]), dtype=v2.dtype, device=v2.device)
                    out[k] = torch.cat([pad, v2], dim=0)
                else:
                    out[k] = v2
            else:
                out[k] = v2
    return out


def run_epoch_for_ot_policy(
    model,
    ot_params: dict,
    bc_dataloader,
    ot_dataloader,
    epoch: int,
    validate: bool = False,
    num_steps: Optional[int] = None,
    obs_normalization_stats: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    OT-specific epoch loop adapted from robomimic's OT sim2real training.

    Expectations on `model` (policy):
      - process_batch_for_training(batch, b_ot) -> dict
      - postprocess_batch_for_training(batch, obs_normalization_stats=None) -> dict
      - train_on_batch(batch, b_ot, ot_params, epoch, validate=False) -> dict
      - on_gradient_step()
      - log_info(info) -> dict

    This function is intended for policies that implement the above OT API.
    If any of these methods are missing, an exception is raised.
    """
    epoch_timestamp = time.time()
    if validate and hasattr(model, "set_eval"):
        model.set_eval()
    elif hasattr(model, "set_train"):
        model.set_train()

    assert num_steps is not None, "num_steps must be provided for OT epoch loop"

    step_logs = []
    timing = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])

    def infinite(dl):
        while True:
            for b in dl:
                yield b

    ot_iter = infinite(ot_dataloader)
    bc_iter = infinite(bc_dataloader)

    supports_ot = all(
        hasattr(model, attr)
        for attr in [
            "process_batch_for_training",
            "postprocess_batch_for_training",
            "train_on_batch",
            "on_gradient_step",
            "log_info",
        ]
    )
    if not supports_ot:
        raise RuntimeError(
            "run_epoch_for_ot_policy requires a policy implementing "
            "'process_batch_for_training', 'postprocess_batch_for_training', "
            "'train_on_batch', 'on_gradient_step', and 'log_info'."
        )

    for _ in range(num_steps):
        t0 = time.time()
        ot_batch = next(ot_iter)
        bc_batch = next(bc_iter)
        timing["Data_Loading"].append(time.time() - t0)

        if supports_ot:
            logger.debug("run_epoch_for_ot_policy: supports OT API, running OT step")
            t1 = time.time()
            src_ot = ot_batch["src"]
            tgt_ot = ot_batch["tgt"]
            b_ot = src_ot["actions"].shape[0]
            ot_cat = concat_two_batch(src_ot, tgt_ot)
            full = concat_two_batch(ot_cat, bc_batch)
            if hasattr(model, "process_batch_for_training"):
                full = model.process_batch_for_training(full, b_ot)
            if hasattr(model, "postprocess_batch_for_training"):
                full = model.postprocess_batch_for_training(
                    full, obs_normalization_stats=obs_normalization_stats
                )
            timing["Process_Batch"].append(time.time() - t1)

            t2 = time.time()
            info = model.train_on_batch(full, b_ot, ot_params, epoch, validate=validate)
            timing["Train_Batch"].append(time.time() - t2)

            t3 = time.time()
            if hasattr(model, "on_gradient_step"):
                model.on_gradient_step()
            step_log = model.log_info(info) if hasattr(model, "log_info") else info
            timing["Log_Info"].append(time.time() - t3)

        step_logs.append(step_log)

    # Aggregate
    agg = {}
    for d in step_logs:
        for k, v in d.items():
            agg.setdefault(k, []).append(v)

    # 1) Preserve the last 'pi' (e.g., heatmap) for outer visualization, like ot-sim2real
    pi_seq = agg.pop("pi", None)

    # 2) Broaden numeric aggregation: accept Python, numpy, and torch scalars
    def _to_scalar(x):
        # Convert supported scalar-like types to Python float; raise on non-scalars
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.item()
            raise TypeError("non-scalar tensor")
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            raise TypeError("non-scalar ndarray")
        try:
            return float(x)
        except Exception as e:
            raise TypeError("non-scalar or non-numeric") from e

    result: Dict[str, Any] = {}
    for k, vs in agg.items():
        try:
            scalars = [_to_scalar(x) for x in vs]
            result[k] = float(np.mean(scalars))
        except TypeError:
            # Skip metrics that aren't scalar-aggregatable
            continue

    for k, arr in timing.items():
        result[f"Time_{k}"] = float(np.sum(arr) / 60.0)
    result["Time_Epoch"] = float((time.time() - epoch_timestamp) / 60.0)
    if pi_seq is not None:
        result["pi"] = pi_seq[-1]
    return result


def make_ot_dataloader(
    ds_src,
    ds_tgt,
    pair_info_path: str,
    *,
    batch_size: int = 8,
    obs_keys: list[str] | None = None,
    action_key: str = "action",
    base_index_src: int = 0,
    base_index_tgt: int = 0,
    window_size: int | None = None,
    # Optional sampling controls (pass-through to LeRobotOTPairDataset)
    sharpness: float = 0.0,
    no_window: bool = False,
    topk_src_episodes: int | None = None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
):
    """
    Convenience helper to build an OT pair dataloader from LeRobot datasets.

    Returns a `torch.utils.data.DataLoader` yielding dict with keys 'src' and 'tgt'.
    """
    from torch.utils.data import DataLoader
    from .ot_dataset import LeRobotOTPairDataset, collate_ot_samples

    dataset = LeRobotOTPairDataset(
        ds_src=ds_src,
        ds_tgt=ds_tgt,
        pair_info_path=pair_info_path,
        obs_keys=obs_keys,
        action_key=action_key,
        base_index_src=base_index_src,
        base_index_tgt=base_index_tgt,
        window_size=window_size,
        sharpness=sharpness,
        no_window=no_window,
        topk_src_episodes=topk_src_episodes,
    )
    if pin_memory is None:
        try:
            device = next(iter(getattr(ds_src, "meta", {}).values()), None)
            pin_memory = torch.cuda.is_available()
        except Exception:
            pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        collate_fn=collate_ot_samples,
        drop_last=True,
    )
    return loader


def make_weighted_mixed_bc_dataloader(
    ds_tgt,
    ds_src,
    *,
    bc_src_weight: float,
    batch_size: int,
    num_workers: int = 4,
    normalize_by_size: bool = True,
    drop_n_last_frames: int = 0,
    pin_memory: bool | None = None,
):
    """
    Build a single DataLoader that mixes target and source BC samples with weighted
    random sampling (replacement=True), analogous to robomimic's MetaDataset.

    Args:
        ds_tgt: main/target LeRobotDataset
        ds_src: source LeRobotDataset
        bc_src_weight: probability mass assigned to source samples in expectation
        batch_size: dataloader batch size
        num_workers: dataloader workers
        normalize_by_size: if True, divide each dataset's weight by its (valid) size
        drop_n_last_frames: optional per-episode tail trimming to align windows
        pin_memory: defaults to CUDA availability
    Returns:
        torch.utils.data.DataLoader over a ConcatDataset with a WeightedRandomSampler
    """
    import torch
    from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    bc_src_weight = float(bc_src_weight)
    bc_tgt_weight = max(0.0, 1.0 - bc_src_weight)

    def _mask_valid_indices(ds, drop_n_last_frames: int) -> torch.Tensor:
        if drop_n_last_frames <= 0:
            return torch.ones(len(ds), dtype=torch.bool)
        mask = torch.zeros(len(ds), dtype=torch.bool)
        starts = ds.meta.episodes["dataset_from_index"]
        ends = ds.meta.episodes["dataset_to_index"]
        for s, e in zip(starts, ends, strict=True):
            end_keep = max(int(s), int(e) - int(drop_n_last_frames))
            if end_keep > int(s):
                mask[torch.arange(int(s), end_keep, dtype=torch.long)] = True
        return mask

    mask_tgt = _mask_valid_indices(ds_tgt, drop_n_last_frames)
    mask_src = _mask_valid_indices(ds_src, drop_n_last_frames)

    if normalize_by_size:
        wt_tgt = (bc_tgt_weight / max(1, int(mask_tgt.sum()))) if bc_tgt_weight > 0 else 0.0
        wt_src = (bc_src_weight / max(1, int(mask_src.sum()))) if bc_src_weight > 0 else 0.0
    else:
        wt_tgt = bc_tgt_weight
        wt_src = bc_src_weight

    weights_tgt = torch.where(mask_tgt, torch.full((len(ds_tgt),), float(wt_tgt)), torch.zeros(len(ds_tgt)))
    weights_src = torch.where(mask_src, torch.full((len(ds_src),), float(wt_src)), torch.zeros(len(ds_src)))
    weights = torch.cat([weights_tgt, weights_src])
    if float(weights.sum()) <= 0:
        weights = torch.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    mixed_dataset = ConcatDataset([ds_tgt, ds_src])
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
