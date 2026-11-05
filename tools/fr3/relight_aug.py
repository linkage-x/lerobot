"""Lighting augmentation for the ee camera (11-05): remove the real lamp shadows and re-light the frame.

Why: the S1 swap probe (outputs/real2sim_20261104/README.md, results 3-7) showed the policy's low close leans on the
gripper's shadow in the ee image; moving the key lamp 0.3-0.4 m on real frames dropped low closes 0.98 -> 0.75-0.88
and 15-35 mm closes 0.29 -> 0.1. The user wants the lighting influence removed from the policy, not the lamps fixed.

How: per training frame with ee <= 100 mm above the peg top, precomputed MuJoCo shadow occlusion maps
(outputs/real2sim_20261104/tools/relight_aug_maps.py) of the two room lamps at their fitted positions and at J
randomly moved positions. Shadow model (relight.py, fit_lamp2.py): a pixel in the key lamp's shadow has k1 = 0.69 x
its lit brightness, in lamp 2's shadow k2 = 0.88 x, multiplicative. Augmented frame:
    lit  = img / (k1o ** occ_key * k2o ** occ_l2)                      (deshadow; k1o, k2o jittered around the fit)
    out  = lit * k1n ** occ_key' * k2n ** occ_l2'                      (each lamp: none / fitted / moved j; k random)
then (all frames, both cameras) brightness / contrast / per-channel gain jitter.

Applied by patching LeRobotDataset.__getitem__ (install()): the item's images are relit before the dataset's own
image_transforms (Resize), using item episode_index / frame_index to find the maps. Env RELIGHT_AUG_MAPS = maps dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

EE_KEY = "observation.images.ee"

# probabilities / ranges (one place, written to the run's log at install)
CFG = dict(
    p_relight=0.75,          # frames with maps: chance to relight at all (else keep the real shadows)
    key_choice=(0.15, 0.15, 0.70),     # new key shadow: none / fitted position / moved (one of J)
    l2_choice=(0.25, 0.25, 0.50),      # new lamp-2 shadow: none / fitted / moved
    k1_new=(0.45, 0.90), k2_new=(0.70, 0.98),
    k1_old_jit=0.06, k2_old_jit=0.04,  # deshadow strength jitter around the fit (so leftovers are not one fixed halo)
    p_photo=0.8, brightness=(0.7, 1.3), contrast=(0.8, 1.2), gain=(0.9, 1.1),
)
K1_FIT, K2_FIT = 0.69, 0.88
# Systematic sim->real misregistration of the shadow maps, full-res crop px (u right, v down): the 11-05 shift check
# (outputs/real2sim_20261104/tools/shadow_shift_check.py) found the best shift ~(+14, +9) px with IQR 13-20 px in
# every source (09-30 .. 11-04) and on the S1 twin (+16, +8); explained variance 0.30 -> 0.38. Applied to all maps.
MAP_SHIFT_UV = (14.0, 9.0)


class RelightAug:
    def __init__(self, maps_dir: str | Path):
        self.dir = Path(maps_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.J = int(self.meta["J"])
        self.hw = tuple(self.meta["map_hw"])
        self.k1, self.k2 = self.meta["k_fit"]["key"], self.meta["k_fit"]["lamp2"]

    def maps(self, ep: int, fi: int) -> np.ndarray | None:
        p = self.dir / f"e{ep:04d}" / f"f{fi:04d}.png"
        if not p.exists():
            return None
        import cv2

        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            return None
        h, w = self.hw
        return a.reshape(-1, h, w).astype(np.float32) / 255.0   # (2 + 2J, h, w)

    def relight(self, img: torch.Tensor, m: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        """img (3, H, W) float in [0, 1] -> relit, same shape."""
        import cv2

        H, W = img.shape[-2:]
        M = np.float32([[1, 0, MAP_SHIFT_UV[0]], [0, 1, MAP_SHIFT_UV[1]]])
        up = lambda o: torch.from_numpy(cv2.warpAffine(cv2.resize(o, (W, H), interpolation=cv2.INTER_LINEAR), M, (W, H),
                                                       borderMode=cv2.BORDER_REPLICATE))
        k1o = self.k1 + rng.uniform(-CFG["k1_old_jit"], CFG["k1_old_jit"])
        k2o = self.k2 + rng.uniform(-CFG["k2_old_jit"], CFG["k2_old_jit"])
        log_f = np.log(k1o) * m[0] + np.log(k2o) * m[1]                # old shadow factor (log)
        c = rng.choice(3, p=CFG["key_choice"])
        if c == 1:
            log_f = log_f - np.log(rng.uniform(*CFG["k1_new"])) * m[0]
        elif c == 2:
            log_f = log_f - np.log(rng.uniform(*CFG["k1_new"])) * m[2 + 2 * rng.integers(self.J)]
        c = rng.choice(3, p=CFG["l2_choice"])
        if c == 1:
            log_f = log_f - np.log(rng.uniform(*CFG["k2_new"])) * m[1]
        elif c == 2:
            log_f = log_f - np.log(rng.uniform(*CFG["k2_new"])) * m[3 + 2 * rng.integers(self.J)]
        div = torch.exp(up(log_f.astype(np.float32)))                  # out = img / (old / new)
        return (img / div[None]).clamp(0, 1)

    @staticmethod
    def photometric(img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        b = rng.uniform(*CFG["brightness"]); c = rng.uniform(*CFG["contrast"])
        g = torch.tensor(rng.uniform(*CFG["gain"], size=3), dtype=img.dtype)[:, None, None]
        mean = img.mean()
        out = ((img - mean) * c + mean) * b * g
        return out.clamp(0, 1)

    def __call__(self, item: dict, cams: list[str]) -> dict:
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))   # torch is reseeded per worker
        ep = int(item["episode_index"]); fi = int(item["frame_index"])
        if EE_KEY in item and rng.uniform() < CFG["p_relight"]:
            m = self.maps(ep, fi)
            if m is not None:
                x = item[EE_KEY]
                item[EE_KEY] = self.relight(x, m, rng) if x.ndim == 3 else torch.stack([self.relight(t, m, rng) for t in x])
        if rng.uniform() < CFG["p_photo"]:
            for cam in cams:
                if cam in item:
                    x = item[cam]
                    item[cam] = self.photometric(x, rng) if x.ndim == 3 else torch.stack([self.photometric(t, rng) for t in x])
        return item


def install(maps_dir: str | Path | None = None) -> RelightAug | None:
    """Patch LeRobotDataset.__getitem__ so every item is relit before its image_transforms."""
    maps_dir = maps_dir or os.environ.get("RELIGHT_AUG_MAPS")
    if not maps_dir:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    aug = RelightAug(maps_dir)
    orig = LeRobotDataset.__getitem__
    if getattr(orig, "_relight", False):
        return aug

    def __getitem__(self, idx):
        tf = self.image_transforms
        self.image_transforms = None
        try:
            item = orig(self, idx)
        finally:
            self.image_transforms = tf
        cams = list(self.meta.camera_keys)
        item = aug(item, cams)
        if tf is not None:
            for cam in cams:
                item[cam] = tf(item[cam])
        return item

    __getitem__._relight = True
    LeRobotDataset.__getitem__ = __getitem__
    print(f"[relight_aug] installed: maps {maps_dir} J={aug.J} cfg={json.dumps(CFG)}", flush=True)
    return aug
