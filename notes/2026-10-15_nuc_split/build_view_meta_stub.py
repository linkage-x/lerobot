"""Rebuild a metadata-only stand-in for the funnel105 training view, whose disk died with tele.

Every value below was printed by fr3_act_infer_real_runtime.py when this exact checkpoint
(…plus_funnel105_030000__pi05__20260930_053656/checkpoints/030000) started a rollout on
2026-10-08 (grasp_20261008_101050). One episode whose first frame *is* the logged start contract
makes the runtime's per-episode mean reproduce it exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

VIEW = "fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000"
repo_root = Path(sys.argv[1]).resolve()
root = repo_root / "outputs/exports/training_views" / VIEW

TASK = "Pick up the peg and insert it fully into the hole"
FPS = 30
START_XYZ = [0.381084, -0.135287, 0.096959]
START_QUAT_XYZW = [0.999976, 0.002863, -0.006337, 0.000385]
START_ACTION_GRIPPER = 0.810  # dataset_start_spread gripper_mean, source=action.gripper.pos
START_OBS_GRIPPER = 0.795  # observation_gripper_mean
CROPS = {"observation.images.ee": [58, 194, 542, 286], "observation.images.side": [108, 58, 444, 382]}
SOURCE_HW = (480, 640)
STATE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
ACTION_NAMES = [
    "delta_ee_from_prev_cmd.dx",
    "delta_ee_from_prev_cmd.dy",
    "delta_ee_from_prev_cmd.dz",
    "delta_ee_from_prev_cmd.drz",
    "gripper.pos",
]

features = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (5,), "names": ACTION_NAMES},
}
for key, (_, _, w, h) in CROPS.items():
    features[key] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

ds = LeRobotDataset.create(
    repo_id=f"local/{VIEW}", fps=FPS, features=features, root=root,
    robot_type="franka_research3_il_view", use_videos=True,
)
quat = np.asarray(START_QUAT_XYZW, dtype=np.float64)
quat /= np.linalg.norm(quat)
frame = {
    "observation.state": np.asarray([*START_XYZ, *quat, START_OBS_GRIPPER], dtype=np.float32),
    "action": np.asarray([0.0, 0.0, 0.0, 0.0, START_ACTION_GRIPPER], dtype=np.float32),
    "task": TASK,
}
for key, (_, _, w, h) in CROPS.items():
    frame[key] = np.zeros((h, w, 3), dtype=np.uint8)
ds.add_frame(frame)
ds.save_episode()
ds.finalize()

# The crops are drawn on the recording's full frame; the runtime reads that (H, W) off a source
# dataset's info.json, so give it one that holds nothing else.
source_stub = root / "source_frame_stub"
(source_stub / "meta").mkdir(parents=True, exist_ok=True)
(source_stub / "meta/info.json").write_text(json.dumps({
    "features": {key: {"dtype": "video", "shape": [*SOURCE_HW, 3], "names": ["height", "width", "channels"]}
                 for key in CROPS},
}, indent=2))

manifest = {
    "recovered_stub": {
        "why": "the real view was lost with the tele workstation disk (2026-10); metadata-only stand-in, "
               "NOT trainable, NOT a dataset",
        "values_from": "rollout startup log of checkpoints/030000, 2026-10-08 grasp_20261008_101050",
        "original_start_contract_episodes": 263,
        "original_start_spread_xyz_std_mm": [57.18, 94.52, 162.19],
    },
    "source_dataset_root": None,
    "source_dataset_roots": [str(source_stub)],
    "repo_id": f"local/{VIEW}",
    "cameras": list(CROPS),
    "camera_crop_specs": CROPS,
    "task_prompts": [TASK],
    "state_dim": 8,
    "action_dim": 5,
    "fps": FPS,
    "action_mode": "delta_ee_from_prev_cmd",
}
(root / "meta/il_view_manifest.json").write_text(json.dumps(manifest, indent=2))
print("STUB_OK", root)
