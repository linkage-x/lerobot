"""Run the rollout runtime's own startup resolution against the stub and compare with the 10-08 log."""
import sys, time
from pathlib import Path
import numpy as np
import torch
sys.path[:0] = ["src", "."]
import tools.fr3.fr3_act_infer_real_runtime as rt

ckpt = rt.resolve_pretrained_model_dir(sys.argv[1])
cfg = rt.load_train_config(ckpt)
root = rt.resolve_dataset_root(ckpt, cfg, None)
align_root, key = rt.resolve_alignment_dataset_root_and_state_key(root)
ds_meta = rt.load_dataset_metadata(root, cfg.dataset.repo_id)
prompt = rt.resolve_rollout_task_prompt(ds_meta, None)
contract, stats = rt.estimate_dataset_start_pose_contract(align_root, state_key=key)
crops, hw = rt.load_camera_crop_specs(root)
action_names = rt.extract_feature_names(ds_meta.features["action"], rt._DEFAULT_ACTION_NAMES)
print("alignment_state_key", key, "same_root", align_root == root)
print("fps", ds_meta.fps, "prompt", repr(prompt))
print("contract xyz=(%+.6f, %+.6f, %+.6f) quat=(%+.6f, %+.6f, %+.6f, %+.6f)" % tuple(contract))
print("gripper_mean", stats.get("gripper_mean"), stats.get("gripper_source"), "obs", stats.get("observation_gripper_mean"))
print("crops", crops, "source_hw", hw)
print("action_names", action_names, "delta_reference", rt.delta_reference_from_action_names(action_names))
want = np.array([0.381084, -0.135287, 0.096959, 0.999976, 0.002863, -0.006337, 0.000385])
print("max_abs_diff_vs_log", float(np.abs(contract - want).max()))
t = time.time()
policy, pre, post = rt.load_policy_stack(ckpt, ds_meta=ds_meta, device=torch.device("cuda"),
                                         rtc_mode="disabled")
print(f"load_policy_stack ok in {time.time()-t:.0f}s, type {type(policy).__name__}")
print("VERIFY_OK")
