import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/hanyu/Codes/lerobot")
from tools.thor.gmsl2 import thor_lerobot_v3 as lr3

EP = Path("/tmp/claude-1000/-home-hanyu-Codes-lerobot/b5149125-e255-4e2a-a296-44c9aebed33d/scratchpad/wp/water_pouring_20260715_102205/episode_000000")
meta = json.load(open(EP/"meta.json")); sr = meta["sync_reference"]
t0_wall, t0_mono = sr["t0_wall_s"], sr["t0_mono_s"]

# reconstruct sensor_samples in the shape _build_episode_rows expects
sensor_samples = {}
for line in open(EP/"box_sensors.jsonl"):
    o = json.loads(line)
    sensor_samples.setdefault(o["sid"], []).append(
        {"t_rel_s": o["t_rel_s"], "wall_s": o["wall_s"], "data": o["data"]}
    )

ft = lr3.camera_frame_times_rel(EP, t0_mono)
print("camera_frame_times_rel: N =", len(ft), " first3 =", [round(x,4) for x in ft[:3]],
      " mean_delta_vs_Nfps_ms =", round(np.mean([ft[i]-i/60 for i in range(len(ft))])*1000, 1))

def build(frame_times):
    return lr3._build_episode_rows(
        fps=60, episode_index=0, snapshots=[], duration_s=meta["duration_s"],
        sensor_samples=sensor_samples, t0_wall_s=t0_wall, frame_times_s=frame_times,
    )

cur = build(None)          # current N/fps
fix = build(ft)            # corrected true-capture-time
gi = lr3.BOX_STATE_NAMES.index("box_gripper.distance_m")
fzi = lr3.BOX_STATE_NAMES.index("box_six_d_force.fz")
g_cur = np.array([r["observation.state"][gi] for r in cur])*1000
g_fix = np.array([r["observation.state"][gi] for r in fix])*1000
f_cur = np.array([r["observation.state"][fzi] for r in cur])
f_fix = np.array([r["observation.state"][fzi] for r in fix])

print(f"rows: current={len(cur)} fixed={len(fix)}")
print(f"timestamp column unchanged? {np.allclose([r['timestamp'] for r in cur],[r['timestamp'] for r in fix])} "
      f"(both == N/fps: {np.allclose([r['timestamp'] for r in cur],[i/60 for i in range(len(cur))])})")
print(f"gripper.distance change  RMS={np.sqrt(np.mean((g_fix-g_cur)**2)):.2f}mm  max={np.max(np.abs(g_fix-g_cur)):.2f}mm")
print(f"six_d_force.fz  change   RMS={np.sqrt(np.mean((f_fix-f_cur)**2)):.2f}N   max={np.max(np.abs(f_fix-f_cur)):.2f}N")
print(f"n frames whose gripper NN pick changed: {(g_fix!=g_cur).sum()}/{len(cur)}")
