import json, glob, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/tmp/claude-1000/-home-hanyu-Codes-lerobot/b5149125-e255-4e2a-a296-44c9aebed33d/scratchpad/wp"
FPS = 60.0

def read_cam(path):
    N, sens = [], []
    for row in csv.DictReader(open(path)):
        N.append(int(row["logical_frame_index"])); sens.append(int(row["sensor_timestamp_ns"]))
    return np.array(N), np.array(sens, float)

# --- independent domain cross-check: does sensor_timestamp (MONOTONIC) mapped to REALTIME
#     land near the box's first wall_s?  Two arithmetic paths must agree on delta. ---
print("independent cross-check (camera sensor_ts -> REALTIME epoch vs box first wall_s):")
eps=[]
for ds in sorted(glob.glob(f"{ROOT}/water_pouring_*")):
    for ep in sorted(glob.glob(f"{ds}/episode_*")):
        meta=json.load(open(f"{ep}/meta.json")); sr=meta["sync_reference"]
        t0_wall,t0_mono=sr["t0_wall_s"],sr["t0_mono_s"]
        N,sens=read_cam(f"{ep}/cam_00.argus_frame_metadata.csv")
        boot_epoch=t0_wall-t0_mono                      # REALTIME epoch of mono==0
        cam0_realtime=sens[0]/1e9+boot_epoch            # frame0 capture in REALTIME
        box0_wall=None
        for line in open(f"{ep}/box_sensors.jsonl"):
            box0_wall=json.loads(line)["wall_s"]; break
        delta_rel=(sens[0]/1e9-t0_mono)*1000            # path A: mono-relative
        delta_epoch=(cam0_realtime-t0_wall)*1000        # path B: realtime epoch (identical bridge, sanity)
        name=f"{os.path.basename(ds)[13:]}/{os.path.basename(ep)[8:]}"
        eps.append((name,N,sens,t0_mono))
        print(f"  {name:32} deltaA={delta_rel:7.1f}ms  deltaB={delta_epoch:7.1f}ms  "
              f"cam0_realtime-box0_wall={(cam0_realtime-box0_wall)*1000:7.1f}ms")

# --- figure: delta[N] flat lines (constant skew, no drift) ---
fig,ax=plt.subplots(1,1,figsize=(9,5))
for name,N,sens,t0_mono in eps:
    delta=(sens/1e9-t0_mono)-N/FPS
    ax.plot(N, delta*1000, lw=1.2, label=f"{name}  (mean {delta.mean()*1000:.0f}ms)")
ax.axhline(0,color="k",lw=0.8,ls="--")
ax.set_xlabel("logical_frame_index N"); ax.set_ylabel("delta[N] = true_capture - N/fps   (ms)")
ax.set_title("Fixed skew between N/fps grid and real camera capture time (7 water_pouring episodes)\n"
             "flat lines => constant per-episode skew, ~0 drift; current pipeline assumes delta=0")
ax.legend(fontsize=7,loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ROOT+"/skew_delta.png",dpi=110)
print("saved", ROOT+"/skew_delta.png")
