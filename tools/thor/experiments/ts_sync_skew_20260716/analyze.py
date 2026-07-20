import json, glob, os, csv
import numpy as np

ROOT = "/tmp/claude-1000/-home-hanyu-Codes-lerobot/b5149125-e255-4e2a-a296-44c9aebed33d/scratchpad/wp"
FPS = 60.0

def read_csv(path):
    N, sof, sens = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            N.append(int(row["logical_frame_index"]))
            sof.append(int(row["sof_tsc_ns"]))
            sens.append(int(row["sensor_timestamp_ns"]))
    return np.array(N), np.array(sof, dtype=np.float64), np.array(sens, dtype=np.float64)

def linfit(x, y):
    A = np.polyfit(x, y, 1)
    resid = y - np.polyval(A, x)
    return A[0], A[1], resid.std()

def load_box(path):
    samples = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sid = d["sid"]
            samples.setdefault(sid, []).append((d["mcu_ts"], d["wall_s"], d["t_rel_s"], d["data"]))
    return samples

def calib(slist, t0_wall):
    # replicate calibrate_mcu_clock: host = slope*mcu + intercept, fallbacks
    mcu = np.array([s[0] for s in slist], dtype=np.float64)
    wall = np.array([s[1] for s in slist], dtype=np.float64)
    if len(slist) < 10 or not np.any(mcu) or np.var(mcu) == 0:
        return np.array([s[2] for s in slist])  # raw t_rel fallback
    slope, intercept = np.polyfit(mcu, wall, 1)
    res_std = (wall - (slope*mcu+intercept)).std()
    if slope == 0.0 or res_std > 0.05:
        return np.array([s[2] for s in slist])
    cal_wall = slope*mcu + intercept
    return cal_wall - t0_wall

def nearest_idx(times_sorted, t):
    j = np.searchsorted(times_sorted, t)
    if j <= 0: return 0
    if j >= len(times_sorted): return len(times_sorted)-1
    return j-1 if abs(times_sorted[j-1]-t) <= abs(times_sorted[j]-t) else j

print(f"{'episode':44} {'frames':>6} {'dur_s':>6} {'realfps_sof':>11} {'realfps_sens':>12} {'jit_sof_us':>10} | {'delta_mean_ms':>13} {'delta_slope_ms/s':>15} {'delta_span_ms':>13}")
rows_summary=[]
for ds in sorted(glob.glob(f"{ROOT}/water_pouring_*")):
    for ep in sorted(glob.glob(f"{ds}/episode_*")):
        meta = json.load(open(f"{ep}/meta.json"))
        sr = meta["sync_reference"]
        t0_wall = sr["t0_wall_s"]; t0_mono = sr["t0_mono_s"]
        cams = sorted(glob.glob(f"{ep}/cam_*.argus_frame_metadata.csv"))
        # camera-side (cam_00 primary), and cross-cam slope check
        slopes_sof=[]
        prim=None
        for c in cams:
            N, sof, sens = read_csv(c)
            a_sof,_,jit_sof = linfit(N, sof)
            slopes_sof.append(a_sof)
            if c.endswith("cam_00.argus_frame_metadata.csv") or prim is None:
                prim=(N,sof,sens,jit_sof)
        N,sof,sens,jit_sof = prim
        a_sof,b_sof,_ = linfit(N,sof)
        a_sens,b_sens,_ = linfit(N,sens)
        realfps_sof = 1e9/a_sof
        realfps_sens = 1e9/a_sens
        # --- fixed skew delta: true cam capture (monotonic) minus N/fps label ---
        cam_rel_true = sens/1e9 - t0_mono            # true capture time in t0-relative (mono==box realtime origin)
        t_label = N/FPS                              # what the code assigns
        delta = cam_rel_true - t_label
        dslope, dint = np.polyfit(N, delta, 1)
        dur = (N[-1]-N[0])/FPS
        name = f"{os.path.basename(ds)[13:]}/{os.path.basename(ep)}"
        print(f"{name:44} {len(N):6d} {dur:6.2f} {realfps_sof:11.5f} {realfps_sens:12.5f} {jit_sof/1000:10.2f} | {delta.mean()*1000:13.1f} {dslope*FPS*1000:15.4f} {(delta.max()-delta.min())*1000:13.3f}")
        rows_summary.append((name, delta.mean()*1000, t0_wall, t0_mono, sens, N, ep, cam_rel_true, t_label, meta))

# cross-camera slope agreement (last episode)
print("\ncross-camera SOF slope spread (ns/frame) on last episode:", f"{max(slopes_sof)-min(slopes_sof):.1f} ns  -> identical PWM" )
np.save(ROOT+"/_summary.npy", np.array([r[1] for r in rows_summary]))
