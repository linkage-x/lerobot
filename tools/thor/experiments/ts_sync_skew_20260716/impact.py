import json, glob, os, csv
import numpy as np

ROOT = "/tmp/claude-1000/-home-hanyu-Codes-lerobot/b5149125-e255-4e2a-a296-44c9aebed33d/scratchpad/wp"
FPS = 60.0

def read_cam(path):
    N, sens = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            N.append(int(row["logical_frame_index"]))
            sens.append(int(row["sensor_timestamp_ns"]))
    return np.array(N), np.array(sens, dtype=np.float64)

def load_box(path):
    s = {}
    for line in open(path):
        d = json.loads(line)
        s.setdefault(d["sid"], []).append((d["mcu_ts"], d["wall_s"], d["data"]))
    return s

def calib_trel(slist, t0_wall):
    mcu = np.array([x[0] for x in slist], float)
    wall = np.array([x[1] for x in slist], float)
    if len(slist) < 10 or not np.any(mcu) or np.var(mcu) == 0:
        return wall - t0_wall
    sl, ic = np.polyfit(mcu, wall, 1)
    if sl == 0.0 or (wall-(sl*mcu+ic)).std() > 0.05:
        return wall - t0_wall
    return (sl*mcu+ic) - t0_wall

def series(slist, key, comp=None):
    out=[]
    for _,_,data in slist:
        v = data.get(key)
        if comp is not None and v is not None: v = v[comp]
        out.append(float(v) if v is not None else np.nan)
    return np.array(out)

def sample_error(t_grid, t_current, tsort, vsort):
    # value the pipeline attaches now (nearest to t_current=N/fps) vs corrected (nearest to true capture)
    def nn(t):
        j=np.searchsorted(tsort,t);
        j=np.clip(j,1,len(tsort)-1)
        return vsort[j-1] if abs(tsort[j-1]-t)<=abs(tsort[j]-t) else vsort[j]
    cur=np.array([nn(t) for t in t_current])
    cor=np.array([nn(t) for t in t_grid])
    return cur, cor

CH = [("box_gripper","distance_m",None,"gripper_dist_m",1000.0,"mm"),
      ("box_six_d_force","fxyz_mxyz",2,"force_fz_N",1.0,"N"),
      ("box_touch_left","total_force_0p1N",2,"touchL_totfz_0p1N",1.0,"0.1N")]

print(f"{'episode':40} {'delta_ms':>8} | " + " | ".join(f"{name:>16}" for _,_,_,name,_,_ in CH))
print(f"{'':40} {'':>8} | " + " | ".join(f"{'rms/max '+u:>16}" for *_,u in CH))
for ds in sorted(glob.glob(f"{ROOT}/water_pouring_*")):
    for ep in sorted(glob.glob(f"{ds}/episode_*")):
        meta = json.load(open(f"{ep}/meta.json")); sr = meta["sync_reference"]
        t0_wall, t0_mono = sr["t0_wall_s"], sr["t0_mono_s"]
        N, sens = read_cam(f"{ep}/cam_00.argus_frame_metadata.csv")
        t_current = N/FPS
        t_true = sens/1e9 - t0_mono                 # proposed grid (real capture time)
        delta_ms = np.mean(t_true - t_current)*1000
        box = load_box(f"{ep}/box_sensors.jsonl")
        cells=[]
        for sid,key,comp,name,scale,unit in CH:
            sl = box.get(sid)
            if not sl: cells.append(f"{'n/a':>16}"); continue
            tr = calib_trel(sl, t0_wall)
            v = series(sl, key, comp)*scale
            order=np.argsort(tr); ts=tr[order]; vs=v[order]
            cur,cor = sample_error(t_true, t_current, ts, vs)
            diff=cor-cur
            cells.append(f"{np.sqrt(np.nanmean(diff**2)):7.2f}/{np.nanmax(np.abs(diff)):6.2f}")
        name = f"{os.path.basename(ds)[13:]}/{os.path.basename(ep)[8:]}"
        print(f"{name:40} {delta_ms:8.1f} | " + " | ".join(f"{c:>16}" for c in cells))
