#!/usr/bin/env python3
"""Pick 132514 frames for the exposure-from-blur measurement.

Projects the recorded cube centre through the 0804 cameras, keeps (frame, camera) pairs where the
production BA actually used that camera, and ranks by image speed among views where the tag is at
least 45 px. Static frames of the same cameras are the control. Writes candidates.json and lists the
episode videos to fetch from Thor into mkv/ as ep<episode>_<cam>.mkv.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import ab_scene as S  # noqa: E402

BLUR = ROOT / 'outputs/rig_target_ab/followups_20260911/blur'
REVIEW = ROOT / 'outputs/metrology/phase5_contact_review_132514'
THOR_EPISODES = 'nvidia@192.168.111.122:lerobot/outputs/datasets/thor_gmsl2_9ch_v1_20260817_132514/episodes'
TAG_M = 0.0555


def vec(seq, dim):
    out = np.full((len(seq), dim), np.nan)
    for k, x in enumerate(seq):
        if x is not None and np.size(x) == dim:
            out[k] = np.asarray(x, float).reshape(dim)
    return out


def project(cam, p):
    T = cam.T_base_cam
    pc = T[:3, :3].T @ (p - T[:3, 3])
    if pc[2] <= 0.05:
        return None, pc[2]
    uv, _ = cv2.fisheye.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
    return uv.reshape(2), pc[2]


def main():
    scene = S.load_scene()
    cams = {c.name: c for c in scene.cameras}
    rows = []
    for hand in ('left', 'right'):
        doc = scene.task['hands'][hand]
        req = np.asarray(doc['requested'], bool)
        ep = np.asarray(doc['episode'])
        fie = np.asarray(doc['frame_in_episode'])
        pos = vec(doc['T_world_cube_position_m'], 3)
        t = np.asarray([np.nan if x is None else x for x in doc['t_s']], float)
        interp = np.asarray([bool(x) for x in doc['interpolated']])
        seen_by = {}
        with (REVIEW / f'src/{hand}_full/marker_rig_ba.{hand}.with_episode.csv').open() as fh:
            for r in csv.DictReader(fh):
                try:
                    seen_by[(int(r['frame_index']), int(r['episode_index']))] = set(json.loads(r['per_camera_rmse_px_json'] or '{}'))
                except (ValueError, json.JSONDecodeError):
                    pass
        for i in range(len(req) - 1):
            j = i + 1
            if not (req[i] and req[j]) or ep[j] != ep[i] or interp[i] or interp[j] or not np.isfinite(pos[[i, j]]).all():
                continue
            seen = seen_by.get((i, int(ep[i])), set())
            speed = float(np.linalg.norm(pos[j] - pos[i]) / (t[j] - t[i]))
            for name, cam in cams.items():
                if name not in seen:
                    continue
                uv0, z = project(cam, pos[i])
                uv1, _ = project(cam, pos[j])
                if uv0 is None or uv1 is None or not (100 < uv0[0] < cam.width - 100 and 100 < uv0[1] < cam.height - 100):
                    continue
                vpx = (uv1 - uv0) / (t[j] - t[i])
                rows.append(dict(hand=hand, idx=i, episode=int(ep[i]), frame=int(fie[i]), cam=name, u=float(uv0[0]),
                                 v=float(uv0[1]), z_m=float(z), tag_px=float(cam.K[0, 0] * TAG_M / z),
                                 vpx=float(np.linalg.norm(vpx)), vdir_deg=float(np.degrees(np.arctan2(vpx[1], vpx[0]))),
                                 speed_mps=speed, dt_s=float(t[j] - t[i])))
    big = [r for r in rows if r['tag_px'] >= 45]
    fast = []
    for r in sorted(big, key=lambda r: -r['vpx']):
        if any(abs(r['idx'] - p['idx']) < 30 and p['cam'] == r['cam'] and p['hand'] == r['hand'] for p in fast):
            continue
        fast.append(r)
        if len(fast) >= 14:
            break
    static = []
    for cam in sorted({p['cam'] for p in fast}):
        static += sorted((r for r in big if r['cam'] == cam and r['vpx'] < 12), key=lambda r: -r['tag_px'])[:2]
    for kind, sel in (('fast', fast), ('static', static)):
        print(f'--- {kind}')
        for r in sel:
            print(f"{r['hand']:5s} ep{r['episode']} f{r['frame']:4d} {r['cam']} tag {r['tag_px']:4.0f} px "
                  f"v {r['vpx']:5.0f} px/s -> L at 4 / 9 ms {r['vpx'] * 0.004:.1f} / {r['vpx'] * 0.009:.1f} px")
    BLUR.mkdir(parents=True, exist_ok=True)
    (BLUR / 'candidates.json').write_text(json.dumps({'fast': fast, 'static': static}, indent=1))
    print(f'mkdir -p {BLUR}/mkv')
    for episode, cam in sorted({(p['episode'], p['cam']) for p in fast + static}):
        print(f'scp {THOR_EPISODES}/episode_{episode:06d}/{cam}.mkv {BLUR}/mkv/ep{episode}_{cam}.mkv')


if __name__ == '__main__':
    main()
