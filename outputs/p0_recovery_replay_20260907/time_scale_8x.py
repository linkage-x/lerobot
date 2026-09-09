"""Stretch timestamps only; no resampling, smoothing or changed joint targets."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    source = HERE / 'contact_ik.right.csv'
    dest = HERE / 'contact_ik.right.time_scaled_8x.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))
    report = []
    for ep in [0, 1]:
        seq = [r for r in rows if int(r['episode_index']) == ep]
        t = np.asarray([float(r['timestamp_s']) for r in seq])
        new_t = t[0] + (t-t[0]) * 8
        q = np.asarray([[float(r[f'fr3_joint{i}']) for i in range(1,8)] for r in seq])
        dt = float(np.median(np.diff(new_t)))
        report.append(dict(episode=ep, frames=len(seq), duration_s=float(new_t[-1]-new_t[0]),
                           effective_sample_rate_hz=1/dt,
                           maximum_discrete_velocity_rad_s=float(np.max(np.abs(np.diff(q,axis=0)/np.diff(new_t)[:,None]))),
                           maximum_discrete_acceleration_rad_s2=float(np.max(np.abs(np.diff(q,n=2,axis=0)/dt**2))),
                           maximum_discrete_jerk_rad_s3=float(np.max(np.abs(np.diff(q,n=3,axis=0)/dt**3)))))
        for row, new in zip(seq, new_t):
            row['timestamp_s'] = repr(float(new))
    with dest.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with source.open() as f:
        old = list(csv.DictReader(f))
    assert all({k:v for k,v in a.items() if k!='timestamp_s'} == {k:v for k,v in b.items() if k!='timestamp_s'} for a,b in zip(old,rows))
    with (HERE / 'time_scaled_8x_manifest.json').open('x') as f:
        json.dump(dict(hardware_ready=False, factor=8, only_timestamps_changed=True,
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                       warning='Diagnostic time scaling only. Do not feed to a launcher that ignores timestamps or assumes 20 Hz. Discrete derivatives do not certify continuous native/OTG dynamics.',
                       episodes=report), f, indent=2)
    print(json.dumps(report))
