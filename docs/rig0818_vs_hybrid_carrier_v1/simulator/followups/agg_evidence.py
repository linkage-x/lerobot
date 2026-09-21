#!/usr/bin/env python3
"""Pool the evidence-level gripper re-smoothing over sessions: p95 per arm and paired delta(R0 - H1).

Run smoother_ablation.py --level evidence --session N --groups gripper --only prod_copy,right for
N = 0..23 first.
"""
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[4] / 'outputs/rig_target_ab/followups_20260911/smoother_ablation_v2/evidence'
ARMS = ('R0', 'H0', 'H1')


def blocks(variant):
    src = 'prod_copy' if variant == 'raw' else variant
    out = {}
    for sdir in sorted(OUT.glob('[0-9][0-9][0-9]')):
        for arm in ARMS:
            for f in sorted(sdir.glob(f'solve__{src}__gripper__{arm}__*.pkl')):
                with f.open('rb') as fh:
                    r = pickle.load(fh)
                key = (sdir.name, r['spec'][4], r['spec'][5])
                out.setdefault(key, {})[arm] = np.linalg.norm(r['raw_evec'] if variant == 'raw' else r['evec'], axis=1)
    return {k: v for k, v in out.items() if all(a in v for a in ARMS)}


def main():
    rng = np.random.default_rng(0)
    print('gripper / evidence, pooled over sessions; bootstrap over (session, hand, episode) blocks, paired')
    for variant in ('raw', 'prod_copy', 'right'):
        b = blocks(variant)
        keys = sorted(b)
        if not keys:
            print(variant, 'no data')
            continue
        p95 = {a: float(np.percentile(np.concatenate([b[k][a] for k in keys]), 95)) for a in ARMS}
        boot = []
        for _ in range(1000):
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            boot.append(np.percentile(np.concatenate([b[k]['R0'] for k in pick]), 95)
                        - np.percentile(np.concatenate([b[k]['H1'] for k in pick]), 95))
        sessions = len({k[0] for k in keys})
        print(f"{variant:10s} R0 {p95['R0']:.2f}  H0 {p95['H0']:.2f}  H1 {p95['H1']:.2f}  | delta(R0-H1) "
              f"{p95['R0'] - p95['H1']:+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"
              f"  | sessions {sessions}, blocks {len(keys)}")
    print('report (production smoother): R0 4.80, H1 7.97, delta -3.17 [-4.12, -2.40]')


if __name__ == '__main__':
    main()
