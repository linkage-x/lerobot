#!/usr/bin/env python3
"""Side by side: the formal 4 ms report vs the 9 ms exposure rerun, gripper group."""
import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
A = SIM / 'l3_report.json'
B = Path(sys.argv[1]) if len(sys.argv) > 1 else SIM / 'l3_report_exposure9ms.json'


def r2(x):
    return None if x is None else round(float(x), 2)


def main():
    a, b = json.loads(A.read_text()), json.loads(B.read_text())
    print('exposure', a['config']['exposure_ms'], 'ms x', a['config']['exposure_samples'], '->',
          b['config']['exposure_ms'], 'ms x', b['config']['exposure_samples'])
    print('mounts', a['selection']['mounts'], '->', b['selection']['mounts'])
    for side, r in (('A', a), ('B', b)):
        d, p = r['decision'], r['decision']['primary']
        print(f"{side} ({r['config']['exposure_ms']} ms): formal {d['formal']} numeric {d['numeric']} | delta {p['delta_mm']:+.2f} "
              f"{[r2(c) for c in p['ci95_mm']]} | R0 {p['r0_p95_mm']:.2f} H1 {p['h1_p95_mm']:.2f} H1 rot {p['h1_rotation_p95_deg']:.3f} "
              f"| cov R0 {p['r0_coverage']:.3f} H1 {p['h1_coverage']:.3f} | robust {d.get('nogo_robust_across_levels_and_outputs')}")
    g = 'gripper'
    for lv, la_all in a['groups'][g]['levels'].items():
        lb_all = b['groups'][g]['levels'].get(lv)
        if not lb_all:
            continue
        for out in ('raw', 'smoothed'):
            la, lb = la_all.get(out), lb_all.get(out)
            if not la or not lb:
                continue
            cells = []
            for arm in ('R0', 'H0', 'H1'):
                xa, xb = la['arms'][arm], lb['arms'][arm]
                cells.append(f"{arm} {xa['p95']:.2f}->{xb['p95']:.2f} (cov {xa['coverage']:.3f}->{xb['coverage']:.3f}, "
                             f"rot {xa['rotation_p95']:.2f}->{xb['rotation_p95']:.2f})")
            pa, pb = la['paired']['R0_H1'], lb['paired']['R0_H1']
            print(f"{lv:8s} {out:8s} " + ' | '.join(cells)
                  + f" | d(R0-H1) {pa['delta']:+.2f} {[r2(c) for c in pa['ci']]} -> {pb['delta']:+.2f} {[r2(c) for c in pb['ci']]}"
                  + f" | n {pa['n']} -> {pb['n']}")
    print('checks not ok in B:', [c.get('id') for c in b['checks'] if not c.get('ok')])


if __name__ == '__main__':
    main()
