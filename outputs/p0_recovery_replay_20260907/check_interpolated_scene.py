"""Dense OFFLINE sampling of joint-linear segments, not a CCD proof.

This does not export a new command trajectory or move any waypoint. It uses
the existing mesh checker on diagnostic samples <= 0.004 rad and 0.25 mm apart.
Native/OTG controller interpolation and the physical starting path are NOT
assumed identical to these joint-linear segments.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mesh_checker', HERE / 'check_contact_scene_20260907.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
original_reader = checker.read_contact_ik
COUNTS = {}


def dense_reader(path):
    episodes = original_reader(path)
    out = {}
    for ep, rows in episodes.items():
        seq = []
        for i, a in enumerate(rows):
            if i == len(rows) - 1:
                seq.append(a)
                continue
            b = rows[i + 1]
            steps = max(1, int(np.ceil(np.max(np.abs(b['q'] - a['q'])) / .004)),
                        int(np.ceil(abs(b['width_m'] - a['width_m']) / .00025)))
            for j in range(steps):
                ratio = j / steps
                sample = dict(a)
                sample['q'] = a['q'] * (1-ratio) + b['q'] * ratio
                sample['width_m'] = a['width_m'] * (1-ratio) + b['width_m'] * ratio
                sample['frame_index'] = float(a['frame_index'] + ratio)
                sample['original'] = dict(a['original'])
                for axis in 'xyz':
                    key = f'contact_target_{axis}_m'
                    sample['original'][key] = float(a['original'][key])*(1-ratio) + float(b['original'][key])*ratio
                seq.append(sample)
        out[ep] = seq
        COUNTS[str(ep)] = dict(source_frames=len(rows), diagnostic_samples=len(seq))
    return out


if __name__ == '__main__':
    checker.read_contact_ik = dense_reader
    checker.main()
    with (HERE / 'dense_sampling_manifest.json').open('x') as f:
        json.dump(dict(hardware_ready=False, continuous_collision_certified=False,
                       sampling='Joint-linear and true-width-linear, <=0.004 rad per joint and <=0.25 mm width.',
                       scope='Dense diagnostic samples only. Not a native/OTG or start-transit safety certificate.',
                       target_comparison='Contact FK compared to endpoint Cartesian position interpolation, tolerating at most 1 mm.',
                       episodes=COUNTS), f, indent=2)
