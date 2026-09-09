"""Offline native-planner audit. This program never constructs a robot or BOX.

Uses the latest verified polynomial *knots*, not the obsolete 20260903 IK.
Only planner calls are reused; importing this file has no device side effects.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np

SOURCE_SHA = '7fa69ec0c215de460095a0707d01fc93c850de5474b45d70296db21b6aa4a335'
CORE_SHA = '5c24b4f8452bc4740794db38fb64be7ab95dbda62a357dcdd42799a61ed1ab03'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def knots(ep):
    c = np.asarray(ep['coeff_descending_unit_interval'], float)
    values = np.concatenate((c[:, -1, :], c[-1].sum(axis=0)[None]), axis=0)
    if values.shape != (ep['frames'], 8) or not np.isfinite(values).all():
        raise ValueError('Invalid source knots')
    if not np.allclose(c[:-1].sum(axis=1), c[1:, -1], atol=1e-9, rtol=0):
        raise ValueError('Discontinuous source knot values')
    return values


def chunk_indices(n, size):
    if n < 2 or size < 2:
        raise ValueError('Invalid chunk size')
    i = 0
    while i < n-1:
        end = min(n, i+size)
        yield i, end
        i = end-1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    assert digest(root/'timed_plan.json') == SOURCE_SHA, 'Unexpected calibration/IK source'
    so, = (root/'python/panda_py').glob('_core*.so')
    assert digest(so) == CORE_SHA, 'Unexpected native extension'
    sys.path.insert(0, str(root/'python'))
    from panda_py import _core
    assert Path(_core.__file__).resolve() == so.resolve()
    data = json.loads((root/'timed_plan.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    logging.getLogger().setLevel(logging.CRITICAL)
    results = dict(motion_commands_sent=0, robot_instances=0, gripper_instances=0,
                   input_sha256=SOURCE_SHA, native_sha256=CORE_SHA,
                   hardware_released=False, cases=[])
    for ep in data['episodes']:
        src = knots(ep)
        np.savez_compressed(args.output/f'source_{ep["episode"]}.npz',
                            q=src[:, :7], width=src[:, 7], original_times=ep['original_times_s'])
        for deviation in (0.0, 0.0001, 0.02):
            case = dict(episode=ep['episode'], speed_factor=.01, chunk_size=200,
                        max_deviation_rad=deviation, source_frames=len(src),
                        previous_duration_s=ep['duration_s'], chunks=[])
            started = time.monotonic()
            try:
                for index, (a, b) in enumerate(chunk_indices(len(src), 200)):
                    trajectory = _core.JointTrajectory(src[a:b, :7].tolist(), .01, deviation, 5.)
                    duration = float(trajectory.get_duration())
                    if not 0 < duration <= 600:
                        raise ValueError('Native duration invalid')
                    t = np.linspace(0., duration, int(np.ceil(duration/.001))+1)
                    q = np.array([trajectory.get_joint_positions(float(x)) for x in t])
                    v = np.array([trajectory.get_joint_velocities(float(x)) for x in t])
                    acc = np.array([trajectory.get_joint_accelerations(float(x)) for x in t])
                    jerk_sampled = np.diff(acc, axis=0)/np.diff(t)[:, None]
                    if not np.isfinite(np.r_[q.ravel(),v.ravel(),acc.ravel()]).all():
                        raise ValueError('Nonfinite native plan')
                    name = f'ep{ep["episode"]}_dev{deviation:g}_chunk{index}.npz'
                    np.savez_compressed(args.output/name, t=t, q=q, dq=v, ddq=acc,
                                        source_start=a, source_end=b)
                    case['chunks'].append(dict(source_start=a, source_end=b,
                        duration_s=duration, samples=len(t), artifact=name,
                        max_velocity_rad_s=float(abs(v).max()),
                        max_acceleration_rad_s2=float(abs(acc).max()),
                        sampled_max_jerk_rad_s3=float(abs(jerk_sampled).max()),
                        endpoint_error_rad=float(max(abs(q[0]-src[a,:7]).max(),abs(q[-1]-src[b-1,:7]).max()))))
                case['duration_s'] = sum(x['duration_s'] for x in case['chunks'])
                case['planning_ok'] = True
            except Exception as exc:
                case.update(planning_ok=False, error=f'{type(exc).__name__}: {exc}')
            case['audit_wall_s'] = time.monotonic()-started
            results['cases'].append(case)
            print(json.dumps(case, allow_nan=False), flush=True)
    (args.output/'benchmark.json').write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
