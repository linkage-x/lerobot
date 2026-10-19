"""Robot-host side of the split, without the robot: connect to a policy server and time real requests.

Loads only the config and processors (CPU), builds a synthetic observation at the policy's
input shapes, and chains RTC prefixes the way the rollout's queue would. Nothing here opens a
camera or the arm.

    PYTHONPATH=src:. .venv-fr3/bin/python notes/2026-10-15_nuc_split/remote_client_probe.py <ckpt>/checkpoints/030000 HOST[:PORT] [N]
"""
import sys
import time

import numpy as np

from tools.fr3 import fr3_act_infer_real_runtime as rt

ckpt = rt.resolve_pretrained_model_dir(sys.argv[1])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
client, _pre, _post = rt.load_remote_policy_stack(ckpt, policy_server=sys.argv[2], rtc_mode='auto')
cfg = client.config
rng = np.random.default_rng(0)
obs = {'observation.state': rng.standard_normal(8).astype(np.float32)}
for key, ft in cfg.input_features.items():
    if key.startswith('observation.images'):
        c, h, w = ft.shape
        obs[key] = rng.integers(0, 255, size=(h, w, c), dtype=np.uint8)
payload_mb = sum(v.nbytes for v in obs.values()) / 1e6
client.reset()
prev = None
rows = []
for i in range(n):
    t = time.perf_counter()
    original, processed = rt.predict_action_chunk_for_rollout(
        obs, policy=client, device=None, preprocessor=None, postprocessor=None, use_amp=False,
        inference_delay=0 if prev is None else 6, prev_chunk_left_over=prev,
        execution_horizon=int(cfg.rtc_config.execution_horizon), task='insert the peg', robot_type='franka_research3',
    )
    total = time.perf_counter() - t
    rows.append((total, client.last_server_latency_s))
    prev = original[10:]
    print(f'req {i:2d} total={total:.3f}s server={client.last_server_latency_s:.3f}s net+codec={total - client.last_server_latency_s:.3f}s shape={tuple(processed.shape)}')
steady = rows[2:]
print(f'payload={payload_mb:.2f}MB steady total median={np.median([r[0] for r in steady]):.3f}s max={max(r[0] for r in steady):.3f}s '
      f'net+codec median={np.median([r[0] - r[1] for r in steady]):.3f}s max={max(r[0] - r[1] for r in steady):.3f}s')
client.close()
print('CLIENT_PROBE_OK')
