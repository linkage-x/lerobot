"""Same checkpoint, same seed: in-process chunk vs. a chunk fetched through the policy server.

Loads the real stack once on the GPU, serves it on a loopback port from this process, and
connects a `load_remote_policy_stack` client to it -- so the handshake compares the client's
CPU-side config against the server's loaded one, exactly as the NUC would. No hardware.

    PYTHONPATH=src:. .venv-fr3/bin/python notes/2026-10-15_nuc_split/remote_consistency_probe.py <ckpt>/checkpoints/030000
"""
import socket
import sys
import threading
import time

import numpy as np
import torch

from tools.fr3 import fr3_act_infer_real_runtime as rt
from tools.fr3.policy_remote import policy_fingerprint, serve_connection

ckpt = rt.resolve_pretrained_model_dir(sys.argv[1])
train_cfg = rt.load_train_config(ckpt)
ds_meta = rt.load_dataset_metadata(rt.resolve_dataset_root(ckpt, train_cfg, None), train_cfg.dataset.repo_id)
overrides = dict(rtc_mode='auto')
device = torch.device('cuda')
policy, pre, post = rt.load_policy_stack(ckpt, ds_meta=ds_meta, device=device, **overrides)
fingerprint = policy_fingerprint(policy.config, ckpt)
print('server fingerprint', fingerprint)

listener = socket.create_server(('127.0.0.1', 0))
port = listener.getsockname()[1]


def predict(observation, **kwargs):
    return rt.predict_action_chunk_for_rollout(observation, policy=policy, preprocessor=pre, postprocessor=post, **kwargs)


def reset():
    policy.reset(); pre.reset(); post.reset()


def loop():
    conn, _ = listener.accept()
    with conn:
        serve_connection(conn, log=lambda _m: None, predict=predict, reset=reset, fingerprint=fingerprint,
                         device=device, use_amp=bool(policy.config.use_amp))


threading.Thread(target=loop, daemon=True).start()
client, _cpre, _cpost = rt.load_remote_policy_stack(ckpt, policy_server=f'127.0.0.1:{port}', **overrides)

rng = np.random.default_rng(0)
obs = {'observation.state': rng.standard_normal(8).astype(np.float32)}
for key, ft in policy.config.input_features.items():
    if key.startswith('observation.images'):
        c, h, w = ft.shape
        obs[key] = rng.integers(0, 255, size=(h, w, c), dtype=np.uint8)
common = dict(device=device, use_amp=bool(policy.config.use_amp), execution_horizon=int(policy.config.rtc_config.execution_horizon),
              task='insert the peg', robot_type='franka_research3')


def both(**kw):
    out = []
    for target, p, pp in ((policy, pre, post), (client, None, None)):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.synchronize(); t = time.perf_counter()
        o, a = rt.predict_action_chunk_for_rollout(obs, policy=target, preprocessor=p, postprocessor=pp, **common, **kw)
        torch.cuda.synchronize(); out.append((o.cpu(), a.cpu(), time.perf_counter() - t))
    (lo, la, lt), (ro, ra, rtt) = out
    return lo, la, lt, ro, ra, rtt


for _ in range(2):  # warm-up
    both(inference_delay=0, prev_chunk_left_over=None)
lo, la, lt, ro, ra, rtt = both(inference_delay=0, prev_chunk_left_over=None)
print(f'bootstrap  orig_maxdiff={float((lo - ro).abs().max()):.3e} proc_maxdiff={float((la - ra).abs().max()):.3e} '
      f'local={lt:.3f}s remote={rtt:.3f}s server={client.last_server_latency_s:.3f}s shape={tuple(ra.shape)}')
prev = lo[10:]
lo2, la2, lt2, ro2, ra2, rtt2 = both(inference_delay=6, prev_chunk_left_over=prev)
print(f'rtc_prefix orig_maxdiff={float((lo2 - ro2).abs().max()):.3e} proc_maxdiff={float((la2 - ra2).abs().max()):.3e} '
      f'local={lt2:.3f}s remote={rtt2:.3f}s server={client.last_server_latency_s:.3f}s')
client.reset()
exact = all(torch.equal(x, y) for x, y in ((lo, ro), (la, ra), (lo2, ro2), (la2, ra2)))
print('EXACT_MATCH' if exact else 'NOT_EXACT')
