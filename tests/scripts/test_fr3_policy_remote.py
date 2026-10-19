#!/usr/bin/env python

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.fr3 import fr3_act_infer_real_runtime as rt
from tools.fr3 import policy_remote
from tools.fr3.policy_remote import LoadedPolicy, PolicyHost, PolicyServerError, RemotePolicyClient


class _FakeChunkPolicy:
    """Deterministic in its inputs, random in its noise -- like a flow policy -- and records its calls."""

    def __init__(self, chunk_size: int = 6, action_dim: int = 3):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.calls: list[dict] = []
        self.resets = 0

    def predict_action_chunk(self, batch, *, inference_delay, prev_chunk_left_over, execution_horizon):
        self.calls.append(
            {
                'inference_delay': inference_delay,
                'execution_horizon': execution_horizon,
                'prev': None if prev_chunk_left_over is None else prev_chunk_left_over.detach().cpu().clone(),
                'task': batch.get('task'),
            }
        )
        state = batch['observation.state']
        image_mean = batch['observation.images.ee'].float().mean()
        bsize = state.shape[0]
        noise = torch.randn(bsize, self.chunk_size, self.action_dim)
        actions = noise + state[:, : self.action_dim].unsqueeze(1) + image_mean
        if prev_chunk_left_over is not None:
            prev = prev_chunk_left_over if prev_chunk_left_over.dim() == 3 else prev_chunk_left_over.unsqueeze(0)
            actions[:, : prev.shape[1]] += prev
        return actions

    def reset(self):
        self.resets += 1


class _Processor:
    def __init__(self, fn=lambda x: x):
        self.fn = fn
        self.resets = 0

    def __call__(self, x):
        return self.fn(x)

    def reset(self):
        self.resets += 1


_SPEC = {'checkpoint': 'outputs/train/run/checkpoints/030000/pretrained_model', 'dataset_root': None, 'rtc_mode': 'auto'}
_FINGERPRINT = {'checkpoint': _SPEC['checkpoint'], 'rtc_enabled': True, 'chunk_size': 6}


def _observation(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        'observation.state': rng.standard_normal(8).astype(np.float32),
        'observation.images.ee': rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8),
    }


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def served():
    policy = _FakeChunkPolicy()
    preprocessor = _Processor()
    postprocessor = _Processor(lambda a: a * 2.0)
    loads: list[dict] = []
    releases: list[str] = []

    def predict(observation, **kwargs):
        return rt.predict_action_chunk_for_rollout(
            observation, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, **kwargs
        )

    def reset():
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    def loader(spec):
        loads.append(dict(spec))
        if spec['checkpoint'].endswith('missing'):
            raise FileNotFoundError(f"{spec['checkpoint']} is not on this policy server")
        return LoadedPolicy(
            spec=dict(spec),
            fingerprint={**_FINGERPRINT, 'checkpoint': spec['checkpoint']},
            device=torch.device('cpu'),
            use_amp=False,
            predict=predict,
            reset=reset,
            release=lambda: releases.append(spec['checkpoint']),
        )

    logs: list[str] = []
    clock = _Clock()
    host = PolicyHost(loader, idle_unload_s=60.0, log=logs.append, clock=clock)
    listener = socket.create_server(('127.0.0.1', 0))
    port = listener.getsockname()[1]

    def serve(conn, peer):
        with conn:
            policy_remote.serve_connection(conn, host=host, peer=peer, log=logs.append)

    def loop():
        while True:
            try:
                conn, (addr, peer_port) = listener.accept()
            except OSError:
                return
            threading.Thread(target=serve, args=(conn, f'{addr}:{peer_port}'), daemon=True).start()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield SimpleNamespace(
        address=f'127.0.0.1:{port}', policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
        predict=predict, logs=logs, loads=loads, releases=releases, host=host, clock=clock,
    )
    listener.close()


def _client(address: str, fingerprint: dict | None = None, spec: dict | None = None) -> RemotePolicyClient:
    spec = spec or _SPEC
    client = RemotePolicyClient(
        address,
        config=SimpleNamespace(use_amp=False),
        fingerprint=fingerprint or {**_FINGERPRINT, 'checkpoint': spec['checkpoint']},
        load_spec=spec,
        request_timeout_s=10.0,
        load_timeout_s=10.0,
    )
    client.connect()
    return client


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, 'condition not reached'
        time.sleep(0.01)


def _rollout_kwargs(**overrides) -> dict:
    kwargs = dict(
        device=torch.device('cpu'),
        use_amp=False,
        inference_delay=3,
        prev_chunk_left_over=None,
        execution_horizon=4,
        task='insert the peg',
        robot_type='fr3',
    )
    kwargs.update(overrides)
    return kwargs


def test_parse_policy_server_address_defaults_the_port():
    assert policy_remote.parse_policy_server_address('192.168.1.179') == ('192.168.1.179', 8790)
    assert policy_remote.parse_policy_server_address('192.168.1.179:9001') == ('192.168.1.179', 9001)
    with pytest.raises(ValueError):
        policy_remote.parse_policy_server_address(':9001')


def test_observation_round_trip_keeps_numpy_dtypes_and_strings():
    observation = {**_observation(), 'task': 'insert', 'already_tensor': torch.arange(3)}
    decoded = policy_remote.decode_observation(
        policy_remote._decode(policy_remote._encode(policy_remote.encode_observation(observation)))
    )
    assert decoded['task'] == 'insert'
    for key in ('observation.state', 'observation.images.ee'):
        assert isinstance(decoded[key], np.ndarray)
        assert decoded[key].dtype == observation[key].dtype
        np.testing.assert_array_equal(decoded[key], observation[key])
    assert torch.equal(decoded['already_tensor'], observation['already_tensor'])


def test_remote_chunk_equals_in_process_chunk_under_the_same_seed(served):
    client = _client(served.address)
    prev = torch.randn(4, 3)

    torch.manual_seed(7)
    local = rt.predict_action_chunk_for_rollout(
        _observation(), policy=served.policy, preprocessor=served.preprocessor, postprocessor=served.postprocessor,
        **_rollout_kwargs(prev_chunk_left_over=prev),
    )
    torch.manual_seed(7)
    remote = rt.predict_action_chunk_for_rollout(
        _observation(), policy=client, preprocessor=None, postprocessor=None, **_rollout_kwargs(prev_chunk_left_over=prev)
    )

    assert torch.equal(local[0], remote[0])
    assert torch.equal(local[1], remote[1])
    local_call, remote_call = served.policy.calls
    assert remote_call['inference_delay'] == 3 and remote_call['execution_horizon'] == 4
    assert remote_call['task'] == local_call['task']
    assert torch.equal(remote_call['prev'], prev)
    assert client.last_server_latency_s > 0.0


def test_remote_sampling_path_selects_the_same_draw(served):
    client = _client(served.address)
    kwargs = _rollout_kwargs(
        action_samples=4, lateral_action_indices=(0, 1), gripper_action_index=2, action_sample_horizon=3, action_sample_offset=1
    )
    torch.manual_seed(11)
    local = rt.predict_action_chunk_for_rollout(
        _observation(), policy=served.policy, preprocessor=served.preprocessor, postprocessor=served.postprocessor, **kwargs
    )
    torch.manual_seed(11)
    remote = rt.predict_action_chunk_for_rollout(_observation(), policy=client, preprocessor=None, postprocessor=None, **kwargs)
    assert torch.equal(local[1], remote[1])


def test_reset_reaches_the_served_policy_and_processors(served):
    client = _client(served.address)
    client.reset()
    assert (served.policy.resets, served.preprocessor.resets, served.postprocessor.resets) == (1, 1, 1)
    client.close()
    # A fresh client is a fresh rollout: the hold ends with one more reset.
    _wait_until(lambda: served.policy.resets == 2)


def test_fingerprint_mismatch_is_refused_at_connect(served):
    with pytest.raises(PolicyServerError, match='policy mismatch.*chunk_size'):
        _client(served.address, fingerprint={**_FINGERPRINT, 'chunk_size': 50})


def test_requests_before_hello_are_refused(served):
    host, port = policy_remote.parse_policy_server_address(served.address)
    with socket.create_connection((host, port), timeout=5) as sock:
        policy_remote.send_message(sock, {'op': 'reset'})
        reply = policy_remote.recv_message(sock)
    assert reply['ok'] is False and 'before a matching hello' in reply['error']
    assert served.policy.resets == 0


def test_server_side_failure_surfaces_and_the_connection_survives(served):
    client = _client(served.address)
    bad = _observation()
    del bad['observation.images.ee']
    with pytest.raises(PolicyServerError, match='KeyError'):
        rt.predict_action_chunk_for_rollout(bad, policy=client, preprocessor=None, postprocessor=None, **_rollout_kwargs())
    original, processed = rt.predict_action_chunk_for_rollout(
        _observation(), policy=client, preprocessor=None, postprocessor=None, **_rollout_kwargs()
    )
    assert tuple(processed.shape) == (6, 3)


def test_non_tensor_payloads_are_refused_by_the_weights_only_loader(served):
    client = _client(served.address)
    with pytest.raises(PolicyServerError, match='UnpicklingError|Weights only load failed'):
        client._request({'op': 'predict', 'observation': SimpleNamespace(evil=True), 'kwargs': {}})
    client.reset()  # the stream is still in step


def test_the_server_loads_what_the_hello_names_and_keeps_it(served):
    _client(served.address).close()
    _wait_until(lambda: served.host.status()['holder'] is None)
    _client(served.address).close()
    _wait_until(lambda: served.host.status()['holder'] is None)
    assert served.loads == [_SPEC]

    other = {**_SPEC, 'rtc_execution_horizon': 20}
    _client(served.address, spec=other)
    assert served.loads == [_SPEC, other]
    assert served.releases == [_SPEC['checkpoint']]


def test_a_second_client_is_refused_while_the_first_holds_the_policy(served):
    first = _client(served.address)
    with pytest.raises(PolicyServerError, match='in use by'):
        _client(served.address)
    first.close()
    _wait_until(lambda: served.host.status()['holder'] is None)
    _client(served.address)


def test_status_and_unload_need_no_hello_and_never_take_the_policy(served):
    client = _client(served.address)
    status = policy_remote.query_policy_server(served.address)['status']
    assert status['loaded']['spec'] == _SPEC and status['holder'] is not None
    with pytest.raises(PolicyServerError, match='stop that rollout first'):
        policy_remote.query_policy_server(served.address, 'unload')
    original, processed = rt.predict_action_chunk_for_rollout(
        _observation(), policy=client, preprocessor=None, postprocessor=None, **_rollout_kwargs()
    )
    assert tuple(processed.shape) == (6, 3)
    client.close()
    _wait_until(lambda: served.host.status()['holder'] is None)
    assert policy_remote.query_policy_server(served.address, 'unload')['unloaded'] is True
    assert policy_remote.query_policy_server(served.address)['status']['loaded'] is None


def test_idle_policy_is_unloaded_after_the_timeout(served):
    _client(served.address).close()
    _wait_until(lambda: served.host.status()['holder'] is None)
    served.clock.now += 59.0
    assert served.host.unload_if_idle() is False
    served.clock.now += 2.0
    assert served.host.unload_if_idle() is True
    assert served.releases == [_SPEC['checkpoint']]


def test_a_failed_load_is_reported_and_frees_the_server(served):
    with pytest.raises(PolicyServerError, match='not on this policy server'):
        _client(served.address, spec={**_SPEC, 'checkpoint': 'outputs/train/missing'})
    assert served.host.status()['holder'] is None
    _client(served.address)


def test_load_spec_names_paths_from_outputs_and_the_server_refuses_anything_else(tmp_path):
    ckpt = tmp_path / 'repo' / 'outputs' / 'train' / 'job' / 'checkpoints' / '030000' / 'pretrained_model'
    view = tmp_path / 'repo' / 'outputs' / 'exports' / 'training_views' / 'v'
    spec = policy_remote.policy_load_spec(ckpt, dataset_root=view, rtc_mode='auto', rtc_execution_horizon=16)
    assert spec['checkpoint'] == 'outputs/train/job/checkpoints/030000/pretrained_model'
    assert spec['dataset_root'] == 'outputs/exports/training_views/v'
    # Omitted overrides stay omitted, so both sides fall back to the same defaults.
    assert 'n_action_steps_override' not in spec and spec['rtc_execution_horizon'] == 16
    with pytest.raises(ValueError, match='not policy overrides'):
        policy_remote.policy_load_spec(ckpt, dataset_root=None, device='cuda')
    with pytest.raises(ValueError, match='not under an outputs/'):
        policy_remote.policy_load_spec(tmp_path / 'elsewhere', dataset_root=None)

    root = tmp_path / 'server'
    assert policy_remote.resolve_outputs_path(root, spec['checkpoint']) == root / spec['checkpoint']
    for bad in ('/etc/passwd', 'outputs/../../etc', 'src/lerobot', ''):
        with pytest.raises(ValueError):
            policy_remote.resolve_outputs_path(root, bad)


def test_remote_stack_refuses_checkpoints_without_rtc(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, 'prepare_rollout_policy_config', lambda *_a, **_k: SimpleNamespace(type='act', rtc_config=None))
    with pytest.raises(SystemExit, match='only serves the RTC chunk path'):
        rt.load_remote_policy_stack(tmp_path, policy_server='127.0.0.1:1', rtc_mode='auto')


@pytest.mark.parametrize('error_type', [ValueError, KeyboardInterrupt, SystemExit])
def test_runtime_releases_remote_policy_on_startup_failure(served, monkeypatch, tmp_path, error_type):
    client = _client(served.address)
    client.config.input_features = {}
    monkeypatch.setattr(rt, 'resolve_pretrained_model_dir', lambda *_a: tmp_path)
    monkeypatch.setattr(rt, 'load_train_config', lambda *_a: SimpleNamespace(dataset=SimpleNamespace(repo_id='test')))
    monkeypatch.setattr(rt, 'resolve_dataset_root', lambda *_a: tmp_path)
    monkeypatch.setattr(rt, 'resolve_alignment_dataset_root_and_state_key', lambda *_a: (tmp_path, 'observation.state'))
    monkeypatch.setattr(rt, 'load_dataset_metadata', lambda *_a: SimpleNamespace())
    monkeypatch.setattr(rt, 'resolve_rollout_task_prompt', lambda *_a: 'test')
    monkeypatch.setattr(rt, 'estimate_dataset_start_pose_contract', lambda *_a, **_k: (None, None))
    monkeypatch.setattr(rt, 'load_camera_configs', lambda *_a: {})
    monkeypatch.setattr(rt, 'load_camera_crop_specs', lambda *_a: ({}, {}))
    monkeypatch.setattr(rt, 'load_remote_policy_stack', lambda *_a, **_k: (client, None, None))

    def fail_validation(**_kwargs):
        raise error_type('startup failed after connecting')

    monkeypatch.setattr(rt, 'validate_camera_keys', fail_validation)
    try:
        # Keep the exception/traceback and client alive: release must not rely on garbage collection.
        with pytest.raises(error_type, match='startup failed') as error:
            rt.run_inference(rt.parse_args(['--policy-server', served.address]))
        assert error.value is not None
        _wait_until(lambda: served.host.status()['holder'] is None)
        assert served.policy.resets == 1
        _client(served.address).close()
        assert served.loads == [_SPEC]
    finally:
        client.close()


def test_policy_server_flag_defaults_off():
    assert rt.parse_args([]).policy_server is None
    assert rt.parse_args(['--policy-server', '192.168.1.179']).policy_server == '192.168.1.179'


def test_server_side_processors_reset_quietly_and_refuse_to_run():
    processor = policy_remote.ServerSideProcessor('postprocessor')
    processor.reset()
    with pytest.raises(RuntimeError, match='postprocessor runs on the policy server'):
        processor(torch.zeros(1))


def test_a_rollout_arriving_during_a_warm_up_waits_for_it_and_reuses_the_load(served):
    gate = threading.Event()
    loader = served.host._loader

    def slow_loader(spec):
        gate.wait(5.0)
        return loader(spec)

    served.host._loader = slow_loader
    warmed: list = []
    warm = threading.Thread(target=lambda: warmed.append(served.host.warm(_SPEC)))
    warm.start()
    _wait_until(lambda: served.host.status()['loading'] is not None)
    connected: list = []
    client = threading.Thread(target=lambda: connected.append(_client(served.address)))
    client.start()
    time.sleep(0.2)
    assert not connected, 'the hello must wait for the warm-up, not be refused or overtake it'
    gate.set()
    warm.join(5.0)
    client.join(5.0)
    assert warmed and warmed[0] is not None and connected
    assert served.loads == [_SPEC]
    connected[0].close()


def test_a_warm_up_yields_to_a_rollout_that_already_holds_the_policy(served):
    client = _client(served.address)
    assert served.host.warm({**_SPEC, 'rtc_execution_horizon': 20}) is None
    assert served.loads == [_SPEC]
    client.close()


def test_code_manifest_hashes_each_file_once_and_names_the_process(tmp_path, monkeypatch):
    import json
    import os
    import sys

    from tools.fr3 import fr3_policy_server as server

    module_file = tmp_path / 'pkg' / 'mod.py'
    module_file.parent.mkdir()
    module_file.write_text('X = 1\n')
    monkeypatch.setitem(sys.modules, '_manifest_probe', SimpleNamespace(__file__=str(module_file)))
    manifest = server.CodeManifest(tmp_path, Path('run/logs/code.json'))
    manifest.update()
    record = json.loads((tmp_path / 'run/logs/code.json').read_text())
    assert record['pid'] == os.getpid()
    first = record['files']['pkg/mod.py']

    module_file.write_text('X = 2\n')
    manifest.update()
    assert json.loads((tmp_path / 'run/logs/code.json').read_text())['files']['pkg/mod.py'] == first


def test_policy_server_ensure_keeps_a_server_running_unchanged_code(tmp_path):
    import hashlib
    import json
    import os
    import re
    import subprocess

    script = (Path(__file__).resolve().parents[2] / 'run/policy_server.sh').read_text()
    body = re.search(r"code_current\(\) \{\n  python3 - \"\$1\" \"\$code_manifest\" <<'PY'\n(.*?)\nPY\n", script, re.S).group(1)
    (tmp_path / 'a.py').write_text('A = 1\n')
    manifest = tmp_path / 'code.json'
    digest = hashlib.sha256(b'A = 1\n').hexdigest()
    manifest.write_text(json.dumps({'pid': 4242, 'files': {'a.py': digest}}))

    def check(pid: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            ['python3', '-', str(pid), str(manifest)], input=body, text=True, capture_output=True, cwd=tmp_path
        )

    assert check(4242).returncode == 0
    assert 'another server process' in check(os.getpid()).stdout
    (tmp_path / 'a.py').write_text('A = 2\n')
    result = check(4242)
    assert result.returncode == 1 and 'a.py' in result.stdout


def test_warm_forward_runs_an_unguided_then_a_guided_pass_and_resets():
    from lerobot.configs.types import FeatureType

    from tools.fr3 import fr3_policy_server as server

    policy = _FakeChunkPolicy(chunk_size=12)
    pre, post = _Processor(), _Processor()
    stack = server._PolicyStack(policy, pre, post)
    cfg = SimpleNamespace(
        input_features={
            'observation.state': SimpleNamespace(type=FeatureType.STATE, shape=(8,)),
            'observation.images.ee': SimpleNamespace(type=FeatureType.VISUAL, shape=(3, 12, 16)),
        },
        rtc_config=SimpleNamespace(execution_horizon=5),
        use_amp=False,
    )
    lines: list[str] = []
    server.warm_forward(stack, cfg, torch.device('cpu'), task='insert', log=lines.append)

    assert [call['inference_delay'] for call in policy.calls] == [0, server.WARM_FORWARD_DELAY_STEPS]
    assert policy.calls[0]['prev'] is None
    assert policy.calls[1]['prev'].shape == (12 - server.WARM_FORWARD_DELAY_STEPS, 3)
    assert all(call['execution_horizon'] == 5 and call['task'] == 'insert' for call in policy.calls)
    # Nothing of the blank pass may carry into the first rollout's chunk.
    assert policy.resets == pre.resets == post.resets == 1
    assert lines and 'warm_forward' in lines[0]
