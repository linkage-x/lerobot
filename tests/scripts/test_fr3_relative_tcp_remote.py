"""The remote policy (tools/fr3/relative_tcp_remote.py): stub, wire, server gates, and the client end to end."""
# ruff: noqa: N806  -- T: the usual name of a transform

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.fr3 import (  # noqa: E402
    relative_tcp as rt,
    relative_tcp_policy as rtp,
    relative_tcp_remote as rr,
    rollout_relative_tcp as rrt,
)


def _contract() -> rt.PolicyContract:
    cam = rt.CameraInput("cam_10", 1920, 1080, [0, 0, 1920, 1080], 32, 18)
    return rt.PolicyContract(hand="right", fps=60, frame_stride=2, horizon=8, history=2, cameras=[cam],
                             width_name="box_gripper.distance_m", task="stack the cubes")


def _checkpoint(tmp_path: Path, weights: bytes = b"weights-1") -> Path:
    model = tmp_path / "run" / "lerobot" / "checkpoints" / "020000" / "pretrained_model"
    model.mkdir(parents=True)
    _contract().write(model)
    (model / "config.json").write_text(json.dumps({"type": "pi05", "n_action_steps": 4, "chunk_size": 8}))
    (model / "adapter_model.safetensors").write_bytes(weights)
    (model / "adapter_config.json").write_text("{}")
    (model / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_bytes(b"stats")
    (model / rt.PROVENANCE_FILE).write_text(json.dumps({"workspace": None}))
    return model


def _sample(seed: int = 0) -> dict:
    c = _contract()
    rng = np.random.default_rng(seed)
    images = {"cam_10": rng.integers(0, 256, (18, 32, 3), dtype=np.uint8)}
    still = np.repeat(np.eye(4)[None], c.history + 1, axis=0)
    return rtp.observation_sample(c, still, 0.031, images, already_cropped=True)


class _FakeLoaded:
    """What the server holds: a 'policy' whose actions depend on every input bit."""

    def __init__(self, path: Path):
        self.path = path
        self.contract = _contract()
        self.device = "cpu"
        self.seen: list[dict] = []

    def predict(self, sample: dict) -> np.ndarray:
        self.seen.append(sample)
        img = sample["observation.images.cam_10"]
        value = float(sample["observation.state"].sum()) + float(img.sum())
        return np.full((8, 10), value, dtype=np.float32)


def test_a_stub_keeps_everything_but_the_weights_and_names_the_server(tmp_path):
    model = _checkpoint(tmp_path)
    stub = tmp_path / "stub"
    record = rr.make_stub(model, stub, "http://gpu:8795/")
    names = {p.name for p in stub.iterdir()}
    assert "adapter_model.safetensors" not in names
    assert {rt.CONTRACT_FILE, "config.json", "adapter_config.json", rt.PROVENANCE_FILE, rr.REMOTE_FILE,
            "policy_preprocessor_step_3_normalizer_processor.safetensors"} <= names
    assert record["server"] == "http://gpu:8795"
    assert record["fingerprint"] == rr.fingerprint(model)
    assert (record["policy_type"], record["n_action_steps"]) == ("pi05", 4)
    assert rr.is_stub(stub) and not rr.is_stub(model)
    with pytest.raises(ValueError, match="stub already"):
        rr.make_stub(stub, tmp_path / "again", "http://gpu:8795")


def test_the_fingerprint_follows_the_weights(tmp_path):
    a = rr.fingerprint(_checkpoint(tmp_path / "a", b"weights-1"))
    b = rr.fingerprint(_checkpoint(tmp_path / "b", b"weights-2"))
    assert a != b and a == rr.fingerprint(_checkpoint(tmp_path / "c", b"weights-1"))


def test_the_wire_rebuilds_the_policys_input_bit_for_bit():
    c = _contract()
    sample = _sample(3)
    body = rr.encode_sample(c, sample, "fp")
    back, fp = rr.decode_sample(c, rr._unpack(body))
    assert fp == "fp" and back["task"] == "stack the cubes"
    for key in ("observation.state", "observation.images.cam_10"):
        assert back[key].dtype == sample[key].dtype
        assert torch.equal(back[key], sample[key])


def test_the_wire_refuses_a_malformed_request():
    c = _contract()
    arrays = rr._unpack(rr.encode_sample(c, _sample(), "fp"))
    arrays["image.cam_10"] = arrays["image.cam_10"][:10]
    with pytest.raises(ValueError, match="image.cam_10"):
        rr.decode_sample(c, arrays)
    arrays = rr._unpack(rr.encode_sample(c, _sample(), "fp"))
    arrays["state"] = arrays["state"].astype(np.float64)
    with pytest.raises(ValueError, match="state"):
        rr.decode_sample(c, arrays)


def test_pickled_objects_in_a_request_are_never_loaded():
    import io

    buf = io.BytesIO()
    np.savez(buf, state=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        rr._unpack(buf.getvalue())


def test_other_gpu_processes_leaves_out_its_own_and_says_when_it_cannot_tell():
    class Out:
        def __init__(self, rc, stdout):
            self.returncode, self.stdout = rc, stdout

    rows = "111, python, 10226\n222, .venv/bin/python, 9000\n"
    assert rr.other_gpu_processes(222, run=lambda *a, **k: Out(0, rows)) == [{"pid": 111, "name": "python", "mib": 10226}]
    assert rr.other_gpu_processes(222, run=lambda *a, **k: Out(0, "")) == []
    assert rr.other_gpu_processes(222, run=lambda *a, **k: Out(9, "")) is None


def _serve(tmp_path, others, *, exclusive=True, allow=("127.0.0.1",)):
    model = _checkpoint(tmp_path)
    loaded = _FakeLoaded(model)
    state = rr.PolicyServer(loaded, rr.fingerprint(model), allow_from=list(allow), exclusive=exclusive,
                            gpu_probe=lambda pid: others)
    state.poll_gpu()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rr.make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stub = tmp_path / "stub"
    rr.make_stub(model, stub, f"http://127.0.0.1:{httpd.server_address[1]}")
    return httpd, state, loaded, stub


def test_a_stub_predicts_through_the_server_exactly_as_the_server_would_locally(tmp_path):
    httpd, state, loaded, stub = _serve(tmp_path, [])
    try:
        policy = rtp.load_checkpoint(stub)  # a stub loads as a RemotePolicy
        assert isinstance(policy, rr.RemotePolicy)
        assert rtp._config(policy.policy).n_action_steps == 4
        sample = _sample(5)
        actions = policy.predict(sample)
        assert np.array_equal(actions, _FakeLoaded(stub).predict(sample))
        assert torch.equal(loaded.seen[-1]["observation.images.cam_10"], sample["observation.images.cam_10"])
        assert state.status()["predictions"] == 1 and policy.last_server_s >= 0.0
    finally:
        httpd.shutdown()


def test_the_server_refuses_while_another_process_uses_its_gpu(tmp_path):
    others = [{"pid": 2168529, "name": ".venv-fr3/bin/python", "mib": 10226}]
    httpd, state, loaded, stub = _serve(tmp_path, others)
    try:
        policy = rtp.load_checkpoint(stub)
        assert "2168529" in policy.status["gpu_problem"]
        with pytest.raises(rr.RemotePolicyError, match="GPU is shared with pid 2168529"):
            policy.predict(_sample())
        assert loaded.seen == [] and state.refused == 1
    finally:
        httpd.shutdown()


def test_the_server_refuses_when_it_cannot_list_the_gpus_processes(tmp_path):
    httpd, state, loaded, stub = _serve(tmp_path, None)
    try:
        with pytest.raises(rr.RemotePolicyError, match="cannot list"):
            rtp.load_checkpoint(stub).predict(_sample())
    finally:
        httpd.shutdown()


def test_a_shared_gpu_server_predicts_but_says_so(tmp_path):
    httpd, state, loaded, stub = _serve(tmp_path, [{"pid": 1, "name": "x", "mib": 1}], exclusive=False)
    try:
        policy = rtp.load_checkpoint(stub)
        assert policy.status["exclusive"] is False
        policy.predict(_sample())
    finally:
        httpd.shutdown()


def test_a_stub_of_other_weights_is_refused_at_load(tmp_path):
    httpd, state, loaded, stub = _serve(tmp_path, [])
    try:
        remote = json.loads((stub / rr.REMOTE_FILE).read_text())
        remote["fingerprint"] = "0" * 16
        (stub / rr.REMOTE_FILE).write_text(json.dumps(remote))
        with pytest.raises(rr.RemotePolicyError, match="not the weights this stub was made from"):
            rtp.load_checkpoint(stub)
    finally:
        httpd.shutdown()


def test_a_client_not_allowed_is_refused(tmp_path):
    httpd, state, loaded, stub = _serve(tmp_path, [], allow=("192.168.111.122",))
    try:
        with pytest.raises(rr.RemotePolicyError, match="not allowed"):
            rtp.load_checkpoint(stub)
    finally:
        httpd.shutdown()


def test_an_unreachable_server_is_a_remote_policy_error(tmp_path):
    stub = tmp_path / "stub"
    rr.make_stub(_checkpoint(tmp_path), stub, "http://127.0.0.1:9")
    with pytest.raises(rr.RemotePolicyError, match="cannot be reached"):
        rtp.load_checkpoint(stub)


def test_execution_allows_a_remote_policy_the_remote_prediction_time_only():
    class Dist:
        step_m = np.array([0.02])
        step_deg = np.array([3.0])

    class Plan:
        distribution = Dist()

    bus = Path("/dev/shm/lerobot_online_sync")
    local = rrt.execution_limits(Plan(), bus)
    remote = rrt.execution_limits(Plan(), bus, remote=True)
    assert local.predict_s == 0.10
    assert remote.predict_s == rr.REMOTE_PREDICT_S == 0.30
    assert {k: v for k, v in vars(local).items() if k != "predict_s"} == \
        {k: v for k, v in vars(remote).items() if k != "predict_s"}


# ------------------------------------------------------------------ RTC ---


def _postprocessor():
    """A real unnormaliser with quantile stats, as pi0.5's postprocessor has."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep

    q01 = np.array([-0.2, -0.2, -0.3, -1, -1, -1, -1, -1, -1, 0.0], dtype=np.float32)
    q99 = np.array([0.2, 0.25, 0.1, 1, 1, 1, 1, 1, 1, 0.09], dtype=np.float32)
    step = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={"action": {"q01": q01, "q99": q99}})

    class Post:
        steps = [step]

    return Post(), step


def test_a_prefix_is_normalised_with_the_checkpoints_own_action_stats():
    post, step = _postprocessor()
    phys = np.random.default_rng(0).uniform(-0.1, 0.1, (12, 10)).astype(np.float32)
    norm = rtp.normalize_actions(post, phys, "cpu")
    back = step._normalize_action(norm, inverse=True)  # what the policy's postprocessor does to its output
    assert np.allclose(back.numpy(), phys, atol=1e-6)
    assert not np.allclose(norm.numpy(), phys)


def test_the_rtc_prefix_is_the_previous_chunks_poses_re_anchored_at_this_observation():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(1)

    def pose(t, rv):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
        T[:3, 3] = t
        return T

    T_old = pose([0.4, 0.1, 0.3], [0.1, -0.2, 0.3])
    old_actions = np.concatenate([rt.encode(np.stack([pose(rng.normal(0, 0.05, 3), rng.normal(0, 0.1, 3)) for _ in range(32)])),
                                  rng.uniform(0, 0.08, (32, 1))], axis=1)
    world, widths = rt.targets_from_actions(T_old, old_actions)
    T_new = pose([0.42, 0.09, 0.28], [0.12, -0.18, 0.31])  # the TCP at the new observation, 9 steps later
    prefix = rtp.rtc_prefix(T_new, world[9:], widths[9:])
    again, w_again = rt.targets_from_actions(T_new, prefix)  # the new chunk's frame -> the same world poses
    assert prefix.shape == (23, 10)
    assert np.allclose(again, world[9:], atol=1e-6) and np.allclose(w_again, widths[9:], atol=1e-6)
    assert not np.allclose(prefix, old_actions[9:], atol=1e-3)  # the old chunk's own actions are another frame


def test_an_rtc_prefix_is_checked_and_crosses_the_wire_bit_for_bit():
    c = _contract()
    prefix = np.random.default_rng(2).normal(0, 0.05, (5, 10)).astype(np.float32)
    with pytest.raises(ValueError, match="delay"):
        rtp.with_rtc_prefix(_sample(), prefix, delay=6, horizon=4)
    with pytest.raises(ValueError, match="expected"):
        rtp.with_rtc_prefix(_sample(), prefix[:, :9], delay=1, horizon=4)
    sample = rtp.with_rtc_prefix(_sample(), prefix, delay=2, horizon=4)
    back, _ = rr.decode_sample(c, rr._unpack(rr.encode_sample(c, sample, "fp")))
    assert np.array_equal(back[rtp.RTC_PREFIX_KEY], prefix)
    assert (back[rtp.RTC_DELAY_KEY], back[rtp.RTC_HORIZON_KEY]) == (2, 4)
    plain, _ = rr.decode_sample(c, rr._unpack(rr.encode_sample(c, _sample(), "fp")))
    assert rtp.RTC_PREFIX_KEY not in plain


def test_a_policy_without_rtc_refuses_a_prefix():
    class Cfg:
        type = "act"
        rtc_config = None

    class Policy:
        config = Cfg()

        def parameters(self):
            yield torch.zeros(1)

    sample = rtp.with_rtc_prefix(_sample(), np.zeros((4, 10), np.float32), delay=1, horizon=3)
    with pytest.raises(ValueError, match="without real-time chunking"):
        rtp.predict_chunk(Policy(), lambda b: b, lambda a: a, sample)
