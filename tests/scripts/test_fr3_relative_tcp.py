# ruff: noqa: N803, N806  -- T, R, A: transforms and rotations
from __future__ import annotations

import numpy as np
import pytest

from tools.fr3 import relative_tcp as rt, relative_tcp_data as rd


def _random_pose(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return np.concatenate([rng.normal(scale=0.3, size=3), q])


def _contract(**kw) -> rt.PolicyContract:
    cam = rt.CameraInput("cam_10", 1920, 1080, [0, 0, 1920, 1080], 320, 180)
    base = {"hand": "right", "fps": 60, "frame_stride": 2, "horizon": 4, "history": 2, "cameras": [cam],
                "width_name": "box_gripper.distance_m"}
    return rt.PolicyContract(**{**base, **kw})


def test_relative_quantities_do_not_depend_on_the_world_frame():
    """The reason for the representation: camera world on one side, the arm's base on the other."""
    rng = np.random.default_rng(0)
    contract = _contract()
    poses = rt.pose7_to_matrix(np.stack([_random_pose(rng) for _ in range(3)]))
    targets = rt.pose7_to_matrix(np.stack([_random_pose(rng) for _ in range(4)]))
    A = rt.pose7_to_matrix(_random_pose(rng))  # any rigid change of frame (a registration edge)
    moved = lambda T: np.einsum("ij,...jk->...ik", A, T)  # noqa: E731

    s1 = rt.state_vector(contract, poses, 0.05)
    s2 = rt.state_vector(contract, moved(poses), 0.05)
    a1 = rt.action_chunk(poses[-1], targets, np.full(4, 0.03))
    a2 = rt.action_chunk(moved(poses)[-1], moved(targets), np.full(4, 0.03))
    np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(a1, a2, atol=1e-5)


def test_targets_from_actions_inverts_action_chunk():
    rng = np.random.default_rng(1)
    T_obs = rt.pose7_to_matrix(_random_pose(rng))
    targets = rt.pose7_to_matrix(np.stack([_random_pose(rng) for _ in range(5)]))
    actions = rt.action_chunk(T_obs, targets, np.linspace(0.0, 0.08, 5))
    back, widths = rt.targets_from_actions(T_obs, actions)
    np.testing.assert_allclose(back, targets, atol=1e-5)
    np.testing.assert_allclose(widths, np.linspace(0.0, 0.08, 5), atol=1e-6)
    np.testing.assert_allclose(rt.matrix_to_pose7(rt.pose7_to_matrix(_random_pose(rng)))[..., :3].shape, (3,))


def test_rot6d_round_trip_and_gram_schmidt_repairs_a_noisy_prediction():
    rng = np.random.default_rng(2)
    R = rt.quat_xyzw_to_matrix(_random_pose(rng)[3:])
    np.testing.assert_allclose(rt.matrix_from_rot6d(rt.rot6d_from_matrix(R)), R, atol=1e-9)
    noisy = rt.matrix_from_rot6d(rt.rot6d_from_matrix(R) + rng.normal(scale=0.01, size=6))
    np.testing.assert_allclose(noisy @ noisy.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(noisy) == pytest.approx(1.0)


def test_contract_layout_and_round_trip(tmp_path):
    contract = _contract(wrench_names=[f"box_six_d_external.{a}" for a in rt.WRENCH_AXES])
    assert contract.history_offsets == [-4, -2, 0]
    assert contract.action_offsets == [2, 4, 6, 8]
    assert len(contract.state_names) == 2 * 9 + 1 + 6
    contract.write(tmp_path)
    assert rt.PolicyContract.read(tmp_path) == contract
    with pytest.raises(ValueError, match="FK target"):
        _contract(tcp_link="fr3_link8").validate()
    with pytest.raises(FileNotFoundError, match="not trained by"):
        rt.PolicyContract.read(tmp_path / "missing")


def test_samples_pad_untracked_targets_and_skip_untracked_observations():
    contract = _contract(history=1, horizon=3)
    n = 12
    poses = np.tile([0.1, 0.2, 0.3, 0, 0, 0, 1.0], (n, 1))
    poses[:, 0] += np.arange(n) * 0.01
    poses[6] = np.nan  # tracking lost for one frame
    arrays = rd.EpisodeArrays(poses=poses, widths=np.full(n, 0.04), wrench=None)

    assert rd.sample_vectors(contract, arrays, 6) is None  # its own pose is missing
    assert rd.sample_vectors(contract, arrays, 8) is None  # its history (t-2) is missing
    state, actions, pad = rd.sample_vectors(contract, arrays, 2)  # targets at 4, 6 (missing), 8
    assert pad.tolist() == [False, True, False]
    assert actions[0, 0] == pytest.approx(0.02) and actions[2, 0] == pytest.approx(0.06)
    _, _, pad_end = rd.sample_vectors(contract, arrays, 9)  # 11, then past the end
    assert pad_end.tolist() == [False, True, True]
    first, _, _ = rd.sample_vectors(contract, arrays, 0)  # history before the start: not moving
    np.testing.assert_allclose(first[:3], 0.0, atol=1e-9)


def test_history_dropout_shows_some_samples_a_still_history_and_keeps_their_width():
    import torch

    contract = _contract(history=2, horizon=3, cameras=[])
    n = 20
    poses = np.tile([0.1, 0.2, 0.3, 0, 0, 0, 1.0], (n, 1))
    poses[:, 2] += np.arange(n) * 0.01  # moving: the own history is not the still one
    arrays = rd.EpisodeArrays(poses=poses, widths=np.full(n, 0.07), wrench=None)
    own, actions, _ = rd.sample_vectors(contract, arrays, 10)
    still = rd.still_history(contract, own)
    at_rest = rt.state_vector(contract, np.tile(rt.pose7_to_matrix(poses[10]), (3, 1, 1)), 0.07)
    np.testing.assert_allclose(still, at_rest, atol=1e-6)  # what a TCP that did not move gives
    assert still[-1] == pytest.approx(0.07) and not np.allclose(own, still)

    ds = object.__new__(rd.RelativeTcpDataset)  # the sample path only: no views on disk
    ds.contract, ds.arrays, ds.index, ds.image_transforms = contract, [[arrays]], [(0, 0, 10)], None
    ds.history_dropout = 0.5
    torch.manual_seed(0)
    got = [ds[0] for _ in range(400)]
    dropped = sum(np.allclose(g["observation.state"].numpy(), still) for g in got)
    assert 150 < dropped < 250  # about half
    assert all(np.allclose(g["observation.state"].numpy(), own) for g in got
               if not np.allclose(g["observation.state"].numpy(), still))
    assert all(np.array_equal(g["action"].numpy(), actions) for g in got)  # the targets never change
    ds.history_dropout = 0.0
    assert all(np.allclose(ds[0]["observation.state"].numpy(), own) for _ in range(20))

    with pytest.raises(rd.TrainingDataError, match="0 <= p < 1"):
        rd.RelativeTcpDataset([], contract, {}, history_dropout=1.0)
    with pytest.raises(rd.TrainingDataError, match="needs a pose history"):
        rd.RelativeTcpDataset([], _contract(history=0, cameras=[]), {}, history_dropout=0.3)


def test_view_videos_return_each_frame_in_order(tmp_path):
    """A view is decoded with PyAV directly (Thor has neither torchcodec nor torchvision's VideoReader)."""
    import json
    import shutil
    import subprocess

    import pyarrow as pa
    import pyarrow.parquet as pq

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    cam = rt.CameraInput("cam_10", 64, 48, [0, 0, 64, 48], 64, 48)
    video = tmp_path / "videos" / cam.key / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    # 20 frames whose brightness rises 10 per frame; episode 1 starts at frame 8 of the file.
    frames = np.stack([np.full((48, 64, 3), 10 * i + 5, np.uint8) for i in range(20)])
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48", "-r", "60", "-i", "-",
         "-c:v", "libx264", "-qp", "0", "-g", "1", "-pix_fmt", "yuv444p", str(video)],
        input=frames.tobytes(), capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    (tmp_path / "meta" / "episodes").mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 60}), encoding="utf-8")
    pq.write_table(
        pa.table({
            "episode_index": [0, 1],
            f"videos/{cam.key}/chunk_index": [0, 0],
            f"videos/{cam.key}/file_index": [0, 0],
            f"videos/{cam.key}/from_timestamp": [0.0, 8 / 60],
        }),
        tmp_path / "meta" / "episodes" / "file-000.parquet",
    )
    videos = rd.ViewVideos(tmp_path, [cam])
    levels = [int(videos.frame(1, t, "cam_10").mean()) for t in (0, 3, 1, 11)]
    assert levels == pytest.approx([85, 115, 95, 195], abs=3)


def test_pi05_load_that_would_not_fit_in_ram_is_refused(tmp_path):
    """A pi0.5 load peaks at ~3x its weights in RAM; one that needs swap froze a workstation (2026-11-17)."""
    from tools.fr3 import relative_tcp_policy as rtp

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\0" * 1000)
    assert rtp.weights_file(tmp_path) == weights
    need = int(1000 * rtp.LOAD_PEAK_PER_WEIGHT_BYTE)
    rtp.check_host_memory("pi05", weights, available=need)
    with pytest.raises(rtp.HostMemoryError, match="2 processes"):
        rtp.check_host_memory("pi05", weights, processes=2, available=need)
    with pytest.raises(rtp.HostMemoryError):
        rtp.check_host_memory("pi05", weights, available=need - 1)
    rtp.check_host_memory("act", weights, available=0)  # ACT is small and loads as before
    rtp.check_host_memory("pi05", None, available=0)  # ACT: nothing to measure
    # A lean load's peak is the process's, not a multiple of the weights file.
    rtp.check_host_memory("pi05", weights, available=rtp.LEAN_LOAD_PEAK_BYTES, lean=True)
    with pytest.raises(rtp.HostMemoryError):
        rtp.check_host_memory("pi05", weights, available=rtp.LEAN_LOAD_PEAK_BYTES - 1, lean=True)


class _ToyPolicy:
    """The parts of a pi0.5 the lean load touches: a module tree, its key fixer and a config."""

    def __init__(self):
        import torch

        self.module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2, bias=False))
        self.module[1].to(torch.bfloat16)
        self.config = None

    def state_dict(self, keep_vars=False):
        return {f"model.{k}": v for k, v in self.module.state_dict(keep_vars=keep_vars).items()}

    def _fix_pytorch_state_dict_keys(self, state_dict, config):
        fixed = {k.replace("old_", ""): v for k, v in state_dict.items()}
        if "lm_head.weight" in fixed:  # as pi0.5 does: the head's weights also become the embedding's
            fixed["1.weight"] = fixed.pop("lm_head.weight").clone()
        return fixed


def test_lean_weight_load_copies_every_tensor_and_refuses_a_mismatch(tmp_path):
    import torch
    from safetensors.torch import save_file

    from tools.fr3 import relative_tcp_policy as rtp

    toy = _ToyPolicy()
    saved = {"old_0.weight": torch.randn(4, 3), "0.bias": torch.randn(4), "lm_head.weight": torch.randn(2, 4)}
    save_file(saved, tmp_path / "model.safetensors")
    with torch.no_grad():
        rtp._load_weights_streaming(toy, tmp_path / "model.safetensors")
    assert torch.equal(toy.module[0].weight, saved["old_0.weight"])
    assert torch.equal(toy.module[0].bias, saved["0.bias"])
    assert toy.module[1].weight.dtype == torch.bfloat16  # cast as load_state_dict casts
    assert torch.equal(toy.module[1].weight, saved["lm_head.weight"].to(torch.bfloat16))

    # A key the model does not have, or one it lacks, is an error: never a partly random model.
    save_file({**saved, "extra.weight": torch.zeros(1)}, tmp_path / "extra.safetensors")
    with pytest.raises(RuntimeError, match="1 unexpected"), torch.no_grad():
        rtp._load_weights_streaming(toy, tmp_path / "extra.safetensors")
    save_file({k: v for k, v in saved.items() if k != "0.bias"}, tmp_path / "short.safetensors")
    with pytest.raises(RuntimeError, match="1 missing"), torch.no_grad():
        rtp._load_weights_streaming(toy, tmp_path / "short.safetensors")
    save_file({**saved, "0.bias": torch.zeros(5)}, tmp_path / "shape.safetensors")
    with pytest.raises(RuntimeError, match=r"model\.0\.bias"), torch.no_grad():
        rtp._load_weights_streaming(toy, tmp_path / "shape.safetensors")


def test_lean_load_refuses_an_unsaved_buffer_it_cannot_compute():
    import torch

    from tools.fr3 import relative_tcp_policy as rtp

    module = torch.nn.Module()
    module.register_buffer("mystery", torch.zeros(3), persistent=False)
    with pytest.raises(RuntimeError, match="mystery"), torch.no_grad():
        rtp._recompute_unsaved_buffers(module)


def test_missing_pi05_base_weights_are_refused_not_replaced_by_random_ones(tmp_path):
    """PI05Policy.from_pretrained silently returns a random model when its weights are missing."""
    from tools.fr3 import relative_tcp_policy as rtp

    with pytest.raises(FileNotFoundError, match="random weights"):
        rtp.require_weights(tmp_path)
    with pytest.raises(FileNotFoundError):
        rtp.require_weights("no-such-org/no-such-model")
    (tmp_path / "model.safetensors").write_bytes(b"\0")
    assert rtp.require_weights(tmp_path) == tmp_path / "model.safetensors"
    assert rtp.host_memory_available() > 0


def test_pi05_is_refused_outright_on_an_8gb_gpu():
    """The RTX 4060 (8 GB): pi0.5 inference peaks at 8.8 GiB, so it is refused with the reason, ACT is not."""
    from tools.fr3 import relative_tcp_policy as rtp

    reason = rtp.gpu_too_small("pi05", "NVIDIA GeForce RTX 4060", 7.6)
    assert reason and "8.8 GiB" in reason and "41 GB" in reason and "A100" in reason
    assert rtp.gpu_too_small("act", "NVIDIA GeForce RTX 4060", 7.6) is None
    assert rtp.gpu_too_small("pi05", "NVIDIA A100-SXM4-80GB", 79.3) is None
    rtp.check_gpu_memory("pi05", "cpu")  # no GPU involved: the RAM check decides
