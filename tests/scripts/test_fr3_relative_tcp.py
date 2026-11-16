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
