# ruff: noqa: N803, N806  -- T, R, T_ref: the usual names of transforms and rotations
"""Poses relative to the TCP: the one conversion training and inference share.

User guide section 8 ("infer-relative", decided 11-16): the policy never sees an absolute pose. With
``T_t`` the TCP pose at the observation frame t (in whatever frame the poses come in: the camera world
for a recording, the arm's base for FK on the arm):

* **action** = for k in the chunk, ``T_t^-1 · T_{t+k}`` (the target relative to the TCP at t, expressed
  in the TCP frame at t) and the absolute gripper width;
* **state** = for a few frames back, ``T_t^-1 · T_{t-j}``, the width now and, if the policy uses it, the
  BOX's external wrench (already a quantity of the end effector).

A rigid change of frame ``T -> A·T`` leaves every ``T_t^-1 · T_x`` unchanged, which is the whole point:
the camera world of the recording and the base of the arm never meet, so the world->base edge stays out
of the control loop. On the arm a target is ``FK(q_t) · ΔT`` with ``q_t`` read at the observation, not
at execution (:func:`targets_from_actions`).

Rotations go in the 6D representation (first two columns of R): continuous, unlike quaternions or
rotation vectors, and turned back into a rotation by Gram-Schmidt. numpy only; torch stays out of here so
the gateway and tests can import it.

The policy contract (:class:`PolicyContract`) says what goes in and comes out, in which order, from which
dataset columns, after which image crop. It is written next to every checkpoint as
``fr3_policy_contract.json``; inference reads it back and refuses a checkpoint without one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_SCHEMA = "fr3_relative_tcp/1"
CONTRACT_FILE = "fr3_policy_contract.json"
PROVENANCE_FILE = "fr3_provenance.json"
#: The point the TCP pose is of: the gripper's ball-socket centre. Exports label it (info.json
#: ``tcp_frame.link``) and the arm's FK must reach the same point; a different point turns every
#: rotation into a wrong translation, which a relative pose does not cancel.
TCP_LINK = "corenetic_gripper_ee"

POSE_DIM = 9  # dx, dy, dz (m) + rotation 6D
POSE_NAMES = ("dx_m", "dy_m", "dz_m", "r00", "r10", "r20", "r01", "r11", "r21")
WIDTH_NAME = "gripper_width_m"
WRENCH_AXES = ("fx", "fy", "fz", "mx", "my", "mz")


# ------------------------------------------------------------------ rotations ---


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions (x, y, z, w) -> (..., 3, 3); normalised first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) quaternions (x, y, z, w) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
        q = np.asarray(q)
        out[i] = q if q[3] >= 0 else -q
    return out.reshape(*R.shape[:-2], 4)


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first column, then second column."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def matrix_from_rot6d(v: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt; any finite non-degenerate input gives a rotation."""
    v = np.asarray(v, dtype=np.float64)
    a, b = v[..., :3], v[..., 3:6]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(c0 * b, axis=-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)


# --------------------------------------------------------------------- poses ---


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(..., 7) (x, y, z, qx, qy, qz, qw) -> (..., 4, 4)."""
    pose = np.asarray(pose, dtype=np.float64)
    T = np.zeros((*pose.shape[:-1], 4, 4))
    T[..., :3, :3] = quat_xyzw_to_matrix(pose[..., 3:7])
    T[..., :3, 3] = pose[..., :3]
    T[..., 3, 3] = 1.0
    return T


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[..., :3, 3], matrix_to_quat_xyzw(T[..., :3, :3])], axis=-1)


def invert(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def relative(T_ref: np.ndarray, T: np.ndarray) -> np.ndarray:
    """``T_ref^-1 · T``, broadcasting ``T_ref`` (4, 4) over ``T`` (..., 4, 4)."""
    return np.einsum("ij,...jk->...ik", invert(T_ref), np.asarray(T, dtype=np.float64))


def encode(T_rel: np.ndarray) -> np.ndarray:
    """(..., 4, 4) relative transforms -> (..., 9)."""
    T_rel = np.asarray(T_rel)
    return np.concatenate([T_rel[..., :3, 3], rot6d_from_matrix(T_rel[..., :3, :3])], axis=-1)


def decode(v: np.ndarray) -> np.ndarray:
    """(..., 9) -> (..., 4, 4) relative transforms."""
    v = np.asarray(v, dtype=np.float64)
    T = np.zeros((*v.shape[:-1], 4, 4))
    T[..., :3, :3] = matrix_from_rot6d(v[..., 3:9])
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


# ------------------------------------------------------------------ contract ---


@dataclass
class CameraInput:
    """One camera the policy sees, and the crop applied before it is resized (source pixels)."""

    name: str  # cam_10, cam_06, ...
    source_width: int
    source_height: int
    crop: list[int]  # x0, y0, x1, y1 in source pixels (x1, y1 exclusive)
    width: int  # what the policy gets
    height: int

    @property
    def key(self) -> str:
        return f"observation.images.{self.name}"

    def validate(self) -> None:
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= self.source_width and 0 <= y0 < y1 <= self.source_height):
            raise ValueError(
                f"{self.name}: crop {self.crop} is not inside the {self.source_width}x{self.source_height} image"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.name}: output size {self.width}x{self.height}")


@dataclass
class PolicyContract:
    """What a policy trained here takes and returns. Written next to every checkpoint."""

    hand: str  # which hand's TCP the training data followed (right / left)
    fps: int  # dataset frame rate
    frame_stride: int  # frames between consecutive action steps (and history samples)
    horizon: int  # action steps per chunk
    history: int  # past relative poses in the state (each frame_stride apart)
    cameras: list[CameraInput]
    width_name: str  # dataset state column of the gripper width (metres)
    wrench_names: list[str] = field(default_factory=list)  # dataset state columns, empty = not used
    tcp_link: str = TCP_LINK
    schema: str = CONTRACT_SCHEMA

    # ---- layout
    @property
    def step_s(self) -> float:
        return self.frame_stride / float(self.fps)

    @property
    def state_names(self) -> list[str]:
        names = [f"hist{j}.{n}" for j in range(1, self.history + 1) for n in POSE_NAMES]
        names.append(WIDTH_NAME)
        names += [f"wrench.{axis}" for axis in WRENCH_AXES[: len(self.wrench_names)]]
        return names

    @property
    def action_names(self) -> list[str]:
        return [*POSE_NAMES, WIDTH_NAME]

    @property
    def history_offsets(self) -> list[int]:
        """Frame offsets of the history samples, oldest first, then 0 (the observation frame)."""
        return [-(j * self.frame_stride) for j in range(self.history, 0, -1)] + [0]

    @property
    def action_offsets(self) -> list[int]:
        """Frame offsets of the action targets: the tracked TCP pose at t+s, t+2s, ..., t+horizon*s.

        The tracked pose itself, not the export's ``action.ee_pose`` column: that one is the sidecar's
        own "next pose", one frame ahead whatever the stride, and the chunk's steps must be evenly spaced.
        """
        return [(k + 1) * self.frame_stride for k in range(self.horizon)]

    @property
    def pose_offsets(self) -> list[int]:
        """Every frame offset the TCP pose column is read at: history, the observation, the targets."""
        return [*self.history_offsets, *self.action_offsets]

    def validate(self) -> None:
        if self.schema != CONTRACT_SCHEMA:
            raise ValueError(f"policy contract schema {self.schema!r}, this code reads {CONTRACT_SCHEMA!r}")
        if self.tcp_link != TCP_LINK:
            raise ValueError(f"policy TCP is {self.tcp_link!r}; the arm's FK target is {TCP_LINK!r}")
        if self.fps <= 0 or self.frame_stride <= 0 or self.horizon <= 0 or self.history < 0:
            raise ValueError("fps, frame_stride and horizon must be positive, history >= 0")
        if self.wrench_names and len(self.wrench_names) != len(WRENCH_AXES):
            raise ValueError(f"wrench needs {len(WRENCH_AXES)} columns, got {self.wrench_names}")
        if not self.cameras:
            raise ValueError("a policy needs at least one camera")
        for cam in self.cameras:
            cam.validate()

    # ---- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state_names"] = self.state_names
        out["action_names"] = self.action_names
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyContract:
        fields = {k: v for k, v in data.items() if k not in ("state_names", "action_names")}
        fields["cameras"] = [CameraInput(**cam) for cam in fields.get("cameras") or []]
        contract = cls(**fields)
        contract.validate()
        return contract

    def write(self, directory: Path) -> Path:
        path = Path(directory) / CONTRACT_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, directory: Path) -> PolicyContract:
        path = Path(directory) / CONTRACT_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing: this checkpoint was not trained by train_relative_tcp.py, so nothing says "
                "what its inputs are"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------- state / action vectors ---


def state_vector(
    contract: PolicyContract,
    history_poses: np.ndarray,
    width: float,
    wrench: np.ndarray | None = None,
) -> np.ndarray:
    """The policy's ``observation.state``.

    ``history_poses``: (history + 1, 4, 4) TCP poses at :attr:`PolicyContract.history_offsets`, the last
    one at the observation frame, all in one frame (camera world or arm base, it does not matter).
    """
    poses = np.asarray(history_poses, dtype=np.float64)
    if poses.shape != (contract.history + 1, 4, 4):
        raise ValueError(f"history poses {poses.shape}, expected {(contract.history + 1, 4, 4)}")
    parts = [encode(relative(poses[-1], poses[:-1])).reshape(-1), np.asarray([width], dtype=np.float64)]
    if contract.wrench_names:
        if wrench is None or np.asarray(wrench).shape != (len(WRENCH_AXES),):
            raise ValueError("this policy takes the external wrench; none (or the wrong shape) was given")
        parts.append(np.asarray(wrench, dtype=np.float64))
    return np.concatenate(parts).astype(np.float32)


def action_chunk(T_obs: np.ndarray, targets: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """(horizon, 10): each target relative to the TCP at the observation, then the absolute width."""
    rel = encode(relative(np.asarray(T_obs), np.asarray(targets)))
    return np.concatenate([rel, np.asarray(widths, dtype=np.float64)[:, None]], axis=-1).astype(np.float32)


def targets_from_actions(T_obs_base: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execution layer: ``FK(q_t) · ΔT`` for every step, and the widths.

    ``T_obs_base`` must be the TCP pose (FK to :data:`TCP_LINK`) at the frame the observation was taken,
    not the pose when the result arrives; the latency would otherwise drift the whole trajectory.
    Returns ((horizon, 4, 4) targets in the base, (horizon,) widths).
    """
    actions = np.asarray(actions, dtype=np.float64)
    deltas = decode(actions[..., :POSE_DIM])
    targets = np.einsum("ij,...jk->...ik", np.asarray(T_obs_base, dtype=np.float64), deltas)
    return targets, actions[..., POSE_DIM]


# ---------------------------------------------------------------------- images ---


def crop_resize(image: np.ndarray, camera: CameraInput) -> np.ndarray:
    """HxWx3 uint8 source frame -> camera.height x camera.width x 3 uint8 (area interpolation).

    Inference applies this to the recorder's frame; training gets the same crop from a transcoded copy of
    the export's video (``ffmpeg crop + scale``), so the two differ only by the video codec.
    """
    import cv2  # local: the gateway imports this module without cv2 on some machines

    h, w = image.shape[:2]
    if (w, h) != (camera.source_width, camera.source_height):
        raise ValueError(
            f"{camera.name}: frame is {w}x{h}, the policy was trained on {camera.source_width}x"
            f"{camera.source_height} (crop {camera.crop})"
        )
    x0, y0, x1, y1 = camera.crop
    return cv2.resize(image[y0:y1, x0:x1], (camera.width, camera.height), interpolation=cv2.INTER_AREA)
