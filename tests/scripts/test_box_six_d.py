# ruff: noqa: N803, N806  -- matrices keep their math names
"""box_six_d: the per-BOX payload model, its fit from a free-air rotation, and offline compensation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.thor.gmsl2 import box_six_d as bsd

G = bsd.G
# A rigid 0.5 kg payload, CoM 6 cm out along FT z, FT turned 180 deg about z from the IMU-ish frame.
R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
MASS, COM = 0.5, np.array([0.0, 0.003, 0.06])
KF = -MASS * G * R
K_TRUE = np.vstack([KF, np.stack([np.cross(COM, KF[:, j]) for j in range(3)], axis=1)])


def _model(K=K_TRUE, valid_from=100.0):
    return bsd.PayloadModel(1, valid_from, "test", K, "test")


def _faces(n_per=40, seed=0, faces=("x+", "x-", "y+", "y-", "z+", "z-")):
    rng = np.random.default_rng(seed)
    out = []
    for face in faces:
        axis, sign = "xyz".index(face[0]), 1.0 if face[1] == "+" else -1.0
        for _ in range(n_per):
            v = rng.normal(0, 0.08, 3)
            v[axis] = sign
            out.append(v / np.linalg.norm(v))
    return np.array(out)


def test_fit_recovers_the_payload_and_reports_a_rigid_reading():
    acc = _faces()
    bias = np.array([0.3, -0.5, -2.7, -0.18, 0.02, 0.0])  # whatever the last zero left
    f = acc @ K_TRUE.T + bias + np.random.default_rng(1).normal(0, 0.03, (len(acc), 6))
    fit = bsd.fit_payload(f, acc, np.zeros_like(acc))

    assert np.allclose(fit["K"], K_TRUE, atol=0.05)
    assert fit["meets_target"]
    assert fit["missing_attitudes"] == []
    assert fit["rigid_reading"]["mass_kg"] == pytest.approx(MASS, abs=0.01)
    assert fit["rigid_reading"]["com_in_ft_m"] == pytest.approx(COM.tolist(), abs=0.003)
    assert fit["zero_offset_seen"] == pytest.approx(bias.tolist(), abs=0.02)


def test_fit_refuses_too_few_attitudes_and_moving_samples():
    acc = _faces(faces=("x+", "x-", "y+"))
    with pytest.raises(ValueError, match="attitudes"):
        bsd.fit_payload(acc @ K_TRUE.T, acc, np.zeros_like(acc))
    acc = _faces()
    with pytest.raises(ValueError, match="quasi-static"):
        bsd.fit_payload(acc @ K_TRUE.T, acc, np.full_like(acc, 30.0))  # turning at 52 deg/s


def test_compensation_undoes_the_zero_bias_at_every_attitude_and_keeps_a_real_push():
    # SDK's wrong model zeroed gripper-down: f equals ITS model there, so f carries the difference.
    model = _model()
    acc0 = np.array([-1.0, 0.0, 0.0])
    sdk_wrong_at_zero = np.array([0.0, 0.0, -7.78, -0.17, 0.0, 0.0])
    bias_true = sdk_wrong_at_zero - model.wrench(acc0)
    zero = {
        "kind": "software",
        "fxyz_mxyz": sdk_wrong_at_zero.tolist(),
        "acc_g": acc0.tolist(),
        "wall_s": 200.0,
    }
    acc = _faces(n_per=5)
    push = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    f = acc @ K_TRUE.T + bias_true + push

    ext = bsd.external_wrench(f, acc, model, bsd.zero_bias(model, zero))

    assert np.allclose(ext, push, atol=1e-9)


def test_zero_bias_refuses_an_origin_zero_and_a_zero_from_an_earlier_end_effector():
    model = _model(valid_from=100.0)
    with pytest.raises(ValueError, match="origin"):
        bsd.zero_bias(model, {"kind": "origin", "wall_s": 200.0})
    with pytest.raises(ValueError, match="predates"):
        bsd.zero_bias(model, {"kind": "software", "fxyz_mxyz": [0] * 6, "acc_g": [1, 0, 0], "wall_s": 50.0})


def test_the_payload_record_in_force_is_the_latest_before_the_time():
    reg = {
        "schema": bsd.SCHEMA,
        "devices": {
            "7": {
                "role": "handheld",
                "hand": "left",
                "six_d_payload": [
                    {
                        "valid_from_unix": 100,
                        "model": bsd.MODEL,
                        "K": K_TRUE.tolist(),
                        "end_effector": "no cube",
                    },
                    {
                        "valid_from_unix": 300,
                        "model": bsd.MODEL,
                        "K": None,
                        "end_effector": "cube on, not fitted",
                    },
                ],
            }
        },
    }
    assert bsd.payload_model(reg, 7, 50) is None
    assert bsd.payload_model(reg, 7, 200).end_effector == "no cube"
    assert bsd.payload_model(reg, 7, 400) is None  # the cube changed the load; the old K must not apply
    assert bsd.payload_model(reg, 8, 200) is None
    assert bsd.box_roles(reg, [7, 8]) == {"7": {"role": "handheld", "hand": "left"}}


def test_the_checked_in_registry_names_the_three_boxes():
    reg = bsd.load_registry()
    roles = bsd.box_roles(reg, [3969377039, 3906437858, 3100715645])
    assert roles == {
        "3969377039": {"role": "handheld", "hand": "left"},
        "3906437858": {"role": "handheld", "hand": "right"},
        "3100715645": {"role": "fr3_gripper"},
    }
    left = bsd.payload_model(reg, 3969377039, 1793677900)
    assert left is not None and left.K.shape == (6, 3)
    # gripper down (IMU x up) the left payload weighs ~5 N on FT z
    assert left.wrench(np.array([1.0, 0.0, 0.0]))[2] == pytest.approx(5.22, abs=0.05)


def test_an_episode_is_compensated_from_its_meta_and_box_sensors(tmp_path):
    reg = {
        "schema": bsd.SCHEMA,
        "devices": {
            "7": {"six_d_payload": [{"valid_from_unix": 100, "model": bsd.MODEL, "K": K_TRUE.tolist()}]}
        },
    }
    acc0 = np.array([-1.0, 0.0, 0.0])
    bias = np.array([0.2, 0.1, -2.7, -0.18, 0.0, 0.0])
    f0 = acc0 @ K_TRUE.T + bias
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "box_six_d_zero": {
                    "7": {
                        "kind": "software",
                        "fxyz_mxyz": f0.tolist(),
                        "acc_g": acc0.tolist(),
                        "wall_s": 150.0,
                    }
                }
            }
        )
    )
    acc1 = np.array([0.0, 1.0, 0.0])
    rows = [
        {"sid": "box_imu", "wall_s": 200.0, "data": {"acc_g": acc1.tolist()}},
        {
            "sid": "box_six_d_force",
            "wall_s": 200.001,
            "data": {"fxyz_mxyz": (acc1 @ K_TRUE.T + bias).tolist()},
        },
    ]
    (tmp_path / "box_sensors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = bsd.compensate_episode(tmp_path, reg)

    assert np.allclose(out["7"]["external"], 0.0, atol=1e-9)
