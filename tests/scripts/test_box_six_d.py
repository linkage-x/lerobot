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


FIXTURE_ACC = [-1.0, 0.0, -0.05]
BOOT_S = 2_000_000_000.0  # the re-power


def _origin(**over):
    # 35 min after the previous power-on, on the fixture, 15 s before the re-power
    return {"kind": "origin", "wall_s": BOOT_S - 15, "mcu_us": 2100e6, "acc_g": FIXTURE_ACC, **over}


def _zero(at_s=BOOT_S + 80, **over):
    return {"kind": "software", "wall_s": at_s, "mcu_us": (at_s - BOOT_S) * 1e6, "acc_g": FIXTURE_ACC,
            "fxyz_mxyz_raw": [0.15, 0.01, -0.27, 0.001, 0.0, 0.002], **over}


def _checked_origin(**over):
    origin = bsd.with_recheck(_origin(**over), _zero())
    assert origin is not None
    return origin


def _warnings(origin, zero, *, at_s=BOOT_S + 600, boot_s=BOOT_S):
    return bsd.calibration_warnings(origin, zero, now_s=at_s, uptime_s=at_s - boot_s)


def test_the_full_procedure_warns_of_nothing():
    assert _checked_origin()["recheck"]["pass"] is True
    assert _warnings(_checked_origin(), _zero()) == []


def test_each_missed_step_of_the_procedure_is_named():
    origin, zero = _checked_origin(), _zero()
    assert any("没有 origin" in w for w in _warnings(None, zero))
    assert any("预热了 5 分钟" in w for w in _warnings(_checked_origin(mcu_us=300e6), zero))
    assert any("origin 时不在工装" in w for w in _warnings(_checked_origin(acc_g=[0.0, 0.0, 1.0]), zero))
    assert any("还没有复核" in w for w in _warnings(_origin(), zero))
    assert any("还没有做动态校准" in w for w in _warnings(origin, None))
    assert any("origin 之后还没有" in w for w in _warnings(origin, {**_origin(), "kind": "origin"}))
    assert any("动态校准时不在工装" in w for w in _warnings(origin, _zero(acc_g=[0.7, -0.1, 0.7])))


def test_an_origin_with_no_power_cycle_after_it_has_not_taken_effect():
    # the 11-06 16:46:38 origin: run 27 s into a boot, nothing since
    origin = _origin(wall_s=BOOT_S + 27, mcu_us=27e6)
    found = _warnings(origin, _zero(at_s=BOOT_S + 42))
    assert any("origin 后还没有重新上电" in w for w in found)
    assert any("预热了 0 分钟" in w for w in found)
    assert bsd.with_recheck(origin, _zero(at_s=BOOT_S + 42)) is None  # same boot: raw says nothing


def test_the_recheck_is_the_first_fixture_zero_after_the_re_power_and_no_later_one():
    bad = bsd.with_recheck(_origin(), _zero(fxyz_mxyz_raw=[0.7, 0, 0, 0, 0, 0]))
    assert bad["recheck"]["pass"] is False
    assert any("复核 raw 超限" in w for w in _warnings(bad, _zero()))
    assert bsd.with_recheck(_origin(), _zero(acc_g=[0.0, 0.0, 1.0])) is None
    assert bsd.with_recheck(_checked_origin(), _zero(at_s=BOOT_S + 4000)) is None


def test_old_calibrations_expire():
    origin = _checked_origin()
    late = BOOT_S + 1900
    assert any("31 分钟" in w for w in _warnings(origin, _zero(), at_s=BOOT_S + 80 + 1860))
    assert _warnings(origin, _zero(at_s=late), at_s=late + 60) == []
    day = BOOT_S + 25 * 3600
    assert any("要求每天一次" in w for w in _warnings(origin, _zero(at_s=day), at_s=day + 60))


def test_a_new_power_on_needs_a_new_zero_and_a_warm_up_unless_it_is_the_origins_re_power():
    origin = _checked_origin()
    assert _warnings(origin, _zero(), at_s=BOOT_S + 200) == []  # warm: the sensor was on for the origin
    cold = BOOT_S + 3 * 3600
    found = _warnings(origin, _zero(), at_s=cold + 300, boot_s=cold)
    assert any("重新上电后还没有做动态校准" in w for w in found)
    assert any("才上电 5 分钟" in w for w in found)


def test_an_unknown_uptime_skips_what_needs_it_and_still_checks_the_rest():
    found = bsd.calibration_warnings(None, None, now_s=BOOT_S, uptime_s=None)
    assert len(found) == 2
