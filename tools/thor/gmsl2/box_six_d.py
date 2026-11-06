# ruff: noqa: N806  -- matrices keep their math names (K, R, U S Vt)
"""6D force on a BOX: a payload model fitted per BOX, and the external wrench computed offline.

Why not the SDK's ``fxyz_mxyz_no_gravity``: box_sdk.conf holds ONE IMU->FT extrinsic and ONE payload
(mass, CoM) for every BOX the process opens (the vendor's defaults, which predate the gripper we
mount), and the library exports no per-device setter. Measured 11-03 on the left handheld
(3969377039), free air: the conf's extrinsic is rotated ~180 deg about the FT z axis, the payload is
~0.50 kg not 0.7935, and the per-axis gains differ (FT Fy ~17 % below Fx/Fz), so ``no_gravity`` is right only at
the attitude of the last software zero and 5-13 N off elsewhere.

The model here is what a free-air rotation actually shows, with no rigid-body assumption:

    payload wrench W(acc) = K @ acc_g        (K: 6x3, N and N*m per g)

``acc_g`` is the IMU's specific force. For a rigid payload the sensor feels ``-m*(a - g)``, which IS
``-m * 9.81 * acc_g`` in the IMU frame, so the same K also takes out the payload's inertia under
linear acceleration (not the rotational terms, ~0.03 N at 3 rad/s on a 6 cm lever). A rigid body
would make K's force block ``-m*9.81*R_ft_imu``; the fit reports how far it is from that.

The software zero (``cali_6d_force_sensor``) shifts ``fxyz_mxyz`` so it equals the SDK's model at
the attitude ``acc0`` it ran at, which leaves the constant ``b = f0 - K @ acc0`` (with f0 the reading
right after the zero) in every later sample. The recorder keeps ``f0``/``acc0`` per BOX
(``meta.json`` ``box_six_d_zero``); then

    external(t) = f(t) - K @ acc(t) - b

Parameters live in ``tools/thor/box_devices.json``, per BOX id and per end-effector configuration
(``valid_from``): the payload belongs to what is mounted, not to the BOX, and both change.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "box_devices.json"
SCHEMA = "box_devices/v1"
MODEL = "affine_acc/v1"
G = 9.80665

#: A free-air sample is quasi-static when the BOX turns slower than this and |acc| is 1 g within
#: STATIC_ACC_TOL_G; hand tremor on a held BOX stays well under both.
STATIC_GYR_DEG_S = 15.0
STATIC_ACC_TOL_G = 0.05
#: Held-out-attitude rms the fit must reach before it is worth writing to the registry.
TARGET_HELDOUT_RMS_N = 0.3


@dataclass(frozen=True)
class PayloadModel:
    device_id: int
    valid_from_unix: float
    end_effector: str
    K: np.ndarray  # (6, 3)
    status: str

    def wrench(self, acc_g: np.ndarray) -> np.ndarray:
        """Payload wrench (N, N*m) for specific force(s) ``acc_g`` (..., 3) in g."""
        return np.asarray(acc_g, dtype=float) @ self.K.T


# --- registry ----------------------------------------------------------------------------------


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {reg.get('schema')!r}, expected {SCHEMA!r}")
    return reg


def device_entry(registry: dict[str, Any], device_id: int | str) -> dict[str, Any] | None:
    return (registry.get("devices") or {}).get(str(int(device_id)))


def payload_model(registry: dict[str, Any], device_id: int | str, at_unix: float) -> PayloadModel | None:
    """The payload model in force for ``device_id`` at ``at_unix``: the latest ``valid_from`` not after it.

    None when the BOX is unknown, nothing was fitted yet, or the latest configuration before
    ``at_unix`` has no fit (``K`` null: the end effector changed and nobody re-fitted).
    """
    entry = device_entry(registry, device_id)
    if not entry:
        return None
    eligible = [p for p in entry.get("six_d_payload") or [] if float(p["valid_from_unix"]) <= at_unix]
    if not eligible:
        return None
    p = max(eligible, key=lambda p: float(p["valid_from_unix"]))
    if p.get("model") != MODEL or p.get("K") is None:
        return None
    K = np.asarray(p["K"], dtype=float)
    if K.shape != (6, 3):
        raise ValueError(f"BOX {device_id}: K has shape {K.shape}, expected (6, 3)")
    return PayloadModel(
        int(device_id),
        float(p["valid_from_unix"]),
        str(p.get("end_effector", "")),
        K,
        str(p.get("status", "")),
    )


def box_roles(registry: dict[str, Any], device_ids: list[int | str]) -> dict[str, dict[str, Any]]:
    """``{id: {role, hand?}}`` for the BOXes the registry knows (what episode meta records)."""
    out: dict[str, dict[str, Any]] = {}
    for did in device_ids:
        entry = device_entry(registry, did)
        if entry:
            out[str(int(did))] = {k: entry[k] for k in ("role", "hand") if k in entry}
    return out


# --- compensation ------------------------------------------------------------------------------


def zero_bias(model: PayloadModel, zero: dict[str, Any]) -> np.ndarray:
    """The constant a software zero leaves in ``fxyz_mxyz``: ``f0 - K @ acc0`` (6,).

    ``zero`` is one BOX's ``box_six_d_zero`` record. An origin calibration (MCU side) re-zeros the
    raw channel instead, so the readings after it carry an offset nothing recorded; only a software
    zero afterwards makes them usable again.
    """
    if zero.get("kind") != "software":
        raise ValueError(
            f"last 6D zero of BOX {model.device_id} was {zero.get('kind')!r}, not a software zero"
        )
    f0 = np.asarray(zero["fxyz_mxyz"], dtype=float)
    acc0 = np.asarray(zero["acc_g"], dtype=float)
    if zero.get("wall_s") is not None and float(zero["wall_s"]) < model.valid_from_unix:
        # The zero was taken with an earlier end effector: its f0 is that payload's.
        raise ValueError(
            f"BOX {model.device_id}: zero at {zero['wall_s']} predates the payload "
            f"configuration valid from {model.valid_from_unix}; zero again"
        )
    return f0 - model.wrench(acc0)


def external_wrench(f: np.ndarray, acc_g: np.ndarray, model: PayloadModel, bias: np.ndarray) -> np.ndarray:
    """External wrench on the payload (N, N*m), (..., 6), from ``fxyz_mxyz`` and the IMU ``acc_g``."""
    return np.asarray(f, dtype=float) - model.wrench(acc_g) - bias


# --- fitting -----------------------------------------------------------------------------------


# The vendor's procedure (数采Box传感器标定与操作说明 v1.1): warm up 30 min, origin on the upright
# fixture, re-power, check raw, dynamic zero on the fixture, again every 30 min and after a re-power.
WARMUP_S = 30 * 60
ORIGIN_MAX_AGE_S = 24 * 3600
DYNAMIC_MAX_AGE_S = 30 * 60
# A boot this soon after an origin is the re-power the procedure asks for: the sensor stayed warm,
# so its short uptime is not a cold start.
REPOWER_WINDOW_S = 10 * 60
# IMU acc on the upright fixture, measured 11-06 on 3969377039. It only says the attitude was right:
# a hand holding the same attitude passes too.
FIXTURE_ACC_G = (-1.0, 0.0, -0.05)
FIXTURE_ACC_TOL_G = 0.15
ORIGIN_RAW_LIMITS = (0.5, 0.5, 0.5, 0.01, 0.01, 0.01)


def _on_fixture(record: dict[str, Any]) -> bool | None:
    acc = record.get("acc_g")
    if not acc:
        return None
    return bool(np.linalg.norm(np.asarray(acc, dtype=float) - FIXTURE_ACC_G) <= FIXTURE_ACC_TOL_G)


def with_recheck(origin: dict[str, Any], zero: dict[str, Any]) -> dict[str, Any] | None:
    """``origin`` with its re-check filled in from software zero ``zero``, or None if that is no re-check.

    The first zero on the fixture after the re-power sees the raw channel the manual wants checked;
    a software zero moves ``fxyz_mxyz`` only, so raw there is still what the origin left. Later zeros
    say nothing: raw may drift over a day without the origin having been wrong.
    """
    raw, mcu_us = zero.get("fxyz_mxyz_raw"), zero.get("mcu_us")
    if origin.get("recheck") or not raw or not mcu_us or not _on_fixture(zero):
        return None
    if zero["wall_s"] - mcu_us / 1e6 <= origin["wall_s"]:
        return None  # same boot as the origin, which has not taken effect yet
    ok = all(abs(v) <= limit for v, limit in zip(raw, ORIGIN_RAW_LIMITS, strict=True))
    return {**origin, "recheck": {"wall_s": zero["wall_s"], "fxyz_mxyz_raw": raw, "pass": ok}}


def calibration_warnings(
    origin: dict[str, Any] | None, zero: dict[str, Any] | None, *, now_s: float, uptime_s: float | None
) -> list[str]:
    """What an episode starting now misses of the vendor's procedure, in the operator's words.

    ``origin`` is the BOX's last origin (with its re-check), ``zero`` its last record in this recorder
    process, ``uptime_s`` how long it has been on now (None: unknown, the checks needing it are skipped).
    """
    found: list[str] = []
    boot_s = None if uptime_s is None else now_s - uptime_s
    if not origin:
        found.append("没有 origin 记录")
    else:
        age_s = now_s - origin["wall_s"]
        if age_s > ORIGIN_MAX_AGE_S:
            found.append(f"origin 已是 {age_s / 3600:.0f} 小时前，要求每天一次")
        if origin.get("mcu_us") is not None and origin["mcu_us"] / 1e6 < WARMUP_S:
            found.append(f"origin 时只预热了 {origin['mcu_us'] / 6e7:.0f} 分钟，要求 30 分钟")
        if _on_fixture(origin) is False:
            found.append("origin 时不在工装姿态")
        recheck = origin.get("recheck")
        if boot_s is not None and origin["wall_s"] >= boot_s:
            found.append("origin 后还没有重新上电，新零点未生效")
        elif not recheck:
            found.append("origin 重新上电后还没有复核 raw（在工装上做一次动态校准即可记录）")
        elif not recheck["pass"]:
            raw = ", ".join(f"{v:.3f}" for v in recheck["fxyz_mxyz_raw"])
            found.append(f"origin 复核 raw 超限（{raw}）")

    if not zero or zero.get("kind") != "software":
        found.append("这次连接后还没有做动态校准" if not zero else "origin 之后还没有做动态校准")
    else:
        age_s = now_s - zero["wall_s"]
        if boot_s is not None and zero["wall_s"] < boot_s:
            found.append("BOX 重新上电后还没有做动态校准")
        elif age_s > DYNAMIC_MAX_AGE_S:
            found.append(f"动态校准已过 {age_s / 60:.0f} 分钟，要求每 30 分钟一次")
        if _on_fixture(zero) is False:
            found.append("动态校准时不在工装姿态")

    if uptime_s is not None and uptime_s < WARMUP_S:
        repowered = origin is not None and 0 < boot_s - origin["wall_s"] <= REPOWER_WINDOW_S
        if not repowered:
            found.append(f"BOX 才上电 {uptime_s / 60:.0f} 分钟，要求预热 30 分钟")
    return found


def static_mask(acc_g: np.ndarray, gyr_deg_s: np.ndarray | None) -> np.ndarray:
    acc_g = np.asarray(acc_g, dtype=float)
    ok = np.abs(np.linalg.norm(acc_g, axis=1) - 1.0) < STATIC_ACC_TOL_G
    if gyr_deg_s is not None:
        ok &= np.linalg.norm(np.asarray(gyr_deg_s, dtype=float), axis=1) < STATIC_GYR_DEG_S
    return ok


def attitude_labels(acc_g: np.ndarray) -> np.ndarray:
    """Which IMU axis points up (``"x+"`` ... ``"z-"``): the six faces of a rotation test."""
    acc_g = np.asarray(acc_g, dtype=float)
    i = np.argmax(np.abs(acc_g), axis=1)
    sign = np.where(acc_g[np.arange(len(acc_g)), i] > 0, "+", "-")
    return np.array([f"{'xyz'[k]}{s}" for k, s in zip(i, sign, strict=True)])


def _affine_fit(f: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.hstack([acc, np.ones((len(acc), 1))])
    sol, *_ = np.linalg.lstsq(X, f, rcond=None)
    return sol[:3].T, sol[3]  # K (6,3), offset (6,)


def _rms(e: np.ndarray) -> list[float]:
    return [float(v) for v in np.sqrt((e**2).mean(axis=0))]


def fit_payload(f: np.ndarray, acc_g: np.ndarray, gyr_deg_s: np.ndarray | None = None) -> dict[str, Any]:
    """Fit K from free-air samples (the BOX held by its body, nothing touching the payload).

    Returns K, the in-sample and leave-one-attitude-out residuals, and the rigid-body reading of
    K (mass from the force block's singular values, their spread as the gain anisotropy, CoM from
    the moment block). The offset the fit also finds is the current zero's bias, not a property of
    the BOX, and is reported only for comparison.
    """
    f = np.asarray(f, dtype=float)
    acc = np.asarray(acc_g, dtype=float)
    keep = static_mask(acc, gyr_deg_s)
    f, acc = f[keep], acc[keep]
    if len(f) < 30:
        raise ValueError(f"only {len(f)} quasi-static samples; hold each attitude still for 2-3 s")
    labels = attitude_labels(acc)
    poses = {lab: int((labels == lab).sum()) for lab in sorted(set(labels))}
    if len([n for n in poses.values() if n >= 5]) < 4:
        raise ValueError(f"attitudes {poses}: need at least 4 of the 6 faces (5+ samples each)")

    K, offset = _affine_fit(f, acc)
    resid = f - (acc @ K.T + offset)
    heldout: dict[str, list[float]] = {}
    for lab, n in poses.items():
        test = labels == lab
        if n < 5 or (~test).sum() < 30:
            continue
        Kh, oh = _affine_fit(f[~test], acc[~test])
        heldout[lab] = _rms(f[test] - (acc[test] @ Kh.T + oh))

    # rigid-body reading: K_F = -m*G*R  =>  -K_F = U S Vt, R = U Vt, m = mean(S)/G
    U, S, Vt = np.linalg.svd(-K[:3])
    R = U @ Vt
    proper = bool(np.linalg.det(R) > 0)
    mass = float(S.mean() / G)
    # moment block: K_M = [com]x K_F (rigid) -> least squares for com over the three columns
    KF, KM = K[:3], K[3:]
    A = np.vstack([-_skew(KF[:, j]) for j in range(3)])
    com, *_ = np.linalg.lstsq(A, np.concatenate([KM[:, j] for j in range(3)]), rcond=None)

    worst_n = max((max(v[:3]) for v in heldout.values()), default=math.nan)
    worst_nm = max((max(v[3:]) for v in heldout.values()), default=math.nan)
    return {
        "K": K.round(5).tolist(),
        "n_static": int(len(f)),
        "attitudes": poses,
        "missing_attitudes": [lab for lab in ("x+", "x-", "y+", "y-", "z+", "z-") if poses.get(lab, 0) < 5],
        "rms_in_sample": [round(v, 4) for v in _rms(resid)],
        "rms_heldout_attitude": {k: [round(x, 4) for x in v] for k, v in heldout.items()},
        "heldout_worst_force_N": round(worst_n, 4),
        "heldout_worst_moment_Nm": round(worst_nm, 4),
        "meets_target": bool(worst_n <= TARGET_HELDOUT_RMS_N),
        "rigid_reading": {
            "mass_kg": round(mass, 4),
            "axis_gain_kg": [round(float(s / G), 4) for s in S],
            "R_ft_imu": R.round(4).tolist(),
            "R_proper": proper,
            "com_in_ft_m": [round(float(c), 4) for c in com],
        },
        "zero_offset_seen": [round(float(v), 4) for v in offset],
    }


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# --- inputs ------------------------------------------------------------------------------------


def read_live_log(path: Path, *, sensor_key: str = "", t0: float | None = None, t1: float | None = None):
    """Rows of a gateway-preview log (``{"t","k","f","acc","gyr"}`` per line) -> f, acc, gyr arrays."""
    f, acc, gyr = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if sensor_key and row.get("k") != sensor_key:
            continue
        t = row.get("t")
        if (t0 is not None and t < t0) or (t1 is not None and t > t1):
            continue
        if row.get("f") and row.get("acc") and row.get("gyr"):
            f.append(row["f"])
            acc.append(row["acc"])
            gyr.append(row["gyr"])
    return np.asarray(f, dtype=float), np.asarray(acc, dtype=float), np.asarray(gyr, dtype=float)


def read_box_sensors(path: Path, device_id: int | str | None = None):
    """An episode's ``box_sensors.jsonl`` -> per-force-sample f with the nearest-earlier IMU acc.

    Returns (wall_s, f (n,6), acc (n,3)); force samples before the first IMU sample are dropped.
    With several BOXes the sids carry ``box<id>/``; ``device_id`` picks one.
    """
    prefix = f"box{int(device_id)}/" if device_id is not None else ""
    imu_t, imu_acc, ft_t, ft_f = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            sid = s.get("sid", "")
            if prefix and not sid.startswith(prefix):
                continue
            data = s.get("data") or {}
            if sid.endswith("box_imu") and data.get("acc_g"):
                imu_t.append(s["wall_s"])
                imu_acc.append(data["acc_g"])
            elif sid.endswith("box_six_d_force") and data.get("fxyz_mxyz"):
                ft_t.append(s["wall_s"])
                ft_f.append(data["fxyz_mxyz"])
    imu_t_a, ft_t_a = np.asarray(imu_t), np.asarray(ft_t)
    idx = np.searchsorted(imu_t_a, ft_t_a, side="right") - 1
    ok = idx >= 0
    return ft_t_a[ok], np.asarray(ft_f, dtype=float)[ok], np.asarray(imu_acc, dtype=float)[idx[ok]]


def compensate_episode(ep_dir: Path, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """External wrench for every BOX of an episode, from its meta's zero record and the registry.

    Returns ``{device_id: {"wall_s", "external"} | {"error"}}``; nothing is written.
    """
    registry = registry if registry is not None else load_registry()
    meta = json.loads((Path(ep_dir) / "meta.json").read_text(encoding="utf-8"))
    zeros = meta.get("box_six_d_zero") or {}
    if not zeros:
        return {
            "": {
                "error": "meta.json has no box_six_d_zero (recorded before 11-03, or no 6D zero this session)"
            }
        }
    multi = len(zeros) > 1
    out: dict[str, Any] = {}
    for did, zero in zeros.items():
        model = payload_model(registry, did, float(zero.get("wall_s") or 0.0))
        if model is None:
            out[did] = {"error": f"no fitted payload model for BOX {did} at the zero's time"}
            continue
        try:
            bias = zero_bias(model, zero)
        except ValueError as exc:
            out[did] = {"error": str(exc)}
            continue
        t, f, acc = read_box_sensors(Path(ep_dir) / "box_sensors.jsonl", did if multi else None)
        out[did] = {
            "wall_s": t,
            "external": external_wrench(f, acc, model, bias),
            "model_status": model.status,
        }
    return out


# --- CLI ---------------------------------------------------------------------------------------


def _cmd_fit(args: argparse.Namespace) -> int:
    f, acc, gyr = read_live_log(args.log, sensor_key=args.key, t0=args.t0, t1=args.t1)
    result = fit_payload(f, acc, gyr)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["meets_target"]:
        print(
            f"held-out attitude rms {result['heldout_worst_force_N']} N > {TARGET_HELDOUT_RMS_N} N: "
            "do not register this fit",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_episode(args: argparse.Namespace) -> int:
    res = compensate_episode(args.episode)
    rc = 0
    for did, r in res.items():
        if "error" in r:
            print(f"BOX {did}: {r['error']}")
            rc = 1
            continue
        ext = r["external"]
        norm = np.linalg.norm(ext[:, :3], axis=1)
        print(
            f"BOX {did} ({r['model_status']}): {len(ext)} samples, |F_ext| median {np.median(norm):.2f} N, "
            f"p95 {np.percentile(norm, 95):.2f} N, max {norm.max():.2f} N"
        )
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit", help="fit K from a free-air rotation log (gateway preview format)")
    p.add_argument("log", type=Path)
    p.add_argument("--key", default="", help="sensor key to keep, e.g. box3969377039/box_six_d_force")
    p.add_argument("--t0", type=float)
    p.add_argument("--t1", type=float)
    p.set_defaults(func=_cmd_fit)
    p = sub.add_parser("episode", help="external wrench summary of a recorded episode")
    p.add_argument("episode", type=Path)
    p.set_defaults(func=_cmd_episode)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
