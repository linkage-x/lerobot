"""The station's own 6D payload records: registered from the calibration page, kept out of the repo.

``tools/thor/box_devices.json`` (the repo) says which BOXes exist and holds the payload models (K) a
maintainer reviewed. A hand-held BOX's K changes whenever someone mounts something on its gripper,
and waiting for a pull request and a production deploy for every re-fit blocks the export. So a
station also keeps an append-only log of its own records, in ``outputs/`` (which a deploy does not
touch):

    outputs/calibration/box_six_d/registry.jsonl     one JSON event per line

    {"event": "register", "device_id": "3906437858", "at_unix": ..., "record": {six_d_payload entry}}
    {"event": "retire", "device_id": "...", "valid_from_unix": ..., "registered_unix": ... | null,
     "reason": "...", "by": "...", "at_unix": ...}

:func:`load_merged` lays the log over the repo registry; everything that picks a K (the export, the
recorder's live external wrench, the arm-held zero) reads the merged view, so a registration is in
force at once. Which record is in force at time ``t``: the latest ``valid_from`` not after ``t``; of
several records with that ``valid_from`` (a re-fit of the same mounted configuration), the last
registered one that is not retired. If all of them are retired, nothing is in force and the export
refuses: retiring a bad K must not fall back to an older configuration's K silently.

Who may write what:

* The calibration page registers a record for a **hand-held** BOX only after
  :func:`self_service_refusals` finds nothing wrong with the fit (held-out face <= 0.3 N, all six
  faces, the BOX warm, the drift check measured and passed). Nobody signs it off; the gate does.
* An arm BOX's K (the FR3 lift fit) stays maintainer-only: reviewed in a pull request to the repo.
* Retiring a record (``retire`` below) is a maintainer's action. An operator who registered a bad K
  registers a better one for the same mount time instead; it replaces the old one for every export
  after it.

Numpy-free on purpose: the gateway (system python, no numpy) reads and writes it too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO / "tools" / "thor" / "box_devices.json"
LOCAL_SUBPATH = Path("outputs") / "calibration" / "box_six_d" / "registry.jsonl"
LOCAL_REGISTRY_PATH = REPO / LOCAL_SUBPATH
SCHEMA = "box_devices/v1"
MODEL = "affine_acc/v1"
EVENTS = ("register", "retire")
FACES = ("x+", "x-", "y+", "y-", "z+", "z-")

# The self-service gate. Same numbers as box_six_d (a test keeps them equal; that module needs numpy).
TARGET_HELDOUT_N = 0.3  # box_six_d.TARGET_HELDOUT_RMS_N
DRIFT_LIMIT_N = 0.3  # box_six_d.DRIFT_WARN_N
WARMUP_S = 30 * 60  # box_six_d.WARMUP_S, the vendor manual's warm-up
SELF_SERVICE_ROLES = ("handheld",)
#: A mount time before this is a typo (2023-11), not a configuration.
EARLIEST_VALID_FROM = 1.7e9
CN_TZ = timezone(timedelta(hours=8))


def _stamp(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=CN_TZ).strftime("%Y-%m-%d %H:%M:%S +08:00")


# --- reading -----------------------------------------------------------------------------------


def read_events(path: Path) -> list[dict[str, Any]]:
    """The log's events in order; [] when there is none. A line that does not parse is an error.

    Not skipped: a lost ``retire`` line would bring a retired K back into force without a word.
    """
    path = Path(path)
    if not path.is_file():
        return []
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not JSON ({exc.msg})") from exc
        if not isinstance(event, dict) or event.get("event") not in EVENTS or not str(event.get("device_id", "")).isdigit():
            raise ValueError(f"{path}:{n}: not a register/retire event with a device_id")
        events.append(event)
    return events


def _same_record(record: dict[str, Any], valid_from: int, registered: int | None) -> bool:
    if int(record.get("valid_from_unix", -1)) != valid_from:
        return False
    mine = record.get("registered_unix")
    return (mine is None and registered is None) or (mine is not None and registered is not None and int(mine) == registered)


def merge(registry: dict[str, Any], events: list[dict[str, Any]], *, source: str = str(LOCAL_SUBPATH)) -> dict[str, Any]:
    """A copy of the repo ``registry`` with the station's ``events`` applied (see the module doc)."""
    out = json.loads(json.dumps(registry))
    devices = out.setdefault("devices", {})
    for entry in devices.values():
        for record in entry.get("six_d_payload") or []:
            record.setdefault("store", "repo")
    applied = 0
    for event in events:
        did = str(int(event["device_id"]))
        entry = devices.get(did)
        if entry is None:
            raise ValueError(f"{source}: BOX {did} is not in tools/thor/box_devices.json")
        records = entry.setdefault("six_d_payload", [])
        if event["event"] == "register":
            record = dict(event.get("record") or {})
            if record.get("registered_unix") is None or record.get("valid_from_unix") is None:
                raise ValueError(f"{source}: a register event for BOX {did} without valid_from_unix/registered_unix")
            record["store"] = "local"
            records.append(record)
        else:
            vf = int(event["valid_from_unix"])
            reg = event.get("registered_unix")
            reg = None if reg is None else int(reg)
            hits = [r for r in records if _same_record(r, vf, reg)]
            if not hits:
                raise ValueError(f"{source}: retire names no record of BOX {did} (valid_from {vf}, registered {reg})")
            for r in hits:
                r["retired"] = {k: event.get(k) for k in ("at_unix", "reason", "by")}
        records.sort(key=lambda r: (float(r["valid_from_unix"]), float(r.get("registered_unix") or 0)))
        applied += 1
    out["local_store"] = {"path": source, "events": applied}
    return out


def load_repo(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {reg.get('schema')!r}, expected {SCHEMA!r}")
    return reg


def load_merged(repo_root: Path | None = None) -> dict[str, Any]:
    """The repo registry with this station's log over it (``repo_root``: the checkout, default this one)."""
    root = REPO if repo_root is None else Path(repo_root)
    local = root / LOCAL_SUBPATH
    return merge(load_repo(root / "tools" / "thor" / "box_devices.json"), read_events(local), source=str(LOCAL_SUBPATH))


# --- which record is in force ------------------------------------------------------------------


def select(entry: dict[str, Any] | None, at_unix: float) -> tuple[dict[str, Any] | None, str]:
    """``(record, "")`` for the record in force at ``at_unix``, else ``(None, why)``."""
    if not entry:
        return None, "the BOX is not in the registry"
    eligible = [r for r in entry.get("six_d_payload") or [] if float(r.get("valid_from_unix") or 0) <= at_unix]
    if not eligible:
        return None, "no payload record valid at that time"
    latest = max(float(r["valid_from_unix"]) for r in eligible)
    group = [r for r in eligible if float(r["valid_from_unix"]) == latest]
    live = [r for r in group if not r.get("retired")]
    if not live:
        why = "; ".join(str((r.get("retired") or {}).get("reason") or "no reason") for r in group)
        return None, f"the record valid from {_stamp(latest)} was retired ({why}); register a new fit for that mount"
    record = max(live, key=lambda r: float(r.get("registered_unix") or 0))
    k = record.get("K")
    if record.get("model") != MODEL or not isinstance(k, list) or len(k) != 6 or any(len(row) != 3 for row in k):
        return None, f"the configuration mounted {_stamp(latest)} has no fitted K yet"
    return record, ""


def in_force(registry: dict[str, Any], device_id: int | str, at_unix: float) -> tuple[dict[str, Any] | None, str]:
    return select((registry.get("devices") or {}).get(str(int(device_id))), at_unix)


# --- registering -------------------------------------------------------------------------------


def payload_record(fit: dict[str, Any], *, end_effector: str, valid_from_unix: float, status: str, source: str) -> dict[str, Any]:
    """A ``six_d_payload`` entry from a box_six_d fit result."""
    k = fit.get("K")
    if not isinstance(k, list) or len(k) != 6 or any(not isinstance(row, list) or len(row) != 3 for row in k):
        raise ValueError("the fit has no 6x3 K")
    return {
        "valid_from_unix": int(valid_from_unix),
        "valid_from": _stamp(valid_from_unix),
        "end_effector": end_effector,
        "model": MODEL,
        "K": k,
        "status": status,
        "fit": {
            "log": source,
            "n_static": fit.get("n_static"),
            "attitudes": fit.get("attitudes"),
            "heldout_worst_force_N": fit.get("heldout_worst_force_N"),
            "heldout_worst_moment_Nm": fit.get("heldout_worst_moment_Nm"),
            "rigid_reading": {
                k2: v for k2, v in (fit.get("rigid_reading") or {}).items() if k2 in ("mass_kg", "axis_gain_kg", "com_in_ft_m")
            },
        },
    }


def self_service_refusals(
    fit: dict[str, Any],
    *,
    device_id: int | str,
    registry: dict[str, Any],
    valid_from_unix: float,
    capture_started_unix: float | None,
    now_unix: float,
) -> list[str]:
    """Why a fit may NOT be registered from the page without a maintainer ([] = it may)."""
    did = str(int(device_id))
    entry = (registry.get("devices") or {}).get(did)
    if entry is None:
        return [f"BOX {did} 不在 tools/thor/box_devices.json 里（新 BOX 先由维护者登记角色）"]
    role = str(entry.get("role") or "")
    if role not in SELF_SERVICE_ROLES:
        return [f"BOX {did} 是 {role or '未知角色'}，不是手持 BOX：臂上 BOX 的 K 由维护者审过后提交到仓库"]
    if fit.get("error") or fit.get("K") is None:
        return [f"拟合被拒：{fit.get('error') or '没有 K'}"]
    out: list[str] = []
    worst = fit.get("heldout_worst_force_N")
    if worst is None or not float(worst) <= TARGET_HELDOUT_N:
        out.append(f"留一面最差 {worst} N，要 ≤ {TARGET_HELDOUT_N} N")
    missing = [f for f in FACES if f in (fit.get("missing_attitudes") or [])] or (
        [] if fit.get("missing_attitudes") is not None else ["?"]
    )
    if missing:
        out.append(f"六个姿态要都采到，缺 {'、'.join(missing)}")
    uptime = (fit.get("box_uptime_s") or {}).get("start_s")
    if uptime is None:
        out.append("拟合里没有 BOX 上电时长（旧采集或旧日志），重新采一次")
    elif float(uptime) < WARMUP_S:
        out.append(f"开始采集时 BOX 才上电 {float(uptime) / 60:.0f} 分钟，要预热 {WARMUP_S // 60} 分钟")
    drift = fit.get("drift_same_attitude")
    if not isinstance(drift, dict) or not drift.get("pairs"):
        out.append("没有隔 30 s 以上回到同一朝向，漂移估计不了：采完六面后回到第一个姿态再静止 3 秒")
    elif float(drift.get("worst_force_N") or 0.0) > DRIFT_LIMIT_N:
        out.append(f"同一朝向漂了 {float(drift['worst_force_N']):.2f} N（> {DRIFT_LIMIT_N} N），多半没预热好")
    if not EARLIEST_VALID_FROM <= valid_from_unix <= now_unix + 60:
        out.append("装上时间不对（未来或太早）")
    if capture_started_unix is not None and valid_from_unix > float(capture_started_unix) + 60:
        out.append("装上时间晚于这次采集：采集时它已经装着了")
    live = [r for r in entry.get("six_d_payload") or [] if not r.get("retired")]
    if live:
        newest = max(float(r["valid_from_unix"]) for r in live)
        if valid_from_unix < newest:
            out.append(
                f"装上时间早于这只 BOX 现有最新的登记（{_stamp(newest)}）：自助登记只能接在最新配置之后，"
                "或与它同一时刻（同一配置重新拟合）；改更早的历史找维护者"
            )
    return out


def append_event(path: Path, event: dict[str, Any]) -> None:
    """One line, appended and flushed to disk (the log is never rewritten)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def register_event(device_id: int | str, record: dict[str, Any], *, now_unix: float | None = None) -> dict[str, Any]:
    now = time.time() if now_unix is None else now_unix
    record = {**record, "registered_unix": int(now), "registered": _stamp(now)}
    return {"event": "register", "device_id": str(int(device_id)), "at_unix": round(now, 3), "record": record}


# --- maintainer CLI ----------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    reg = load_merged(args.repo_root)
    now = time.time()
    for did, entry in (reg.get("devices") or {}).items():
        current, why = select(entry, now)
        print(f"BOX {did} ({entry.get('role')}{' ' + entry['hand'] if entry.get('hand') else ''}): "
              f"{'in force: ' + str(current.get('valid_from')) if current else 'nothing in force (' + why + ')'}")
        for r in entry.get("six_d_payload") or []:
            reg_s = f" registered {r.get('registered')} ({r.get('registered_unix')})" if r.get("registered_unix") else ""
            retired = f"  RETIRED: {r['retired'].get('reason')}" if r.get("retired") else ""
            mark = "*" if r is current else " "
            print(f"  {mark} [{r.get('store')}] from {r.get('valid_from')} ({r.get('valid_from_unix')}){reg_s}: "
                  f"{r.get('end_effector')}; held-out {(r.get('fit') or {}).get('heldout_worst_force_N')} N{retired}")
    return 0


def _cmd_retire(args: argparse.Namespace) -> int:
    path = Path(args.repo_root) / LOCAL_SUBPATH
    event = {
        "event": "retire", "device_id": str(int(args.device_id)), "valid_from_unix": int(args.valid_from_unix),
        "registered_unix": None if args.registered_unix is None else int(args.registered_unix),
        "reason": args.reason, "by": args.by, "at_unix": round(time.time(), 3),
    }
    # check it names a record before writing it (merge raises otherwise)
    merge(load_repo(Path(args.repo_root) / "tools" / "thor" / "box_devices.json"), [*read_events(path), event])
    append_event(path, event)
    print(json.dumps(event, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo-root", type=Path, default=REPO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="every BOX's payload records (repo + this station) and which is in force now")
    p.set_defaults(func=_cmd_list)
    p = sub.add_parser("retire", help="maintainer: take a record out of force (exports that need it then refuse)")
    p.add_argument("--device-id", required=True)
    p.add_argument("--valid-from-unix", type=int, required=True)
    p.add_argument("--registered-unix", type=int, help="the station record's registered_unix (omit for a repo record)")
    p.add_argument("--reason", required=True)
    p.add_argument("--by", required=True, help="who retires it")
    p.set_defaults(func=_cmd_retire)
    args = ap.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
