"""Which FR3 is which: one entry per arm, and no default arm anywhere.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, B2. Until 10-26 every tool that drove an FR3 fell back to
192.168.11.102 when it was not told otherwise, and the gateway filled its IP box with the same. With one arm
that was a convenience; with two it is the way the wrong arm moves. So:

* ``fr3_arms.json`` holds each arm's identity and mounting: ``robot_ip``, the gripper BOX
  (``corenetic_device_id`` -- 10-09 an unpinned BOX discovery picked the handheld), the end-effector
  ``payload`` Desk should hold, ``start_joints_rad`` and its ``replay_config``. The replay YAML still holds the
  controller settings; :func:`validate` refuses a registry whose IP or BOX disagrees with it.
* :func:`resolve` is what every entry point calls. It takes an arm name or an IP and returns the arm; given
  neither it refuses, listing the arms in service. An arm that is not in service is refused with its reason.
* :func:`payload_check` compares what Desk holds (``m_ee``, ``F_x_Cee`` from the robot state) with the
  arm's declared payload, so a Desk set for the other arm -- or for the gripper that is away -- is caught
  before the arm moves rather than as a sag in the replay error.

Stdlib only (YAML read when available): the gateway runs in Thor's system python.
"""

from __future__ import annotations

import ipaddress
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "fr3_arms/v1"
REGISTRY = Path("tools") / "fr3" / "fr3_arms.json"
ARMS = ("right", "left")
#: Desk vs declared payload: beyond these the arm's gravity model is off by more than the 10-12
#: identification could tell apart (its repeat spread was ~0.02 kg, a few mm).
PAYLOAD_TOL_KG = 0.05
PAYLOAD_TOL_COM_M = 0.01


class ArmConfigError(ValueError):
    """No arm, an ambiguous arm, an arm not in service, or a registry that does not hold together."""


def _replay_robot(repo_root: Path, config: str) -> dict[str, Any] | None:
    """``robot:`` of a replay YAML, or None when it cannot be read here (no file, no PyYAML)."""
    path = repo_root / config
    try:
        import yaml
    except ImportError:
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    robot = payload.get("robot") if isinstance(payload, Mapping) else None
    return dict(robot) if isinstance(robot, Mapping) else None


def _ipv4(value: Any) -> bool:
    try:
        return ipaddress.ip_address(str(value)).version == 4
    except ValueError:
        return False


def validate(registry: Mapping[str, Any], *, repo_root: Path | str = _REPO_ROOT) -> list[str]:
    """Everything wrong with ``registry``; empty when it can be used."""
    repo_root = Path(repo_root)
    problems: list[str] = []
    if registry.get("schema") != SCHEMA:
        problems.append(f"schema is {registry.get('schema')!r}, expected {SCHEMA!r}")
    arms = registry.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        return [*problems, "no arms"]
    for name, arm in arms.items():
        if name not in ARMS:
            problems.append(f"arm {name!r} is not one of {ARMS}")
            continue
        if not arm.get("in_service"):
            if not str(arm.get("why_not") or "").strip():
                problems.append(f"arm {name}: not in service and no why_not")
            continue
        if not _ipv4(arm.get("robot_ip")):
            problems.append(f"arm {name}: robot_ip {arm.get('robot_ip')!r} is not an IPv4 address")
        if not isinstance(arm.get("corenetic_device_id"), int):
            problems.append(f"arm {name}: corenetic_device_id must be the arm's gripper BOX id")
        start = arm.get("start_joints_rad")
        if not (isinstance(start, Sequence) and len(start) == 7 and all(math.isfinite(float(v)) for v in start)):
            problems.append(f"arm {name}: start_joints_rad must be 7 finite joint angles")
        payload = arm.get("payload")
        if payload is not None and not (
            isinstance(payload.get("mass_kg"), (int, float))
            and len(payload.get("com_m") or []) == 3
            and str(payload.get("source") or "").strip()
        ):
            problems.append(f"arm {name}: payload needs mass_kg, com_m [x, y, z] and its source")
        config = str(arm.get("replay_config") or "")
        if not (repo_root / config).is_file() or not config:
            problems.append(f"arm {name}: replay_config {config!r} does not exist")
            continue
        robot = _replay_robot(repo_root, config)
        if robot is None:
            continue
        for key in ("robot_ip", "corenetic_device_id"):
            if str(robot.get(key)) != str(arm.get(key)):
                problems.append(f"arm {name}: {key} {arm.get(key)!r} but {config} says {robot.get(key)!r}")
    serving = {n: a for n, a in arms.items() if a.get("in_service")}
    for key in ("robot_ip", "corenetic_device_id"):
        values = [str(a.get(key)) for a in serving.values()]
        if len(values) != len(set(values)):
            problems.append(f"two arms share {key}: {values}")
    return problems


def load(path: Path | str | None = None, *, repo_root: Path | str = _REPO_ROOT) -> dict[str, Any]:
    """The registry, refused unless :func:`validate` passes."""
    path = Path(repo_root) / REGISTRY if path is None else Path(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(registry, repo_root=repo_root)
    if problems:
        raise ArmConfigError(f"{path}: " + "; ".join(problems))
    return registry


def in_service(registry: Mapping[str, Any]) -> list[str]:
    return [name for name in ARMS if (registry["arms"].get(name) or {}).get("in_service")]


def arm_for_ip(registry: Mapping[str, Any], robot_ip: str) -> str | None:
    return next(
        (n for n, a in registry["arms"].items() if a.get("in_service") and str(a.get("robot_ip")) == robot_ip),
        None,
    )


def resolve(registry: Mapping[str, Any], arm: str | None = None, robot_ip: str | None = None) -> dict[str, Any]:
    """The arm an entry point was told to drive: ``{"arm", **its entry}``.

    ``arm`` and ``robot_ip`` may both be given (they must agree) or either; neither is refused -- there is no
    default arm. An IP that is no arm's is refused too: an FR3 that drives must be in the registry.
    """
    arm = str(arm or "").strip().lower() or None
    robot_ip = str(robot_ip or "").strip() or None
    serving = in_service(registry)
    choices = f"in service: {', '.join(serving) or 'none'}"
    if arm is None and robot_ip is None:
        raise ArmConfigError(f"choose an arm ({choices}); there is no default arm")
    if arm is not None:
        if arm not in registry["arms"]:
            raise ArmConfigError(f"no arm {arm!r} ({choices})")
        entry = registry["arms"][arm]
        if not entry.get("in_service"):
            raise ArmConfigError(f"the {arm} arm is not in service: {entry.get('why_not')}")
        if robot_ip is not None and robot_ip != str(entry.get("robot_ip")):
            raise ArmConfigError(f"the {arm} arm is {entry.get('robot_ip')}, not {robot_ip}")
        return {"arm": arm, **entry}
    found = arm_for_ip(registry, robot_ip)
    if found is None:
        raise ArmConfigError(f"{robot_ip} is no FR3 in {REGISTRY} ({choices})")
    return {"arm": found, **registry["arms"][found]}


def add_arguments(parser: Any) -> None:
    """``--arm`` and ``--robot-ip`` for a CLI that drives one FR3; resolve them with :func:`resolve_args`."""
    parser.add_argument("--arm", choices=ARMS, default=None, help=f"which FR3 ({REGISTRY}); no default")
    parser.add_argument("--robot-ip", default=None, help="the FR3's IP instead of --arm; must be a registered arm")


def resolve_args(args: Any, *, repo_root: Path | str = _REPO_ROOT) -> dict[str, Any]:
    """Resolve a CLI's ``--arm`` / ``--robot-ip`` and write the IP back to ``args.robot_ip``."""
    entry = resolve(load(repo_root=repo_root), getattr(args, "arm", None), getattr(args, "robot_ip", None))
    args.arm = entry["arm"]
    args.robot_ip = str(entry["robot_ip"])
    return entry


def payload_check(
    entry: Mapping[str, Any], m_ee: float, F_x_Cee: Sequence[float]  # noqa: N803
) -> tuple[bool, str]:
    """Does Desk hold the arm's declared payload? Undeclared passes and says what Desk holds."""
    held = f"Desk holds m_ee {float(m_ee):.3f} kg, F_x_Cee ({', '.join(f'{v * 1e3:.1f}' for v in F_x_Cee)}) mm"
    declared = entry.get("payload")
    if not declared:
        return True, f"{held}; no payload declared for the {entry.get('arm')} arm"
    dm = abs(float(m_ee) - float(declared["mass_kg"]))
    dc = math.dist([float(v) for v in F_x_Cee], [float(v) for v in declared["com_m"]])
    ok = dm <= PAYLOAD_TOL_KG and dc <= PAYLOAD_TOL_COM_M
    verdict = "matches" if ok else "does NOT match"
    return ok, (
        f"{held}; {verdict} the declared {float(declared['mass_kg']):.3f} kg "
        f"(|dm| {dm:.3f} kg, |dCoM| {dc * 1e3:.1f} mm; {declared['source']})"
    )


def summary(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What a GUI lists: every arm, in service or not, with what it would drive."""
    rows = []
    for name in ARMS:
        arm = registry["arms"].get(name)
        if arm is None:
            continue
        rows.append(
            {
                "arm": name,
                "inService": bool(arm.get("in_service")),
                "whyNot": str(arm.get("why_not") or ""),
                "robotIp": str(arm.get("robot_ip") or ""),
                "gripperBox": arm.get("corenetic_device_id"),
                "payload": arm.get("payload"),
                "startJointsRad": arm.get("start_joints_rad"),
                "replayConfig": str(arm.get("replay_config") or ""),
            }
        )
    return rows
