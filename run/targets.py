#!/usr/bin/env python3
"""The deployment-target registry: every machine the run/ scripts reach, one file per machine.

    python3 run/targets.py list               # every target, how it is reached, which station
    python3 run/targets.py shell NAME         # TARGET_* assignments for the bash scripts (eval)
    python3 run/targets.py show NAME          # one target as resolved JSON (overrides applied)

A target file (run/targets/<name>.json) is committed and describes the machine itself: its address
on its own LAN, its hostname, where production lives on it and what profile it runs. How *you*
reach it is personal and lives in the git-ignored run/targets.local.json -- e.g. an ~/.ssh/config
alias with a ProxyJump for a machine on another LAN, or a gateway URL through a tunnel:

    {"thor2": {"ssh": "thor2-via-lab", "gateway_url": "http://127.0.0.1:18765"}}

Only those two keys can be overridden. The hostname cannot: the scripts check it over ssh before
they touch a machine, so a moved DHCP address, a cloned image or a wrong alias stops the deploy
instead of landing it on the wrong machine. See run/targets/README.md for the design.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parent
TARGETS_DIR = RUN_DIR / "targets"
LOCAL_OVERRIDES = RUN_DIR / "targets.local.json"

PROFILES = ("thor", "workstation")
REQUIRED = ("station", "user", "address", "hostname", "repo_dir", "profile", "config", "gateway_port",
            "sandboxes", "calibration_in_tree", "box_net_addr")
OPTIONAL = ("aliases", "note")
LOCAL_KEYS = ("ssh", "gateway_url")

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,23}$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]*$")
ADDRESS_RE = re.compile(r"^[A-Za-z0-9.-]+$")
HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
# An ssh destination or ~/.ssh/config alias; no spaces or shell metacharacters (it reaches ssh -- and
# the remote side of some scripts -- unquoted in places that predate this registry).
SSH_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
URL_RE = re.compile(r"^https?://[A-Za-z0-9.-]+(:[0-9]+)?/?$")
CIDR_RE = re.compile(r"^([0-9]{1,3}\.){3}[0-9]{1,3}/[0-9]{1,2}$")


class TargetError(ValueError):
    pass


def _check(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise TargetError(f"{where}: {message}")


def _validate(name: str, raw: Any, where: str) -> dict[str, Any]:
    _check(isinstance(raw, dict), where, "must be a JSON object")
    _check(NAME_RE.match(name) is not None, where, f"file name '{name}' is not a target name (a-z, 0-9, - or _)")
    missing = [key for key in REQUIRED if key not in raw]
    _check(not missing, where, f"missing {', '.join(missing)}")
    unknown = sorted(set(raw) - set(REQUIRED) - set(OPTIONAL))
    _check(not unknown, where, f"unknown key(s) {', '.join(unknown)}")
    station = raw["station"]
    _check(station is None or (isinstance(station, str) and NAME_RE.match(station) is not None), where,
           "station must be null (not part of a recording station) or a short name")
    _check(isinstance(raw["user"], str) and USER_RE.match(raw["user"]) is not None, where, "bad user")
    _check(isinstance(raw["address"], str) and ADDRESS_RE.match(raw["address"]) is not None, where, "bad address")
    _check(isinstance(raw["hostname"], str) and HOSTNAME_RE.match(raw["hostname"]) is not None, where,
           "hostname must be the machine's real hostname (ssh it and run `hostname`)")
    _check(isinstance(raw["repo_dir"], str) and raw["repo_dir"].startswith("/") and "'" not in raw["repo_dir"]
           and " " not in raw["repo_dir"], where, "repo_dir must be an absolute path without spaces or quotes")
    _check(raw["profile"] in PROFILES, where, f"profile must be one of {', '.join(PROFILES)}")
    _check(isinstance(raw["config"], str) and not raw["config"].startswith("/") and " " not in raw["config"]
           and "'" not in raw["config"], where, "config must be a repo-relative path")
    port = raw["gateway_port"]
    _check(isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535, where, "bad gateway_port")
    _check(isinstance(raw["sandboxes"], bool), where, "sandboxes must be true or false")
    _check(not raw["sandboxes"] or raw["profile"] == "thor", where, "sandboxes exist only on thor-profile targets")
    _check(isinstance(raw["calibration_in_tree"], bool), where, "calibration_in_tree must be true or false")
    box_net = raw["box_net_addr"]
    _check(box_net is None or (isinstance(box_net, str) and CIDR_RE.match(box_net) is not None), where,
           "box_net_addr must be null or an address/prefix such as 192.168.2.45/24")
    aliases = raw.get("aliases", [])
    _check(isinstance(aliases, list) and all(isinstance(a, str) and NAME_RE.match(a) for a in aliases), where,
           "aliases must be a list of target names")
    target = {key: raw[key] for key in REQUIRED}
    target["name"] = name
    target["aliases"] = list(aliases)
    target["note"] = raw.get("note")
    return target


def load_targets(targets_dir: Path = TARGETS_DIR, local_path: Path = LOCAL_OVERRIDES) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in sorted(targets_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TargetError(f"{path}: not valid JSON ({exc})") from exc
        targets[path.stem] = _validate(path.stem, raw, str(path))
    if not targets:
        raise TargetError(f"no targets in {targets_dir}")

    names: dict[str, str] = {}
    hostnames: dict[str, str] = {}
    for name, target in targets.items():
        for label in [name, *target["aliases"]]:
            _check(label not in names, str(targets_dir), f"'{label}' names both {names.get(label)} and {name}")
            names[label] = name
        other = hostnames.get(target["hostname"])
        _check(other is None, str(targets_dir),
               f"{other} and {name} both claim hostname '{target['hostname']}'; every machine needs its own "
               "(a cloned image keeps the old one: set a new hostname on the clone)")
        hostnames[target["hostname"]] = name

    overrides: dict[str, Any] = {}
    if local_path.exists():
        try:
            overrides = json.loads(local_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TargetError(f"{local_path}: not valid JSON ({exc})") from exc
        _check(isinstance(overrides, dict), str(local_path), "must be a JSON object keyed by target name")
    for target in targets.values():
        target["ssh"] = f"{target['user']}@{target['address']}"
        target["gateway_url"] = f"http://{target['address']}:{target['gateway_port']}"
        target["overridden"] = []
    for label, override in overrides.items():
        _check(label in names, str(local_path), f"'{label}' is not a target")
        _check(isinstance(override, dict), str(local_path), f"'{label}' must be an object")
        unknown = sorted(set(override) - set(LOCAL_KEYS))
        _check(not unknown, str(local_path),
               f"'{label}' may only override {', '.join(LOCAL_KEYS)} (not {', '.join(unknown)}); what the "
               "machine is belongs in the committed run/targets/ file")
        target = targets[names[label]]
        if "ssh" in override:
            _check(isinstance(override["ssh"], str) and SSH_RE.match(override["ssh"]) is not None, str(local_path),
                   f"'{label}'.ssh must be an ssh destination or ~/.ssh/config alias")
            target["ssh"] = override["ssh"]
            target["overridden"].append("ssh")
        if "gateway_url" in override:
            _check(isinstance(override["gateway_url"], str) and URL_RE.match(override["gateway_url"]) is not None,
                   str(local_path), f"'{label}'.gateway_url must look like http://host:port")
            target["gateway_url"] = override["gateway_url"].rstrip("/")
            target["overridden"].append("gateway_url")
    return targets


def resolve(label: str, targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for name, target in targets.items():
        if label == name or label in target["aliases"]:
            return target
    known = ", ".join(f"{n} (alias {', '.join(t['aliases'])})" if t["aliases"] else n for n, t in targets.items())
    raise TargetError(f"unknown deployment target '{label}'; known: {known} (run/targets/*.json)")


def shell_assignments(target: dict[str, Any]) -> str:
    values = {
        "TARGET_NAME": target["name"],
        "TARGET_STATION": target["station"] or "",
        "TARGET_SSH": target["ssh"],
        "TARGET_ADDRESS": target["address"],
        "TARGET_HOSTNAME": target["hostname"],
        "TARGET_DIR": target["repo_dir"],
        "TARGET_PROFILE": target["profile"],
        "TARGET_CONFIG": target["config"],
        "TARGET_PORT": str(target["gateway_port"]),
        "TARGET_GATEWAY_URL": target["gateway_url"],
        "TARGET_SANDBOXES": "true" if target["sandboxes"] else "false",
        "TARGET_CALIBRATION_IN_TREE": "true" if target["calibration_in_tree"] else "false",
        "TARGET_BOX_NET_ADDR": target["box_net_addr"] or "",
        "TARGET_OVERRIDDEN": ",".join(target["overridden"]),
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 2
    command, args = argv[0], argv[1:]
    try:
        targets = load_targets()
        if command == "list":
            print(f"{'TARGET':<12} {'STATION':<10} {'PROFILE':<12} {'HOSTNAME':<16} SSH / GATEWAY")
            for name, t in targets.items():
                label = name + (f" ({','.join(t['aliases'])})" if t["aliases"] else "")
                local = " [local override]" if t["overridden"] else ""
                print(f"{label:<12} {t['station'] or '-':<10} {t['profile']:<12} {t['hostname']:<16} "
                      f"{t['ssh']}  {t['gateway_url']}{local}")
            return 0
        if command in ("shell", "show") and len(args) == 1:
            target = resolve(args[0], targets)
            print(shell_assignments(target) if command == "shell" else json.dumps(target, indent=2))
            return 0
    except TargetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
