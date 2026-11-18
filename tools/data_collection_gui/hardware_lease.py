"""Time-share one Thor's hardware between its production gateway and the maintainer's dev gateway.

The cameras (one Argus slot), the BOXes (UDP on 0.0.0.0), the laser tracker and the FR3 arms (one FCI
client) each take exactly one owner. Production owns them unless it has lent them:

* ``run/deploy.sh <thor> --dev`` starts a dev gateway next to production (its own tree and port);
* a person presses Borrow on the dev gateway's page, the dev gateway asks production (over loopback), and
  production lends only while nothing of its own touches the hardware;
* from then on production answers its hardware endpoints with 409 (offline work goes on) until the dev
  gateway returns the hardware -- or production reclaims it from a dev gateway that no longer answers.

The lease is a JSON file in production's ``outputs/`` (deploys never sync it), written under ``flock`` by
whichever side changes it. Inside one gateway a hardware request counts as in flight from the guard to its
response, and a transfer refuses while one is, so a Connect that passed the guard cannot finish after the
hardware was lent.

Agents deploy the dev gateway but never borrow, return or reclaim (AGENTS.md section 3). Like the sandbox
guard, this protects against accidents, not against someone who edits the code.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRODUCTION = "production"
DEV = "dev"
UNKNOWN = "unknown"  # an unreadable lease file: nobody gets the hardware until production reclaims it
FILE_NAME = "hardware_lease.json"


class LeaseRefusedError(Exception):
    """A transfer that may not happen now; the message says why and what to do."""


@dataclass
class Lease:
    holder: str = PRODUCTION
    since: str = ""
    by: str = ""
    # camelCase: the GUI reads these as they are
    devUrl: str = ""  # noqa: N815
    devRepo: str = ""  # noqa: N815
    devCommit: str = ""  # noqa: N815


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_lease(path: Path) -> Lease:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Lease()
    except (OSError, ValueError):
        return Lease(holder=UNKNOWN)
    if not isinstance(raw, dict) or raw.get("holder") not in (PRODUCTION, DEV):
        return Lease(holder=UNKNOWN)
    return Lease(**{k: str(raw.get(k) or "") for k in Lease.__dataclass_fields__})


class HardwareLease:
    """One gateway's side of the lease: ``role`` is PRODUCTION or DEV."""

    def __init__(self, path: Path, role: str) -> None:
        if role not in (PRODUCTION, DEV):
            raise ValueError(f"role must be {PRODUCTION} or {DEV}, not {role!r}")
        self.path = Path(path)
        self.role = role
        self._lock = threading.Lock()
        self._inflight = 0

    def lease(self) -> Lease:
        return read_lease(self.path)

    def refusal(self) -> str | None:
        """Why this gateway may not touch the hardware now, or None."""
        lease = self.lease()
        if lease.holder == self.role:
            return None
        if lease.holder == UNKNOWN:
            return (f"the hardware lease ({self.path}) is unreadable, so neither gateway uses the hardware; "
                    "reclaim it on the production gateway's page")
        if self.role == PRODUCTION:
            return (f"the hardware is lent to the dev gateway ({lease.devUrl or 'dev'}, since {lease.since}"
                    f"{', by ' + lease.by if lease.by else ''}); it comes back when the dev gateway returns it, "
                    "or reclaim it from the banner on this page")
        return ("this dev gateway has not borrowed the hardware: press Borrow in the banner (production lends "
                "it only while it is not using it)")

    def enter(self) -> str | None:
        """Start a hardware request: None and counted in flight, or the refusal (not counted)."""
        with self._lock:
            refusal = self.refusal()
            if refusal is None:
                self._inflight += 1
            return refusal

    def exit(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def transfer(
        self,
        new: Lease,
        *,
        expect: tuple[str, ...],
        busy: Callable[[], list[str]],
        before_write: Callable[[], None] | None = None,
    ) -> Lease:
        """Write ``new`` when the lease is held by one of ``expect`` and ``busy()`` reports nothing.

        ``busy`` names what of this gateway still uses the hardware; ``before_write`` releases what may
        simply be dropped (camera previews). Both run with this side's requests held off.
        """
        with self._lock:
            if self._inflight:
                raise LeaseRefusedError("a hardware request on this gateway is still being answered; try again in a moment")
            with self._file_lock():
                current = self.lease()
                if current.holder not in expect:
                    raise LeaseRefusedError(f"the hardware is held by {current.holder}, not by {' or '.join(expect)}")
                problems = busy()
                if problems:
                    raise LeaseRefusedError("still using the hardware: " + "; ".join(problems))
                if before_write is not None:
                    before_write()
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(asdict(new), indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, self.path)
        return new

    def payload(self) -> dict[str, Any]:
        lease = self.lease()
        return {"role": self.role, **asdict(lease), "mine": lease.holder == self.role, "refusal": self.refusal()}


# --- talking to the other gateway ------------------------------------------------------------------


def post_peer(url: str, body: dict[str, Any], timeout_s: float = 10.0) -> tuple[int, dict[str, Any]]:
    """POST JSON to the other gateway on this machine: (HTTP status, JSON body); status 0 = no answer.

    No proxy: a lab shell's http_proxy must never route a loopback call elsewhere.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return response.status, _json_or_empty(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json_or_empty(exc.read())
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, {}


def _json_or_empty(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# --- a dev gateway that no longer answers ----------------------------------------------------------


def processes_in(tree: Path) -> list[int]:
    """Python processes running in ``tree`` (their working directory is it or below it).

    Every hardware process a gateway starts (recorder, replays, rollout jobs, pivot) runs with the
    gateway's tree as its working directory, and so does the gateway itself.
    """
    root = Path(tree).resolve()
    if root == Path("/") or len(root.parts) < 3:
        return []  # never sweep a whole filesystem or home directory
    own = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in (own, 1):
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                argv0 = handle.read().split(b"\0", 1)[0].decode("utf-8", "ignore")
            cwd = Path(os.readlink(f"/proc/{entry}/cwd"))
        except OSError:
            continue
        if not os.path.basename(argv0).startswith("python"):
            continue
        if cwd == root or root in cwd.parents:
            found.append(int(entry))
    return sorted(found)


def stop_processes(pids: list[int], timeout_s: float = 5.0) -> list[int]:
    """SIGTERM, then SIGKILL what is left; returns the pids still alive."""
    own = os.getpid()
    targets = [int(p) for p in pids if isinstance(p, int) and int(p) > 1 and int(p) != own]

    def alive() -> list[int]:
        out = []
        for pid in targets:
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                    if handle.read().rsplit(")", 1)[-1].split()[0] == "Z":
                        continue  # a zombie holds nothing; its parent reaps it
            except OSError:
                continue
            out.append(pid)
        return out

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in alive():
            with suppress(OSError):
                os.kill(pid, sig)
        deadline = time.monotonic() + timeout_s
        while alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not alive():
            return []
    return alive()
