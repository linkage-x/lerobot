"""Where a rollout's forward pass runs: on this machine's GPU, or on a policy server.

A robot host without a GPU (the NUC) keeps the arm, the cameras and the control loop, and sends
each chunk request to `tools/fr3/fr3_policy_server.py` on a machine that has one. The gateway is
started with the servers it may use (`--policy-server`, from `run/deploy.sh`); the Rollout page
picks one per launch, and the choice is checked here against that list rather than trusted: it
becomes a network address the runtime connects to with a moving arm behind it.

Kept apart from gateway.py, and the policy-server client imported only when a server is actually
asked something, because that client is torch's serialiser and the gateway otherwise never loads
torch.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LOCAL = ""
# tools.fr3.policy_remote.DEFAULT_POLICY_SERVER_PORT, restated so that parsing the gateway's own
# arguments does not import torch; a test holds the two equal.
DEFAULT_PORT = 8790
# Host names and IPv4 addresses only: this string reaches a socket and a subprocess environment.
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
NVIDIA_DRIVER_PROC = Path("/proc/driver/nvidia/version")


class PolicyServerChoiceError(ValueError):
    """A choice the operator can fix, reported as a 4xx."""


def normalise_address(value: str) -> str:
    """`host` or `host:port` as `host:port`; anything else is refused."""
    text = str(value or "").strip()
    host, sep, port_text = text.rpartition(":")
    if not sep:
        host, port_text = text, str(DEFAULT_PORT)
    if not _HOST_RE.match(host) or not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise PolicyServerChoiceError(f"Not a policy server address (host[:port]): {value!r}")
    return f"{host}:{int(port_text)}"


def local_gpu(driver_proc: Path = NVIDIA_DRIVER_PROC) -> dict[str, Any]:
    """Whether this machine can run the forward pass itself.

    The driver's proc entry rather than `torch.cuda.is_available()`: the answer is needed to draw a
    page, and importing torch for it would cost the gateway more than the question is worth. A
    machine with a loaded NVIDIA driver is one the runtime will find a GPU on.
    """
    if driver_proc.exists():
        return {"available": True, "detail": "NVIDIA driver loaded on this machine"}
    return {"available": False, "detail": "no NVIDIA GPU on this machine -- choose a policy server"}


def probe(address: str, timeout_s: float = 1.5) -> dict[str, Any]:
    """One server's state, without becoming its client: it may be serving a rollout right now."""
    try:
        from tools.fr3.policy_remote import query_policy_server

        reply = query_policy_server(address, "status", timeout_s=timeout_s)
    except Exception as exc:  # unreachable, refused, an old protocol: all mean "not usable now"
        return {"address": address, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"address": address, "reachable": True, "status": reply.get("status") or {}}


def listing(addresses: list[str], timeout_s: float = 1.5) -> dict[str, Any]:
    """This machine and every configured server, probed in parallel so one dead host costs one timeout."""
    servers: list[dict[str, Any]] = []
    if addresses:
        with ThreadPoolExecutor(max_workers=min(8, len(addresses))) as pool:
            servers = list(pool.map(lambda address: probe(address, timeout_s), addresses))
    return {"local": local_gpu(), "servers": servers}


def unload(address: str, configured: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    if address not in configured:
        raise PolicyServerChoiceError(f"Unknown policy server {address!r}.")
    from tools.fr3.policy_remote import query_policy_server

    try:
        reply = query_policy_server(address, "unload", timeout_s=timeout_s)
    except Exception as exc:
        raise PolicyServerChoiceError(f"{address}: {exc}") from exc
    return {"ok": True, "address": address, "unloaded": bool(reply.get("unloaded"))}


def resolve_choice(
    choice: Any, configured: list[str], *, needs_policy: bool, local: dict[str, Any] | None = None
) -> str:
    """The address the runtime will be given for `choice`, or LOCAL for this machine.

    `needs_policy` is false for the modes that only print settings: those are allowed anywhere,
    because they are how an operator finds out what a rollout on this machine would use.
    """
    text = str(choice or "").strip()
    if not text:
        local = local if local is not None else local_gpu()
        if needs_policy and not local["available"]:
            options = ", ".join(configured) or "none configured (deploy with a policy server)"
            raise PolicyServerChoiceError(
                f"This machine has no GPU to run the policy on. Choose a policy server: {options}."
            )
        return LOCAL
    address = normalise_address(text)
    if address not in configured:
        raise PolicyServerChoiceError(
            f"Unknown policy server {address}. This gateway was started with: "
            + (", ".join(configured) or "none")
            + "."
        )
    return address
