"""The GPU services of each policy host, as the Rollout page sees and switches them.

Every policy server the gateway is deployed with (`--policy-server host:port`) is on a machine
that runs `tools/fr3/gpu_services_agent.py` on AGENT_PORT: it reports what holds that card (the
policy server, a WAM server, anything else) and starts, stops or switches the services it was
configured with there. This module is the gateway's side: it probes the agents and forwards an
action, refusing any while a rollout runs on this robot host -- the agent cannot see that a WAM
session belongs to one.

The agent's host is the policy server's, so no address is configured twice; a host without an
agent is listed as unreachable and its policy server works as before.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# tools.fr3.gpu_services_agent.DEFAULT_PORT, restated so the gateway does not import the agent; a
# test holds the two equal.
AGENT_PORT = 8792
ACTIONS = ("start", "stop", "switch")
_SERVICE_RE = re.compile(r"^[a-z0-9_]{1,40}$")


class GpuHostError(ValueError):
    """A request the operator can fix, reported as a 4xx."""


def agent_addresses(policy_servers: list[str]) -> list[str]:
    """One agent per policy server host, in the policy servers' order."""
    seen: list[str] = []
    for address in policy_servers:
        host = address.rpartition(":")[0] or address
        agent = f"{host}:{AGENT_PORT}"
        if agent not in seen:
            seen.append(agent)
    return seen


def _call(address: str, method: str, path: str, body: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://{address}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:  # the agent's refusals carry their reason
        try:
            payload = json.loads(exc.read() or b"{}")
        except ValueError:
            payload = {}
        raise GpuHostError(f"{address}: {payload.get('error') or exc}") from exc


def probe(address: str, timeout_s: float = 2.0) -> dict[str, Any]:
    try:
        status = _call(address, "GET", "/status", None, timeout_s)
    except Exception as exc:  # down, refused, not deployed there yet: all mean "no switching from here"
        return {"address": address, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"address": address, "reachable": True, "status": status}


def listing(policy_servers: list[str], timeout_s: float = 2.0) -> dict[str, Any]:
    agents = agent_addresses(policy_servers)
    hosts: list[dict[str, Any]] = []
    if agents:
        with ThreadPoolExecutor(max_workers=min(8, len(agents))) as pool:
            hosts = list(pool.map(lambda address: probe(address, timeout_s), agents))
    return {"hosts": hosts}


def act(
    address: str, policy_servers: list[str], action: str, service: str, *, rollout_running: bool, timeout_s: float = 10.0
) -> dict[str, Any]:
    if address not in agent_addresses(policy_servers):
        raise GpuHostError(f"Unknown GPU host {address!r}; this gateway knows {', '.join(agent_addresses(policy_servers)) or 'none'}.")
    if action not in ACTIONS:
        raise GpuHostError(f"Unknown action {action!r}.")
    if not _SERVICE_RE.match(service):
        raise GpuHostError(f"Not a service id: {service!r}.")
    if rollout_running:
        raise GpuHostError("A rollout is running on this robot host; stop it before changing what the GPU serves.")
    try:
        reply = _call(address, "POST", "/action", {"action": action, "service": service}, timeout_s)
    except (OSError, ValueError) as exc:
        if isinstance(exc, GpuHostError):
            raise
        raise GpuHostError(f"{address}: the GPU services agent did not answer ({exc})") from exc
    return {"ok": True, "address": address, "op": reply.get("op")}
