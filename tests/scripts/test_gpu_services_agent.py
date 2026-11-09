#!/usr/bin/env python
"""The GPU services agent (tools/fr3/gpu_services_agent.py) and the gateway's side (gpu_hosts)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.data_collection_gui import gpu_hosts
from tools.fr3 import gpu_services_agent as agent_mod
from tools.fr3.gpu_services_agent import Agent, ServiceError, parse_config

WAM = {
    "id": "wam_jepa",
    "label": "WAM",
    "unitGlob": "wam-p3-jepa-serve-*",
    "cwd": "/srv/wam",
    "start": ["/usr/bin/true", "serve"],
    "port": 8791,
    "gpuGb": 12,
}
POLICY_PID, WAM_PID = 100, 200


class FakeMachine:
    """A 32 GB card with the policy server and/or the WAM unit on it; actions edit that state."""

    def __init__(self, policy=True, policy_loaded=True, wam=False, holder=None, training=()):
        self.policy = policy
        self.policy_loaded = policy_loaded
        self.wam = wam
        self.holder = holder
        self.training_runs = list(training)
        self.calls: list[str] = []
        self.extra = []

    def gpu(self):
        procs = list(self.extra)
        if self.policy:
            procs.append({"pid": POLICY_PID, "mb": 11700 if self.policy_loaded else 650})
        if self.wam:
            procs.append({"pid": WAM_PID, "mb": 11500})
        return {"name": "RTX 5090", "total_mb": 32607, "used_mb": 1400 + sum(p["mb"] for p in procs), "procs": procs}

    def units(self, glob):
        if self.wam and glob == "wam-p3-jepa-serve-*":
            return [{"unit": "wam-p3-jepa-serve-1.service", "state": "active", "main_pid": WAM_PID, "up_s": 50.0}]
        return []

    def unit_of(self, pid):
        return "wam-p3-jepa-serve-1.service" if pid == WAM_PID and self.wam else "session-2.scope"

    def cmdline(self, pid):
        return ["python", "other.py"]

    def policy_pids(self):
        return [POLICY_PID] if self.policy else []

    def process_up_s(self, pid):
        return 10.0

    def listening(self, port):
        return (port == 8790 and self.policy) or (port == 8791 and self.wam)

    def policy_status(self):
        spec = {"checkpoint": "outputs/train/job/checkpoints/045000/pretrained_model"}
        return {"loaded": {"spec": spec} if self.policy_loaded else None, "loading": None, "holder": self.holder}

    def training(self):
        return self.training_runs

    def start_policy_server(self):
        self.calls.append("start policy_server")
        self.policy, self.policy_loaded = True, False
        return "listening"

    def stop_policy_server(self, why):
        self.calls.append("stop policy_server")
        self.parked_why = why
        self.policy = False
        return "stopped"

    def launch(self, svc, log_path):
        self.calls.append(f"launch {svc.id} {svc.start}")
        self.wam = True
        return "fr3-gpu-launch-wam-jepa-1"

    def stop_units(self, units):
        self.calls.append(f"stop {sorted(units)}")
        self.wam = False


def make_agent(tmp_path, machine):
    return Agent(parse_config({"services": [WAM]}), machine, tmp_path / "gpu.log", sleep=lambda s: None)


def run_op(agent, action, service):
    agent.request(action, service, "test")
    for thread in threading.enumerate():
        if thread.name == "gpu-op":
            thread.join(5)
    return agent.snapshot()["op"]


def test_constants_match_what_they_restate():
    from tools.fr3 import fr3_policy_server, policy_remote

    assert agent_mod.TRAINING_MODULES == fr3_policy_server.TRAINING_MODULES
    assert agent_mod.TRAINING_SCRIPTS == fr3_policy_server.TRAINING_SCRIPTS
    assert agent_mod.WARMUP_HOLDER == policy_remote.WARMUP_HOLDER
    assert agent_mod.POLICY_PORT == policy_remote.DEFAULT_POLICY_SERVER_PORT
    assert gpu_hosts.AGENT_PORT == agent_mod.DEFAULT_PORT


def test_config_refuses_anything_but_a_fixed_absolute_argv():
    assert [s.id for s in parse_config({}).services] == ["policy_server"]
    for bad in (
        {**WAM, "id": "policy_server"},
        {**WAM, "id": "Bad id"},
        {**WAM, "start": "uv run serve"},
        {**WAM, "start": ["uv", "run"]},
        {**WAM, "cwd": "relative"},
        {**WAM, "unitGlob": "x; rm -rf /"},
    ):
        with pytest.raises(ValueError):
            parse_config({"services": [bad]})


def test_snapshot_assigns_gpu_memory_to_services(tmp_path):
    machine = FakeMachine(wam=True)
    machine.extra = [{"pid": 300, "mb": 900}]
    snap = make_agent(tmp_path, machine).snapshot()
    rows = {row["id"]: row for row in snap["services"]}
    assert rows["policy_server"]["state"] == "running" and rows["policy_server"]["loaded"] == "job/045000"
    assert rows["policy_server"]["gpuMb"] == 11700
    assert rows["wam_jepa"]["state"] == "running" and rows["wam_jepa"]["gpuMb"] == 11500
    assert [o["pid"] for o in snap["others"]] == [300]


def test_switch_stops_the_others_then_starts(tmp_path):
    machine = FakeMachine(wam=False)
    agent = make_agent(tmp_path, machine)
    op = run_op(agent, "switch", "wam_jepa")
    assert op["state"] == "done", op
    assert machine.calls[0] == "stop policy_server" and machine.calls[1].startswith("launch wam_jepa")
    assert machine.parked_why == "switch wam_jepa from the GPU services panel"
    machine.calls.clear()
    op = run_op(agent, "switch", "policy_server")
    assert op["state"] == "done", op
    assert machine.calls == ["stop ['wam-p3-jepa-serve-1.service']", "start policy_server"]
    assert "switch policy_server" in (tmp_path / "gpu.log").read_text()


def test_refusals(tmp_path):
    # a rollout holds the policy: neither a stop nor a switch away from it
    agent = make_agent(tmp_path, FakeMachine(holder="10.77.0.156:5"))
    for action, service in (("stop", "policy_server"), ("switch", "wam_jepa")):
        with pytest.raises(ServiceError, match="stop that rollout first"):
            agent.request(action, service, "t")
    # the server's own warm-up load is not a rollout
    assert make_agent(tmp_path, FakeMachine(holder="warmup")).plan("stop", "policy_server")
    # it would not fit beside what is there: start says to switch
    full = FakeMachine(policy_loaded=True)
    full.extra = [{"pid": 300, "mb": 9000}]
    with pytest.raises(ServiceError, match="use switch"):
        make_agent(tmp_path, full).plan("start", "wam_jepa")
    machine = FakeMachine(policy_loaded=False)
    assert make_agent(tmp_path, machine).plan("start", "wam_jepa")[0][1].id == "wam_jepa"
    with pytest.raises(ServiceError, match="training run"):
        make_agent(tmp_path, FakeMachine(policy=False, training=["pid 9 (lerobot_train.py)"])).plan("start", "wam_jepa")
    with pytest.raises(ServiceError, match="already running"):
        make_agent(tmp_path, FakeMachine()).plan("start", "policy_server")
    with pytest.raises(ServiceError, match="unknown service"):
        make_agent(tmp_path, FakeMachine()).plan("start", "rm")


def test_http_allow_list_and_gateway_forwarding(tmp_path):
    machine = FakeMachine(policy_loaded=False)
    agent = make_agent(tmp_path, machine)
    server = ThreadingHTTPServer(("127.0.0.1", 0), agent_mod.make_handler(agent))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"127.0.0.1:{server.server_address[1]}"
    try:
        status = json.loads(urllib.request.urlopen(f"http://{address}/status", timeout=5).read())
        assert status["services"][0]["id"] == "policy_server"
        # the gateway: known address only, and nothing while a rollout runs here
        policy_servers = ["127.0.0.1:8790"]
        monkey_agents = lambda servers: [address]  # noqa: E731 -- the test agent is not on 8792
        original = gpu_hosts.agent_addresses
        gpu_hosts.agent_addresses = monkey_agents
        try:
            with pytest.raises(gpu_hosts.GpuHostError, match="rollout is running"):
                gpu_hosts.act(address, policy_servers, "stop", "policy_server", rollout_running=True)
            with pytest.raises(gpu_hosts.GpuHostError, match="Unknown GPU host"):
                gpu_hosts.act("10.0.0.1:8792", policy_servers, "stop", "policy_server", rollout_running=False)
            with pytest.raises(gpu_hosts.GpuHostError, match="already running"):
                gpu_hosts.act(address, policy_servers, "start", "policy_server", rollout_running=False)
            result = gpu_hosts.act(address, policy_servers, "start", "wam_jepa", rollout_running=False)
            assert result["op"]["service"] == "wam_jepa"
            listing = gpu_hosts.listing(policy_servers)
            assert listing["hosts"][0]["reachable"] is True
        finally:
            gpu_hosts.agent_addresses = original
        agent.config.allow_from = ["10.9.9.9"]
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"http://{address}/status", timeout=5)
        assert refused.value.code == 403
    finally:
        server.shutdown()


def test_agent_addresses_follow_the_policy_servers():
    assert gpu_hosts.agent_addresses(["10.77.0.64:8790", "10.77.0.64:8795", "gpu:8790"]) == ["10.77.0.64:8792", "gpu:8792"]
    assert gpu_hosts.listing([]) == {"hosts": []}
