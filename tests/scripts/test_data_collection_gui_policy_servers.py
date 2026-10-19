#!/usr/bin/env python

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
import torch

from tools.data_collection_gui import policy_servers
from tools.data_collection_gui import rollout as rollout_backend
from tools.fr3 import policy_remote
from tools.fr3.policy_remote import LoadedPolicy, PolicyHost

SERVER = "192.168.1.179:8790"


def test_default_port_matches_the_policy_server():
    assert policy_servers.DEFAULT_PORT == policy_remote.DEFAULT_POLICY_SERVER_PORT


def test_addresses_are_normalised_and_anything_shell_like_is_refused():
    assert policy_servers.normalise_address("192.168.1.179") == SERVER
    assert policy_servers.normalise_address(" gpu-box.lan:9001 ") == "gpu-box.lan:9001"
    for bad in ("", ":8790", "host;rm -rf /", "host:0", "host:99999", "$(id):1", "-oProxyCommand=x"):
        with pytest.raises(policy_servers.PolicyServerChoiceError):
            policy_servers.normalise_address(bad)


def test_local_gpu_is_read_from_the_driver(tmp_path):
    assert policy_servers.local_gpu(tmp_path / "missing")["available"] is False
    driver = tmp_path / "version"
    driver.write_text("NVRM version: 580")
    assert policy_servers.local_gpu(driver)["available"] is True


def test_choices_are_checked_against_the_configured_servers():
    no_gpu = {"available": False, "detail": ""}
    gpu = {"available": True, "detail": ""}
    assert policy_servers.resolve_choice("192.168.1.179", [SERVER], needs_policy=True, local=no_gpu) == SERVER
    assert policy_servers.resolve_choice("", [SERVER], needs_policy=True, local=gpu) == policy_servers.LOCAL
    # Printing settings needs no policy, so it is allowed on a GPU-less host.
    assert policy_servers.resolve_choice(None, [SERVER], needs_policy=False, local=no_gpu) == policy_servers.LOCAL
    with pytest.raises(policy_servers.PolicyServerChoiceError, match="no GPU.*192.168.1.179:8790"):
        policy_servers.resolve_choice("", [SERVER], needs_policy=True, local=no_gpu)
    with pytest.raises(policy_servers.PolicyServerChoiceError, match="Unknown policy server 10.0.0.1:8790"):
        policy_servers.resolve_choice("10.0.0.1", [SERVER], needs_policy=True, local=gpu)


def test_unload_refuses_unconfigured_addresses():
    with pytest.raises(policy_servers.PolicyServerChoiceError, match="Unknown"):
        policy_servers.unload("10.0.0.1:8790", [SERVER])


def test_listing_probes_a_live_server_and_reports_a_dead_one():
    host = PolicyHost(
        lambda spec: LoadedPolicy(spec=spec, fingerprint={}, device=torch.device("cpu"), use_amp=False,
                                  predict=lambda *a, **k: None, reset=lambda: None),
        log=lambda _line: None,
    )
    listener = socket.create_server(("127.0.0.1", 0))
    live = f"127.0.0.1:{listener.getsockname()[1]}"

    def loop():
        while True:
            try:
                conn, (addr, port) = listener.accept()
            except OSError:
                return
            with conn:
                policy_remote.serve_connection(conn, host=host, peer=f"{addr}:{port}", log=lambda _line: None)

    threading.Thread(target=loop, daemon=True).start()
    dead_sock = socket.create_server(("127.0.0.1", 0))
    dead = f"127.0.0.1:{dead_sock.getsockname()[1]}"
    dead_sock.close()
    try:
        result = policy_servers.listing([live, dead], timeout_s=2.0)
    finally:
        listener.close()
    by_address = {entry["address"]: entry for entry in result["servers"]}
    assert by_address[live]["reachable"] is True
    assert by_address[live]["status"]["loaded"] is None and by_address[live]["status"]["holder"] is None
    assert by_address[dead]["reachable"] is False and "Error" in by_address[dead]["error"]


def _command(repo_root: Path, **kwargs):
    launcher = repo_root / rollout_backend.LAUNCHER
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("#!/bin/sh\n")
    return rollout_backend.build_rollout_command(
        repo_root, mode="smoke", checkpoint_path="outputs/train/j/checkpoints/1", dataset_root="",
        target_frame_name="pika_gripper_ee", **kwargs,
    )


def test_rollout_env_sets_the_chosen_server_and_never_inherits_one(tmp_path):
    _, env = _command(tmp_path, base_env={"FR3_POLICY_SERVER": "stale:1"}, policy_server=SERVER)
    assert env["FR3_POLICY_SERVER"] == SERVER
    _, env = _command(tmp_path, base_env={"FR3_POLICY_SERVER": "stale:1"})
    assert "FR3_POLICY_SERVER" not in env
