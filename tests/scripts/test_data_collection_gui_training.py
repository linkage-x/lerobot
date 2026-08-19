#!/usr/bin/env python

"""Training-run backend: host handling, the launch command, and the W&B key.

The two things worth testing here are the ones a manual run would not reveal until it had
already gone wrong: that the argv trains the view it was given rather than rebuilding it,
and that the W&B key never reaches a command line.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.data_collection_gui import training


@pytest.fixture(autouse=True)
def isolated_secrets_dir(tmp_path, monkeypatch):
    """Keep the operator's real key store out of the tests, in both directions."""
    monkeypatch.setattr(training, "SECRETS_DIR", tmp_path / "secrets")
    return tmp_path / "secrets"


def _remote_host() -> training.TrainingHost:
    return training.TrainingHost(
        id="user@10.0.0.5:/home/user/lerobot",
        label="Training box",
        kind="remote",
        sshTarget="user@10.0.0.5",
        repoDir="/home/user/lerobot",
        pythonPath=".venv-fr3/bin/python",
    )


# ------------------------------------------------------------------------ hosts ---


def test_ssh_target_must_not_be_able_to_carry_shell_metacharacters():
    """This string is interpolated into ssh commands, so it is the one input that must not."""
    assert training.validate_ssh_target(" user@10.0.0.5 ") == "user@10.0.0.5"
    for hostile in ("user@host; rm -rf /", "user@host$(id)", "user@host`id`", "nouser", "a@b c"):
        with pytest.raises(training.TrainingError):
            training.validate_ssh_target(hostile)


def test_remote_repo_dir_must_be_an_absolute_path_without_metacharacters():
    assert training.validate_remote_dir("/home/user/lerobot/") == "/home/user/lerobot"
    for hostile in ("relative/path", "/home/$USER/lerobot", "/home/user/'lerobot'", "/home/a\nb"):
        with pytest.raises(training.TrainingError):
            training.validate_remote_dir(hostile)


def test_hosts_round_trip_through_the_store_and_the_local_one_is_always_first(tmp_path):
    repo_root = tmp_path / "repo"
    training.add_remote_host("Box A", "user@10.0.0.5", "/home/user/lerobot", "")

    hosts = training.all_hosts(repo_root)

    assert hosts[0].kind == "local"
    assert hosts[0].id == training.LOCAL_HOST_ID
    assert [host.sshTarget for host in hosts[1:]] == ["user@10.0.0.5"]
    # An empty python path falls back rather than producing a command that runs `` .
    assert hosts[1].pythonPath == ".venv-fr3/bin/python"
    assert training.resolve_host(repo_root, hosts[1].id).repoDir == "/home/user/lerobot"


def test_re_adding_a_host_updates_it_instead_of_duplicating_it(tmp_path):
    training.add_remote_host("Box A", "user@10.0.0.5", "/home/user/lerobot", "")
    training.add_remote_host("Renamed", "user@10.0.0.5", "/home/user/lerobot", ".venv/bin/python")

    hosts = training.load_remote_hosts()

    assert len(hosts) == 1
    assert hosts[0].label == "Renamed"
    assert hosts[0].pythonPath == ".venv/bin/python"


def test_a_host_whose_stored_entry_is_corrupt_is_dropped_not_run(tmp_path):
    """A malformed entry must not become an ssh command; the rest of the store still loads."""
    training.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    training.hosts_store_path().write_text(
        json.dumps(
            {
                "hosts": [
                    {"sshTarget": "user@host; id", "repoDir": "/home/user/lerobot"},
                    {"sshTarget": "user@10.0.0.5", "repoDir": "relative"},
                    {"sshTarget": "user@10.0.0.6", "repoDir": "/home/user/lerobot"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert [host.sshTarget for host in training.load_remote_hosts()] == ["user@10.0.0.6"]


def test_an_unknown_host_is_refused_rather_than_silently_run_locally(tmp_path):
    with pytest.raises(training.TrainingError, match="Unknown training host"):
        training.resolve_host(tmp_path, "user@nowhere:/tmp")


# ------------------------------------------------------------------------ argv ---


def test_training_argv_trains_the_existing_view_instead_of_rebuilding_it():
    """--skip-prepare is what makes a remote training host possible.

    Rebuilding here would re-derive a delta action column from frames whose action column is
    already a delta, and it would need the source recording, which a training machine that
    only ever receives views does not have.
    """
    argv = training.build_train_argv(
        host=_remote_host(),
        view_root="/home/user/lerobot/outputs/exports/training_views/v",
        repo_id="local/v",
        job_name="v__act",
        policy="act",
        steps=20000,
        batch_size=8,
        num_workers=4,
        save_freq=5000,
        log_freq=100,
        device="auto",
        use_amp=True,
        policy_config="",
        wandb_enabled=False,
        wandb_project="",
        wandb_entity="",
    )

    assert "--skip-prepare" in argv
    assert "--overwrite-view" not in argv
    assert "--dataset-root" not in argv
    assert "--prepare-only" not in argv
    assert argv[argv.index("--view-root") + 1] == "/home/user/lerobot/outputs/exports/training_views/v"
    # The view's manifest is the single source for these; a second copy could only disagree.
    assert "--cameras" not in argv
    assert "--action-mode" not in argv
    assert argv[0] == ".venv-fr3/bin/python"


def test_training_argv_only_asks_for_wandb_when_it_is_enabled():
    common = dict(
        view_root="/repo/view",
        repo_id="local/v",
        job_name="v__act",
        policy="act",
        steps=10,
        batch_size=8,
        num_workers=4,
        save_freq=5,
        log_freq=1,
        device="cuda",
        use_amp=False,
        policy_config="",
    )
    off = training.build_train_argv(
        host=_remote_host(), wandb_enabled=False, wandb_project="p", wandb_entity="e", **common
    )
    on = training.build_train_argv(
        host=_remote_host(), wandb_enabled=True, wandb_project="p", wandb_entity="e", **common
    )

    assert "--wandb" not in off
    assert "--use-amp" not in off
    assert "--wandb" in on
    assert on[on.index("--wandb-project") + 1] == "p"
    assert on[on.index("--wandb-entity") + 1] == "e"


def test_policy_config_json_is_passed_through_as_one_argument():
    argv = training.build_train_argv(
        host=_remote_host(),
        view_root="/repo/view",
        repo_id="local/v",
        job_name="v__smolvla",
        policy="smolvla",
        steps=10,
        batch_size=2,
        num_workers=4,
        save_freq=5,
        log_freq=1,
        device="cuda",
        use_amp=False,
        policy_config='{"optimizer_lr": 1e-5}',
        wandb_enabled=False,
        wandb_project="",
        wandb_entity="",
    )

    assert argv[argv.index("--policy-config") + 1] == '{"optimizer_lr": 1e-5}'


# ---------------------------------------------------------------- launch command ---


def test_the_wandb_key_never_appears_in_the_launch_command(tmp_path):
    """Arguments are readable from the target's process list by every user on that machine.

    Locally the key goes in the child's environment; remotely the run reads it back from a
    file that push_wandb_key placed over stdin.
    """
    key = "a" * 40
    local_command, env = training.build_launch_command(
        tmp_path, training.local_host(tmp_path), ["{python}", "train.py"], wandb_key=key
    )
    assert key not in " ".join(local_command)
    assert env is not None and env["WANDB_API_KEY"] == key
    assert env["PYTHONPATH"] == "src:."
    # The placeholder is replaced, or the command would try to run a literal "{python}".
    assert "{python}" not in local_command

    remote_command, remote_env = training.build_launch_command(
        tmp_path, _remote_host(), [".venv-fr3/bin/python", "train.py"], wandb_key=key
    )
    joined = " ".join(remote_command)
    assert key not in joined
    assert remote_env is None
    assert ".wandb_key" in joined
    assert remote_command[0] == "ssh"


def test_a_remote_launch_runs_in_the_hosts_own_repo_directory(tmp_path):
    command, _ = training.build_launch_command(
        tmp_path, _remote_host(), [".venv-fr3/bin/python", "train.py"], wandb_key=""
    )

    remote_script = command[-1]
    assert remote_script.startswith("cd /home/user/lerobot &&")
    assert "PYTHONPATH=src:." in remote_script
    # Nothing to read the key from, so nothing tries to.
    assert ".wandb_key" not in remote_script


# ----------------------------------------------------------------------- wandb ---


def test_the_key_is_stored_readable_only_by_its_owner_and_never_returned():
    training.set_wandb_key("local", "k" * 40)

    path = training.wandb_key_path("local")
    assert path.exists()
    assert oct(path.stat().st_mode)[-3:] == "600"
    status = training.wandb_status("local")
    # The page learns that a key exists and which one, never the key.
    assert status == {"configured": True, "keySuffix": "kkkk", "hostId": "local"}
    assert "k" * 40 not in json.dumps(status)


def test_a_key_that_is_not_a_key_is_refused_before_it_is_stored():
    for bad in ("", "   ", "short", "has spaces in it aaaaaaaaaaaaaaaaaaaa", "k" * 200):
        with pytest.raises(training.TrainingError):
            training.set_wandb_key("local", bad)
    assert training.wandb_status("local")["configured"] is False


def test_keys_are_per_host_so_one_machines_key_is_not_used_on_another():
    remote_id = "user@10.0.0.5:/home/user/lerobot"
    training.set_wandb_key("local", "l" * 40)
    training.set_wandb_key(remote_id, "r" * 40)

    assert training.read_wandb_key("local") == "l" * 40
    assert training.read_wandb_key(remote_id) == "r" * 40
    # The id becomes a filename, so it must not be able to escape the store.
    assert training.wandb_key_path(remote_id).parent == training.SECRETS_DIR
    assert training.wandb_key_path("../../etc/passwd").parent == training.SECRETS_DIR

    training.clear_wandb_key(remote_id)
    assert training.read_wandb_key(remote_id) == ""
    assert training.read_wandb_key("local") == "l" * 40


# --------------------------------------------------------------- log parsing ---
#
# The lines below are verbatim from an ACT run on the FR3 workstation, tqdm bars included,
# because the thing that made the first version of this wrong was assuming what the output
# looked like rather than reading it.


def test_the_step_comes_from_the_progress_bar_not_the_rounded_log_line():
    """lerobot formats the step through format_big_number, so its log line says `step:1K`.

    Read literally that is step 1, and the page would sit there for a thousand steps and then
    jump. The tqdm bar glued to the front of the same line carries the exact counter.
    """
    line = (
        "Training:   6%|\u258b         | 1299/20000 [01:33<27:58, 11.14step/s]"
        "INFO 2026-08-19 16:03:36 ot_train.py:518 step:1K smpl:10K ep:20 epch:1.01 "
        "loss:1.656 grdn:59.609"
    )

    found = training.parse_progress_line(line)

    assert found["step"] == 1299
    assert found["totalSteps"] == 20000
    assert found["loss"] == pytest.approx(1.656)


def test_a_bare_progress_bar_still_advances_the_step():
    found = training.parse_progress_line("Training:   7%|\u258b         | 1307/20000 [01:33<23:52, 13.05step/s]")

    assert found["step"] == 1307
    assert found["totalSteps"] == 20000
    # No loss on a bar; the display keeps the last one it had rather than blanking.
    assert "loss" not in found


def test_the_rounded_step_is_used_only_when_there_is_no_progress_bar():
    """Better than nothing when tqdm is not on the stream, and honestly approximate."""
    assert training.parse_progress_line("step:1K loss:1.6")["step"] == 1000
    assert training.parse_progress_line("step:250K loss:1.6")["step"] == 250_000
    assert training.parse_progress_line("step:12,500 loss:1.4e-3")["step"] == 12500
    assert training.parse_progress_line("step:12,500 loss:1.4e-3")["loss"] == pytest.approx(1.4e-3)


def test_bare_progress_bars_are_kept_out_of_the_visible_log():
    """They arrive ~13 times a second; a 40-line tail made of them shows nothing."""
    assert training.is_progress_bar_noise("Training:   7%|\u258b   | 1307/20000 [01:33<23:52, 13.05step/s]")
    assert training.is_progress_bar_noise("  Training: 100%|\u2588\u2588| 4/4 [00:01<00:00,  3.17step/s]  ")
    # Anything carrying a message of its own stays.
    assert not training.is_progress_bar_noise(
        "Training:   6%| | 1299/20000 [01:33<27:58, 11.14step/s]INFO ot_train.py:518 step:1K loss:1.656"
    )
    assert not training.is_progress_bar_noise("INFO 2026-08-19 16:01:37 ot_train.py:621 End of training")


def test_a_real_log_line_is_shown_without_the_bar_it_arrived_glued_to():
    line = (
        "Training:   6%|\u258b         | 1299/20000 [01:33<27:58, 11.14step/s]"
        "INFO 2026-08-19 16:03:36 ot_train.py:518 step:1K loss:1.656 grdn:59.609"
    )

    assert training.strip_progress_prefix(line) == (
        "INFO 2026-08-19 16:03:36 ot_train.py:518 step:1K loss:1.656 grdn:59.609"
    )
    # A line that never had one is returned unchanged.
    plain = "INFO 2026-08-19 16:01:37 ot_train.py:621 End of training"
    assert training.strip_progress_prefix(plain) == plain


def test_a_line_about_neither_contributes_nothing():
    """Absent fields stay absent rather than resetting the display to zero."""
    assert training.parse_progress_line("INFO Logs will be saved locally.") == {}


def test_the_wandb_run_url_is_picked_up_from_the_log():
    found = training.parse_progress_line(
        "wandb: \U0001f680 View run at https://wandb.ai/team/lerobot/runs/abc123."
    )

    assert found["wandbUrl"] == "https://wandb.ai/team/lerobot/runs/abc123"


def test_sync_is_a_no_op_for_the_local_host(tmp_path):
    """The local host trains from this checkout; there is nothing to copy anywhere."""
    result = training.sync_repo_to_host(tmp_path, training.local_host(tmp_path))

    assert result == {"ok": True, "skipped": True, "message": "Local host trains from this checkout."}


# ------------------------------------------------------------- machine probe ---


def test_no_torch_means_nothing_is_trainable():
    """A freshly synced training box has the code and not yet the environment.

    act and diffusion need no extra packages, so without folding torch in they came back
    "trainable" on a machine that cannot run anything at all -- and that machine is exactly
    the one an operator is most likely to be inspecting.
    """
    from tools.fr3 import probe_training_machine

    without = probe_training_machine.query_policies(torch_installed=False)["policies"]
    assert without["act"] == {"trainable": False, "missing": ["torch"]}
    assert without["smolvla"]["trainable"] is False
    assert without["smolvla"]["missing"][0] == "torch"


def test_policy_requirements_are_reported_under_their_install_names():
    """`import tree` is installed as `dm-tree`; the page shows what to type, not what to import."""
    from tools.fr3 import probe_training_machine

    policies = probe_training_machine.query_policies(torch_installed=True)["policies"]

    assert "torch" not in policies["act"]["missing"]
    groot = policies["groot"]["missing"]
    assert "tree" not in groot
    if groot:
        assert set(groot) <= {"transformers", "peft", "timm", "dm-tree", "safetensors"}


def test_an_ordinary_log_line_that_ends_in_a_bracket_is_not_mistaken_for_a_bar():
    """The filter matches tqdm's `N/M [ ... ]` shape, not "ends with a bracket"."""
    assert not training.is_progress_bar_noise("INFO checkpoint written to outputs/train/x [act]")
    assert not training.is_progress_bar_noise("INFO resolved delta_timestamps: [0.0, 0.033]")
    assert training.is_progress_bar_noise("Training: 30%|###| 6055/20000 [07:18<17:13, 13.49step/s]")
