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


def test_resume_training_names_checkpoint_and_drops_fresh_finetune_knobs():
    checkpoint = "/repo/outputs/train/L4_full48_holdout22_40/checkpoints/030000"
    argv = training.build_train_argv(
        host=_remote_host(),
        view_root="/repo/view",
        repo_id="local/v",
        job_name="v__pi05",
        policy="pi05",
        steps=40000,
        batch_size=2,
        num_workers=4,
        save_freq=2000,
        log_freq=100,
        device="cuda",
        use_amp=True,
        policy_config='{"optimizer_lr": 1e-4}',
        pretrained_path="lerobot/pi05_base",
        lora_enabled=True,
        lora_r=32,
        lora_alpha=32,
        lora_target_modules="all-linear",
        resume=True,
        resume_checkpoint=checkpoint,
        wandb_enabled=False,
        wandb_project="",
        wandb_entity="",
    )

    assert "--resume" in argv
    assert argv[argv.index("--resume-checkpoint") + 1] == checkpoint
    assert argv[argv.index("--steps") + 1] == "40000"
    assert "--pretrained-path" not in argv
    assert "--lora" not in argv
    assert "--policy-config" not in argv
    assert "--use-amp" not in argv


def test_resume_training_requires_a_checkpoint_path():
    with pytest.raises(training.TrainingError, match="Resume training needs a checkpoint path"):
        training.build_train_argv(
            host=_remote_host(),
            view_root="/repo/view",
            repo_id="local/v",
            job_name="v__pi05",
            policy="pi05",
            steps=10,
            batch_size=2,
            num_workers=4,
            save_freq=5,
            log_freq=1,
            device="cuda",
            use_amp=False,
            policy_config="",
            resume=True,
            resume_checkpoint="",
            wandb_enabled=False,
            wandb_project="",
            wandb_entity="",
        )


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
    assert env["WANDB_DIR"] == str(tmp_path / "outputs" / "wandb" / "runs")
    assert env["WANDB_CACHE_DIR"] == str(tmp_path / "outputs" / "wandb" / "cache")
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "outputs" / "wandb" / "xdg-cache")
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
    assert "mkdir -p outputs/wandb/runs outputs/wandb/cache outputs/wandb/config outputs/wandb/data outputs/wandb/xdg-cache" in remote_script
    assert "WANDB_CACHE_DIR=outputs/wandb/cache" in remote_script
    assert "HF_HUB_CACHE=/home/tele/Models" in remote_script
    assert "HF_HUB_OFFLINE=1" in remote_script
    assert "TRANSFORMERS_OFFLINE=1" in remote_script
    # Nothing to read the key from, so nothing tries to.
    assert ".wandb_key" not in remote_script


def test_a_local_launch_uses_the_tele_model_cache_when_present(tmp_path, monkeypatch):
    model_cache = tmp_path / "Models"
    model_cache.mkdir()
    monkeypatch.setattr(training, "TELE_MODEL_CACHE", model_cache)

    _, env = training.build_launch_command(
        tmp_path, training.local_host(tmp_path), ["{python}", "train.py"], wandb_key=""
    )

    assert env is not None
    assert env["HF_HUB_CACHE"] == str(model_cache)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


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
    assert without["act"] == {"trainable": False, "missing": ["torch"], "extras": []}
    assert without["smolvla"]["trainable"] is False
    assert without["smolvla"]["missing"][0] == "torch"


def _finetune_argv(**overrides) -> list[str]:
    common = {
        "host": _remote_host(),
        "view_root": "/repo/view",
        "repo_id": "local/v",
        "job_name": "v__pi05",
        "policy": "pi05",
        "steps": 10,
        "batch_size": 2,
        "num_workers": 4,
        "save_freq": 5,
        "log_freq": 1,
        "device": "cuda",
        "use_amp": False,
        "policy_config": "",
        "wandb_enabled": False,
        "wandb_project": "",
        "wandb_entity": "",
    }
    common.update(overrides)
    return training.build_train_argv(**common)


def test_a_dense_run_asks_for_neither_a_base_model_nor_an_adapter():
    argv = _finetune_argv()

    assert "--pretrained-path" not in argv
    assert "--lora" not in argv


def test_finetuning_from_a_base_checkpoint_names_it_on_the_command_line():
    argv = _finetune_argv(pretrained_path="lerobot/pi05_base")

    assert argv[argv.index("--pretrained-path") + 1] == "lerobot/pi05_base"
    # Weights only. Without --lora the run is a dense finetune of that checkpoint.
    assert "--lora" not in argv


def test_finetuning_accepts_an_absolute_local_base_checkpoint_path():
    """The tele model cache is a local directory, not a Hugging Face repo id."""
    base = "/home/tele/Models/pi05_base"
    argv = _finetune_argv(pretrained_path=base, lora_enabled=True)

    assert argv[argv.index("--pretrained-path") + 1] == base
    assert "--lora" in argv


@pytest.mark.parametrize("value", ("/", "/home/tele/Models/pi05 base", "/home/tele/$(id)", "~/Models/pi05_base"))
def test_pretrained_path_rejects_unsafe_or_ambiguous_local_paths(value: str):
    with pytest.raises(training.TrainingError):
        training.validate_pretrained_path(value)


def test_lora_carries_its_rank_and_leaves_the_targets_to_the_policy():
    argv = _finetune_argv(pretrained_path="lerobot/pi05_base", lora_enabled=True, lora_r=32)

    assert "--lora" in argv
    assert argv[argv.index("--lora-r") + 1] == "32"
    # pi0.5's own default target set is the tuned one; sending nothing is how it survives.
    assert "--lora-target-modules" not in argv

    targeted = _finetune_argv(
        pretrained_path="lerobot/pi05_base",
        lora_enabled=True,
        lora_target_modules="all-linear",
    )
    assert targeted[targeted.index("--lora-target-modules") + 1] == "all-linear"


def test_lora_alpha_is_sent_only_when_the_operator_names_one():
    # Strength is alpha / r, so a rank with no alpha is only half a specification. Sending
    # nothing lets the training script track the rank; sending 0 would pin the scaling to 0.
    default = _finetune_argv(pretrained_path="lerobot/pi05_base", lora_enabled=True, lora_r=64)
    assert "--lora-alpha" not in default

    pinned = _finetune_argv(
        pretrained_path="lerobot/pi05_base", lora_enabled=True, lora_r=64, lora_alpha=8
    )
    assert pinned[pinned.index("--lora-alpha") + 1] == "8"


def test_lora_without_a_base_model_is_refused_before_the_rsync():
    """On a remote host the training script's own refusal is a line in a log nobody is reading."""
    with pytest.raises(training.TrainingError, match="no base checkpoint"):
        _finetune_argv(lora_enabled=True)


def test_a_base_checkpoint_of_the_wrong_shape_is_refused():
    """Caught here as a typo, with a message, rather than as a 404 from the Hub mid-run."""
    with pytest.raises(training.TrainingError, match="Hugging Face repo id"):
        _finetune_argv(pretrained_path="lerobot/pi05_base; rm -rf /")
    with pytest.raises(training.TrainingError, match="Hugging Face repo id"):
        _finetune_argv(pretrained_path="$(whoami)")


def test_a_target_regex_survives_validation():
    """pi0.5's own default target spec is a regex; a shell-metacharacter filter would reject it.

    What makes these safe is `shlex.quote` in build_launch_command, not this function -- which
    only rejects what is never part of a module name and always means a bad paste.
    """
    spec = r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj)"
    argv = _finetune_argv(
        pretrained_path="lerobot/pi05_base", lora_enabled=True, lora_target_modules=spec
    )
    assert argv[argv.index("--lora-target-modules") + 1] == spec

    with pytest.raises(training.TrainingError, match="control characters"):
        _finetune_argv(
            pretrained_path="lerobot/pi05_base",
            lora_enabled=True,
            lora_target_modules="q_proj\nv_proj",
        )


def test_policy_requirements_are_reported_under_their_install_names():
    """`import tree` is installed as `dm-tree`; the page shows what to type, not what to import."""
    from tools.fr3 import probe_training_machine

    policies = probe_training_machine.query_policies(torch_installed=True)["policies"]

    assert "torch" not in policies["act"]["missing"]
    groot = policies["groot"]["missing"]
    assert "tree" not in groot
    if groot:
        # Names carry their floor where one is declared, because that is what an operator has to
        # type: `pip install transformers` on a box that already has 4.49 is a no-op.
        assert set(groot) <= {"transformers>=5.3", "peft>=0.18", "timm", "dm-tree", "safetensors"}


def test_a_module_that_is_present_but_too_old_counts_as_missing(monkeypatch):
    """pi0.5 imports `transformers.masking_utils` at module import; 4.x does not have it.

    Presence is the wrong question here, and answering it made the page say "pi05: trainable"
    on a machine where the run dies at step 0 inside a subprocess.
    """
    import importlib.util

    from tools.fr3 import probe_training_machine

    # Every module is installed as far as the probe can see, so the floor is the only thing
    # left that can make an answer False.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    monkeypatch.setattr(probe_training_machine, "module_version", lambda name: "4.49.0")
    assert not probe_training_machine.module_present("transformers")
    # Modules without a declared floor are unaffected -- any scipy is a scipy.
    assert probe_training_machine.module_present("scipy")

    monkeypatch.setattr(probe_training_machine, "module_version", lambda name: "5.3.1")
    assert probe_training_machine.module_present("transformers")

    # An unreadable version is not evidence of an old one: an editable install still trains.
    monkeypatch.setattr(probe_training_machine, "module_version", lambda name: "")
    assert probe_training_machine.module_present("transformers")


def test_a_version_floor_is_compared_numerically_not_lexically():
    """`4.49.0` must not beat `5.3` the way a string comparison would have it."""
    from tools.fr3 import probe_training_machine

    assert probe_training_machine._version_tuple("4.49.0") < (5, 3)
    assert probe_training_machine._version_tuple("5.3.0.dev0") == (5, 3, 0)
    assert probe_training_machine._version_tuple("5.10.1") > (5, 3)
    # Nothing numeric to read: reported as unknown, which the caller treats as satisfying.
    assert probe_training_machine._version_tuple("main") == ()
    assert probe_training_machine.module_requirement("transformers") == "transformers>=5.3"
    assert probe_training_machine.module_requirement("tree") == "dm-tree"


def test_lora_availability_is_reported_apart_from_the_policy_list(monkeypatch):
    """A machine can train pi0.5 densely and not with an adapter; one flag cannot say both."""
    from tools.fr3 import probe_training_machine

    monkeypatch.setattr(probe_training_machine, "module_present", lambda name: name != "peft")

    features = probe_training_machine.query_features()

    assert features["lora"]["available"] is False
    assert features["lora"]["missing"] == ["peft>=0.18"]


def test_an_ordinary_log_line_that_ends_in_a_bracket_is_not_mistaken_for_a_bar():
    """The filter matches tqdm's `N/M [ ... ]` shape, not "ends with a bracket"."""
    assert not training.is_progress_bar_noise("INFO checkpoint written to outputs/train/x [act]")
    assert not training.is_progress_bar_noise("INFO resolved delta_timestamps: [0.0, 0.033]")
    assert training.is_progress_bar_noise("Training: 30%|###| 6055/20000 [07:18<17:13, 13.49step/s]")


# ------------------------------------------------------- dependency install ---


def test_every_extra_the_probe_offers_is_one_pyproject_actually_has():
    """The probe names the fix; uv is the one that has to recognise it.

    These two lists are edited months apart -- a policy is added to the probe, an extra is
    renamed in pyproject -- and the failure that drift produces is an install button that runs
    a sync, waits, and comes back with "unknown extra" on a machine the operator had to walk to.
    """
    import tomllib

    from tools.fr3 import probe_training_machine

    with open("pyproject.toml", "rb") as handle:
        declared = set(tomllib.load(handle)["project"]["optional-dependencies"])

    offered = {
        extra
        for extras in (
            *probe_training_machine.POLICY_EXTRAS.values(),
            *probe_training_machine.FEATURE_EXTRAS.values(),
        )
        for extra in extras
    }
    assert offered
    assert offered <= declared, f"not in pyproject: {sorted(offered - declared)}"
    # And every one of them survives the validation the gateway puts them through, which is
    # the other end of the same journey.
    assert training.validate_extras(sorted(offered)) == sorted(offered)


def test_every_policy_the_probe_judges_also_says_how_to_fix_it():
    """A policy present in one map and absent from the other reports a gap with no remedy."""
    from tools.fr3 import probe_training_machine

    assert set(probe_training_machine.POLICY_EXTRAS) == set(
        probe_training_machine.POLICY_REQUIREMENTS
    )
    assert set(probe_training_machine.FEATURE_EXTRAS) == set(
        probe_training_machine.FEATURE_REQUIREMENTS
    )
    # A policy that needs no extra module needs no extra: act is fixed by having an environment
    # at all, and offering "install ()" would be a button that does nothing.
    for policy, requirements in probe_training_machine.POLICY_REQUIREMENTS.items():
        if not requirements:
            assert probe_training_machine.POLICY_EXTRAS[policy] == ()


def test_the_extras_are_reported_next_to_what_is_missing():
    from tools.fr3 import probe_training_machine

    policies = probe_training_machine.query_policies(torch_installed=True)["policies"]

    assert policies["pi05"]["extras"] == ["fr3-train"]
    assert policies["act"]["extras"] == []
    assert probe_training_machine.query_features()["lora"]["extras"] == ["peft"]


def test_an_extra_name_that_could_reach_a_shell_is_refused():
    """This string ends up in an ssh command line, so it is filtered rather than quoted."""
    for bad in ("fr3-train; rm -rf /", "$(id)", "../../etc", "fr3 train", "-x", "`id`", "a|b"):
        with pytest.raises(training.TrainingError):
            training.validate_extras([bad])
    # Blank entries are dropped rather than refused: an empty plan is the base-dependency sync,
    # and a page that sends one empty string means the same thing as one that sends none.
    assert training.validate_extras(["", "  "]) == []


def test_the_same_extra_asked_for_twice_is_installed_once():
    """pi0.5 and LoRA are both fixed by fr3-train, and the page can ask for both at once."""
    assert training.validate_extras(["fr3-train", "peft", "fr3-train"]) == ["fr3-train", "peft"]
    # Order is the operator's, not alphabetical: it is what the logged command will read like.
    assert training.validate_extras(["smolvla", "fr3-train"]) == ["smolvla", "fr3-train"]


def test_asking_for_no_extra_at_all_is_a_plan_not_an_error():
    """"act cannot train here: missing torch" has no extra to name and a real fix.

    torch, accelerate and wandb are base dependencies of this project, so `uv sync` with no
    extra is the whole remedy. Refusing an empty list turned the button off on the one machine
    that most needed it -- the freshly synced box with the code and no environment.
    """
    assert training.validate_extras([]) == []
    assert training.validate_extras(None) == []


def test_the_recording_workstation_never_syncs_without_the_recorders_own_extras(tmp_path):
    """The workstation shares one environment between the recorder and the trainer.

    gateway.py hands the recorder `.venv-fr3/bin/python` as soon as that path exists, so an
    environment built for training alone would stop recording on the next Connect -- and even
    when it already exists, resolving without those extras invites a shared dependency being
    moved under the recorder's feet.
    """
    local = training.local_host(tmp_path)
    remote = training.TrainingHost(
        id="user@box:/srv/lerobot",
        label="box",
        kind="remote",
        sshTarget="user@box",
        repoDir="/srv/lerobot",
    )

    assert training.baseline_extras(local, "workstation") == (
        "fr3-workstation-teleop",
        "fr3-host",
    )
    # Nothing records on a remote training box, so its environment is free to be the training
    # one and only that.
    assert training.baseline_extras(remote, "workstation") == ()
    assert training.baseline_extras(local, "thor") == ()


def test_a_sync_with_no_extras_is_a_command_the_script_accepts(tmp_path):
    """The base-dependency case has to survive the whole chain, not just validation."""
    script = tmp_path / training.INSTALL_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    command, _ = training.build_install_command(tmp_path, training.local_host(tmp_path), [])
    assert command == ["bash", str(script)]

    host = training.TrainingHost(
        id="user@box:/srv/lerobot",
        label="box",
        kind="remote",
        sshTarget="user@box",
        repoDir="/srv/lerobot",
    )
    remote, _ = training.build_install_command(tmp_path, host, [])
    # No stray trailing space: the far side is a shell, and `bash script ''` is a different
    # command from `bash script`.
    assert remote[-1] == "cd /srv/lerobot && exec bash tools/fr3/install_training_deps.sh"


def test_the_local_install_runs_the_script_out_of_this_checkout(tmp_path):
    script = tmp_path / training.INSTALL_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    command, env = training.build_install_command(
        tmp_path, training.local_host(tmp_path), ["fr3-train"]
    )

    assert command == ["bash", str(script), "fr3-train"]
    assert env is not None and env["PYTHONUNBUFFERED"] == "1"


def test_an_install_with_no_script_to_run_says_so_rather_than_failing_in_bash(tmp_path):
    with pytest.raises(training.TrainingError, match="Install script missing"):
        training.build_install_command(tmp_path, training.local_host(tmp_path), ["fr3-train"])


def test_a_remote_install_runs_in_the_hosts_own_checkout(tmp_path):
    """Not piped over stdin like the probe: an install needs the repo it is installing from.

    The probe has to work on a machine with no checkout -- that is half of what it reports --
    but `uv sync` reads pyproject.toml, so requiring the sync first turns a puzzling failure
    into an obvious one.
    """
    host = training.TrainingHost(
        id="user@box:/srv/lerobot",
        label="box",
        kind="remote",
        sshTarget="user@box",
        repoDir="/srv/lerobot",
    )

    command, env = training.build_install_command(tmp_path, host, ["fr3-train", "peft"])

    assert command[0] == "ssh"
    assert command[-2] == "user@box"
    assert command[-1] == (
        "cd /srv/lerobot && exec bash tools/fr3/install_training_deps.sh fr3-train peft"
    )
    # No environment override: a remote command carries nothing of this machine's environment.
    assert env is None


def test_a_machine_with_no_environment_is_offered_one_rather_than_an_explanation(tmp_path):
    """No .venv-fr3 is the first sync, not a blocker -- but the page has to say which it is.

    Building takes several GB and a while; extending takes a minute. Same button, and an
    operator who cannot tell which one they pressed will assume it has hung.
    """
    from tools.fr3 import probe_training_machine

    (tmp_path / "tools" / "fr3").mkdir(parents=True)
    (tmp_path / probe_training_machine.INSTALL_SCRIPT).write_text("#!/bin/bash\n", encoding="utf-8")

    report = probe_training_machine.query_installer(str(tmp_path))

    assert report["canInstall"] is True
    assert report["venvExists"] is False
    assert report["willCreateEnvironment"] is True

    # A machine the repo has never been synced to is a different answer, and the fix is a
    # different button: Sync code now, not Install.
    bare = probe_training_machine.query_installer(str(tmp_path / "nowhere"))
    assert bare["canInstall"] is False
    assert bare["scriptPresent"] is False
    assert "sync the repo" in bare["reason"]


def test_an_old_uv_is_reported_as_the_blocker_it_is(tmp_path, monkeypatch):
    """`--inexact` is the flag this depends on, and a 0.4 would fail without saying that."""
    from tools.fr3 import probe_training_machine

    (tmp_path / "tools" / "fr3").mkdir(parents=True)
    (tmp_path / probe_training_machine.INSTALL_SCRIPT).write_text("#!/bin/bash\n", encoding="utf-8")
    venv_bin = tmp_path / probe_training_machine.VENV_PATH / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\necho 'uv 0.4.1'\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(probe_training_machine.shutil, "which", lambda name: str(fake_uv))

    report = probe_training_machine.query_installer(str(tmp_path))

    assert report["canInstall"] is False
    assert report["uvVersion"] == "0.4.1"
    assert "uv self update" in report["reason"]

    # Everything in place: the page is allowed to offer the button.
    fake_uv.write_text("#!/bin/sh\necho 'uv 0.10.2'\n", encoding="utf-8")
    ready = probe_training_machine.query_installer(str(tmp_path))
    assert ready == {
        "canInstall": True,
        "reason": "",
        "uvPath": str(fake_uv),
        "uvVersion": "0.10.2",
        "venvPath": ".venv-fr3",
        "venvExists": True,
        "willCreateEnvironment": False,
        "scriptPresent": True,
    }


def test_no_uv_is_the_one_thing_this_cannot_work_around(tmp_path, monkeypatch):
    """The environment is a uv project environment; a pip install into one is invisible to it."""
    from tools.fr3 import probe_training_machine

    (tmp_path / "tools" / "fr3").mkdir(parents=True)
    (tmp_path / probe_training_machine.INSTALL_SCRIPT).write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(probe_training_machine.shutil, "which", lambda name: None)
    monkeypatch.setattr(probe_training_machine.os.path, "expanduser", lambda path: "/nonexistent")

    report = probe_training_machine.query_installer(str(tmp_path))

    assert report["canInstall"] is False
    assert "uv" in report["reason"]
    assert "docs.astral.sh/uv" in report["reason"]
