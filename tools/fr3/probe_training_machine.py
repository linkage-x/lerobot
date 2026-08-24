#!/usr/bin/env python3
"""Report what a machine can train, as JSON on stdout.

Run with the interpreter whose environment the answer is about -- the point of this
script is that "is transformers installed" has a different answer per venv, and the
Training page is choosing between machines partly on that basis.

Deliberately stdlib-only at import time so it still produces a useful report on a host
that has no torch: a machine with a GPU and no torch is a real state the page has to be
able to show, and a probe that cannot start cannot report it.

Used for both the local host and remote targets; the remote path pipes this file over
stdin, so it must stay a single self-contained file with no imports from the repo.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys

# Kept in step with pyproject's policy extras. Reported per policy so the page can say
# which are trainable here rather than making the operator read a stack trace.
POLICY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "act": (),
    "diffusion": (),
    "vqbet": (),
    "tdmpc": (),
    "sac": (),
    "pi0": ("transformers", "scipy"),
    "pi0_fast": ("transformers", "scipy"),
    "pi05": ("transformers", "scipy"),
    "smolvla": ("transformers", "num2words"),
    "xvla": ("transformers",),
    "groot": ("transformers", "peft", "timm", "tree", "safetensors"),
    "wall_x": ("transformers", "peft", "scipy", "torchdiffeq", "qwen_vl_utils"),
    "sarm": ("transformers",),
}

# Modules a *training option* needs, as opposed to a policy. LoRA is orthogonal to the policy
# choice -- every VLA here can be adapted with it -- so reporting it per policy would say the same
# thing a dozen times and still not answer "can I tick the LoRA box".
FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "lora": ("peft",),
}

# The pyproject extra that installs each policy's requirements, reported alongside them so the
# page can offer the install rather than only naming what is absent. "transformers is missing"
# and "here is the command that fixes it" are one keystroke apart for whoever wrote this file and
# a web search apart for whoever is looking at the page.
#
# pi0/pi0.5/pi0_fast map to fr3-train rather than to upstream's `pi`: this repo's extra is `pi`
# plus `peft`, and a pi0.5 operator who installs the narrower one gets to discover at wrap time
# that LoRA needs another package.
POLICY_EXTRAS: dict[str, tuple[str, ...]] = {
    "act": (),
    "diffusion": (),
    "vqbet": (),
    "tdmpc": (),
    "sac": (),
    "pi0": ("fr3-train",),
    "pi0_fast": ("fr3-train",),
    "pi05": ("fr3-train",),
    "smolvla": ("smolvla",),
    "xvla": ("xvla",),
    "groot": ("groot",),
    "wall_x": ("wallx",),
    "sarm": ("sarm",),
}

FEATURE_EXTRAS: dict[str, tuple[str, ...]] = {
    "lora": ("peft",),
}

# What the Training page's install button runs, and the environment it runs against. Reported
# because "the deps are missing" and "this machine can fix that from here" are different
# questions: a box without uv, or without the venv the setup script builds, answers no to the
# second one, and the page should say which before an operator clicks.
INSTALL_SCRIPT = "tools/fr3/install_training_deps.sh"
VENV_PATH = ".venv-fr3"
UV_MIN_VERSION = (0, 5)


# Import name -> the pip/extra name an operator would install, where they differ.
MODULE_INSTALL_NAMES = {
    "tree": "dm-tree",
    "qwen_vl_utils": "qwen-vl-utils",
}

# Floors from pyproject, for the modules where presence is the wrong question. pi0.5 resolves
# `transformers.masking_utils.create_causal_mask` and `transformers.modeling_layers` at *import*
# time; neither exists in the transformers 4.x a workstation set up before the VLA extra existed
# still has. Without this a machine reports "pi05: trainable" and the run dies in the training
# subprocess with an ImportError -- the exact stack trace this probe exists to prevent.
MODULE_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "transformers": (5, 3),
    "peft": (0, 18),
}


def module_version(name: str) -> str:
    """Installed version of `name`, or "" when it cannot be determined.

    Read from the distribution metadata rather than by importing: importing transformers costs
    seconds and can abort, which is the whole reason this probe does not do it.
    """
    import importlib.metadata

    for dist_name in (name, MODULE_INSTALL_NAMES.get(name, name)):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            return ""
    return ""


def _version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric components of a version string. `5.3.0.dev0` -> (5, 3, 0)."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def module_present(name: str) -> bool:
    """Present, and new enough where a floor is declared.

    Presence without importing: importing transformers costs seconds and can abort.
    """
    import importlib.util

    try:
        if importlib.util.find_spec(name) is None:
            return False
    except (ImportError, ValueError):
        return False
    minimum = MODULE_MIN_VERSIONS.get(name)
    if minimum is None:
        return True
    installed = _version_tuple(module_version(name))
    # An unreadable version is reported as satisfying the floor. The alternative -- calling an
    # editable or vendored install "missing" -- blocks a machine that can train, and this probe
    # is advisory: the run itself is still the thing that decides.
    return not installed or installed >= minimum


def module_requirement(name: str) -> str:
    """How an operator would name this module when installing it, floor included."""
    label = MODULE_INSTALL_NAMES.get(name, name)
    minimum = MODULE_MIN_VERSIONS.get(name)
    return f"{label}>={'.'.join(str(part) for part in minimum)}" if minimum else label


def query_gpus() -> tuple[list[dict], str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], "nvidia-smi not found"
    fields = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi failed: {type(exc).__name__}"
    if out.returncode != 0:
        return [], f"nvidia-smi exited {out.returncode}: {out.stderr.strip()[:200]}"

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 7:
            continue
        def as_int(value: str) -> int | None:
            try:
                return int(float(value))
            except ValueError:
                return None
        gpus.append(
            {
                "index": as_int(parts[0]),
                "name": parts[1],
                "memoryTotalMb": as_int(parts[2]),
                "memoryUsedMb": as_int(parts[3]),
                "utilizationPct": as_int(parts[4]),
                "temperatureC": as_int(parts[5]),
                "driverVersion": parts[6],
            }
        )
    return gpus, ""


def query_torch() -> dict:
    info: dict = {"installed": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return info
    info["installed"] = True
    info["version"] = getattr(torch, "__version__", "")
    try:
        info["cudaAvailable"] = bool(torch.cuda.is_available())
        info["cudaVersion"] = getattr(getattr(torch, "version", None), "cuda", None)
        info["deviceCount"] = int(torch.cuda.device_count()) if info["cudaAvailable"] else 0
        info["bf16Supported"] = (
            bool(torch.cuda.is_bf16_supported()) if info["cudaAvailable"] else False
        )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def query_policies(torch_installed: bool) -> dict:
    """Which policies this environment could train, and what each one is missing.

    torch is folded in rather than reported only alongside: act and diffusion need no extra
    packages, so without this they come back "trainable" on a machine that has no torch at
    all -- which is exactly the machine an operator is most likely to be inspecting, since a
    freshly synced training box has the code and not yet the environment.
    """
    modules = sorted(
        {module for reqs in POLICY_REQUIREMENTS.values() for module in reqs}
        | {module for reqs in FEATURE_REQUIREMENTS.values() for module in reqs}
    )
    present = {module: module_present(module) for module in modules}
    policies = {}
    for policy, requirements in sorted(POLICY_REQUIREMENTS.items()):
        missing = [
            module_requirement(module) for module in requirements if not present.get(module, False)
        ]
        if not torch_installed:
            missing.insert(0, "torch")
        policies[policy] = {
            "trainable": not missing,
            "missing": missing,
            # Present even when nothing is missing: torch is a base dependency, so the extras
            # that fix "act needs torch" are none at all -- syncing the project is the fix, and
            # an empty list is how the caller is told that.
            "extras": list(POLICY_EXTRAS.get(policy, ())),
        }
    return {
        "modules": present,
        "moduleVersions": {module: module_version(module) for module in modules},
        "policies": policies,
    }


def query_features() -> dict:
    """Which training *options* this environment supports, and what each one is missing.

    Reported alongside `policies` rather than folded into it: a machine with transformers and
    without peft trains pi0.5 densely and cannot train it with an adapter, and a single
    "pi05: trainable" would have to pick one of those two answers and be wrong about the other.
    """
    features = {}
    for feature, requirements in sorted(FEATURE_REQUIREMENTS.items()):
        missing = [
            module_requirement(module) for module in requirements if not module_present(module)
        ]
        features[feature] = {
            "available": not missing,
            "missing": missing,
            "extras": list(FEATURE_EXTRAS.get(feature, ())),
        }
    return features


def query_installer(repo_root: str) -> dict:
    """Whether the install button can work here, and what it would run against.

    Answered by the probe rather than by the gateway because for a remote host it is a
    question about *that* machine: the gateway's own uv and its own .venv-fr3 say nothing
    about the box the training would run on.
    """
    uv_path = shutil.which("uv") or ""
    if not uv_path:
        fallback = os.path.expanduser("~/.local/bin/uv")
        if os.access(fallback, os.X_OK):
            uv_path = fallback
    uv_version = ""
    if uv_path:
        try:
            out = subprocess.run([uv_path, "--version"], capture_output=True, text=True, timeout=15)
            if out.returncode == 0:
                parts = out.stdout.split()
                uv_version = parts[1] if len(parts) > 1 else ""
        except (OSError, subprocess.SubprocessError):
            uv_path = ""

    uv_ok = bool(uv_path) and (
        not uv_version or _version_tuple(uv_version) >= UV_MIN_VERSION
    )
    venv_python = os.path.join(repo_root, VENV_PATH, "bin", "python")
    venv_exists = os.access(venv_python, os.X_OK)
    script_present = os.path.isfile(os.path.join(repo_root, INSTALL_SCRIPT))

    if not script_present:
        reason = f"{INSTALL_SCRIPT} is not on this machine; sync the repo here first."
    elif not uv_path:
        reason = (
            "uv is not installed on this machine, and this project's environment is managed "
            "with uv. Install it from https://docs.astral.sh/uv/ there."
        )
    elif not uv_ok:
        reason = (
            f"uv {uv_version} is older than the {'.'.join(str(p) for p in UV_MIN_VERSION)} "
            "this needs. Run `uv self update` there."
        )
    else:
        # A missing environment is not a blocker, it is the first sync. The gateway adds this
        # machine's baseline extras to the plan when it builds one, so the environment that
        # appears is a superset of what was there before rather than a training-only venv the
        # recorder would then be pointed at. See install_training_deps.sh.
        reason = ""

    return {
        "canInstall": not reason,
        "reason": reason,
        "uvPath": uv_path,
        "uvVersion": uv_version,
        "venvPath": VENV_PATH,
        "venvExists": venv_exists,
        # The same button, but the operator should know which one they are pressing: extending
        # an environment takes a minute, building one downloads several GB.
        "willCreateEnvironment": not venv_exists,
        "scriptPresent": script_present,
    }


def query_disk(path: str) -> dict:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "path": path,
        "totalGb": round(usage.total / 1e9, 1),
        "freeGb": round(usage.free / 1e9, 1),
    }


def main() -> None:
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    gpus, gpu_error = query_gpus()
    torch_info = query_torch()
    report = {
        "ok": True,
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "cpuCount": os.cpu_count(),
        "repoRoot": repo_root,
        "repoRootExists": os.path.isdir(repo_root),
        "gpus": gpus,
        "gpuError": gpu_error,
        "torch": torch_info,
        "disk": query_disk(repo_root if os.path.isdir(repo_root) else "/"),
    }
    report.update(query_policies(bool(torch_info.get("installed"))))
    report["features"] = query_features()
    report["installer"] = query_installer(repo_root)
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
