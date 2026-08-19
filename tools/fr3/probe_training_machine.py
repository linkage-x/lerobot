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

# Import name -> the pip/extra name an operator would install, where they differ.
MODULE_INSTALL_NAMES = {
    "tree": "dm-tree",
    "qwen_vl_utils": "qwen-vl-utils",
}


def module_present(name: str) -> bool:
    """Presence without importing: importing transformers costs seconds and can abort."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


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
    modules = sorted({module for reqs in POLICY_REQUIREMENTS.values() for module in reqs})
    present = {module: module_present(module) for module in modules}
    policies = {}
    for policy, requirements in sorted(POLICY_REQUIREMENTS.items()):
        missing = [
            MODULE_INSTALL_NAMES.get(module, module)
            for module in requirements
            if not present.get(module, False)
        ]
        if not torch_installed:
            missing.insert(0, "torch")
        policies[policy] = {"trainable": not missing, "missing": missing}
    return {"modules": present, "policies": policies}


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
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
