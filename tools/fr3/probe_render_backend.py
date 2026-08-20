#!/usr/bin/env python3
"""Report whether this machine can render offscreen, and say precisely what is missing.

The workstation gateway renders every MuJoCo view offscreen: it exports ``MUJOCO_GL=egl``
and starts the sim teleop with ``--no-viewer``, streaming frames into the web UI. That path
needs the GPU's DRM render node, not an X server -- so "is there a display" is the wrong
question to gate a deploy on, and answering it produces an error that points away from the
actual cause.

What actually breaks it: on a desktop distribution ``/dev/dri/renderD*`` is
``root:render rw-rw----`` and non-members reach it only through the ACL logind grants to
whoever holds the console. Deploy over ssh with nobody logged in graphically, and the account
has no GPU access at all -- which looks like an EGL driver failure and is a group membership.

Exits 0 when offscreen rendering works. Prints one ``reason=`` line either way, so a caller
can put the cause in front of the operator instead of a symptom.
"""

from __future__ import annotations

import glob
import grp
import json
import os
import pwd
import sys


def render_nodes() -> list[str]:
    return sorted(glob.glob("/dev/dri/renderD*"))


def can_open_render_node() -> tuple[bool, str]:
    nodes = render_nodes()
    if not nodes:
        return False, "no /dev/dri/renderD* render node exists (no GPU, or no DRM driver loaded)"
    errors = []
    for node in nodes:
        try:
            with open(node, "rb"):
                return True, ""
        except OSError as exc:
            errors.append(f"{node}: {exc.strerror}")
    return False, "; ".join(errors)


def missing_render_groups() -> list[str]:
    """Which of the GPU groups this user is not in, among those that exist here."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return []
    mine = {grp.getgrgid(gid).gr_name for gid in os.getgroups() if _group_name(gid)}
    missing = []
    for name in ("render", "video"):
        try:
            grp.getgrnam(name)
        except KeyError:
            continue
        if name not in mine:
            missing.append(name)
    return missing


def _group_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return ""


def egl_device_display_works() -> tuple[bool, str]:
    """Whether MuJoCo's own headless path can get a device display.

    Uses mujoco's function rather than a reimplementation: the point is to answer for the
    code that will actually run, including its choice of EGL platform extension.
    """
    # Both, and before the import: MUJOCO_GL is what mujoco reads, PYOPENGL_PLATFORM is what
    # PyOpenGL binds at import time. Setting only the first leaves PyOpenGL free to pick GLX
    # whenever a DISPLAY happens to be set, which answers a different question than the one
    # the gateway will ask.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        import OpenGL.EGL as egl
        from mujoco.egl import create_initialized_egl_device_display
    except Exception as exc:  # noqa: BLE001 - an unimportable stack is a real answer here
        return False, f"cannot import the EGL rendering stack: {type(exc).__name__}: {exc}"
    try:
        display = create_initialized_egl_device_display()
    except Exception as exc:  # noqa: BLE001
        return False, f"EGL device display raised {type(exc).__name__}: {exc}"
    if display == egl.EGL_NO_DISPLAY:
        return False, "EGL reported no usable device display"
    return True, ""


def main() -> int:
    node_ok, node_error = can_open_render_node()
    egl_ok, egl_error = ("", "")
    if node_ok:
        egl_ok, egl_error = egl_device_display_works()
    else:
        egl_ok = False
        egl_error = "not attempted (no accessible render node)"

    missing = missing_render_groups()
    report = {
        "offscreenRenderOk": bool(egl_ok),
        "renderNodeOk": node_ok,
        "renderNodeError": node_error,
        "eglError": egl_error,
        "display": os.environ.get("DISPLAY", ""),
        "missingGroups": missing,
        "user": pwd.getpwuid(os.getuid()).pw_name,
    }

    user = report["user"]
    if egl_ok:
        reason = "offscreen GPU rendering available"
    elif not node_ok and missing:
        # The actionable case, and the one a display check reports as something else entirely.
        fix = " && ".join(f"sudo gpasswd -a {user} {group}" for group in missing)
        reason = (
            f"{user} cannot open the GPU render node ({node_error}); it is not in the "
            f"{'/'.join(missing)} group, and this account has no graphical session whose "
            f"logind ACL would grant access. Fix: {fix} (then reconnect)"
        )
    elif not node_ok:
        reason = f"cannot open the GPU render node ({node_error})"
    else:
        reason = f"render node is accessible but EGL is not usable ({egl_error})"
    report["reason"] = reason

    if "--json" in sys.argv:
        json.dump(report, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"reason={reason}")
    return 0 if egl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
