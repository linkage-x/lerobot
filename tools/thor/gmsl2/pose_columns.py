"""Names of the tracked-pose columns of an exported dataset, old and new.

An export stores each tracked hand's TCP pose (``observation.ee_pose.<hand>.<frame>``), the target it
reached for (``action.ee_pose.<hand>.<frame>``) and its cube (``observation.cube_pose.<hand>.<frame>``)
in the dataset's world frame, the one ``info.json`` names in ``world_frame``. From 11-16 the suffix
says so: ``.world``. Exports written before carry ``.base``, a name left over from when exports were
re-expressed in an arm's base (until 11-10); what those columns hold is still whatever ``world_frame``
says. Exports already written are never renamed, so every reader goes through :func:`pose_key` and
takes both. Stdlib only: the gateway, the exporter and training all import it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

WORLD = "world"
LEGACY = "base"  # exports before 11-16
SUFFIXES = (WORLD, LEGACY)

EE_STATE = "observation.ee_pose"
EE_ACTION = "action.ee_pose"
CUBE = "observation.cube_pose"
KINDS = (EE_STATE, EE_ACTION, CUBE)


def world_key(kind: str, hand: str) -> str:
    """The column a new export writes."""
    return f"{kind}.{hand}.{WORLD}"


def pose_key(columns: Iterable[str] | Mapping[str, Any], kind: str, hand: str) -> str | None:
    """The column of ``kind`` for ``hand`` among ``columns`` (feature names or parquet columns), if any."""
    available = set(columns)
    for suffix in SUFFIXES:
        key = f"{kind}.{hand}.{suffix}"
        if key in available:
            return key
    return None


def tracked_hands(columns: Iterable[str] | Mapping[str, Any]) -> list[str]:
    """Hands with a TCP pose column (``observation.ee_pose.<hand>.world`` or ``.base``), sorted."""
    hands = set()
    for key in columns:
        if not key.startswith(EE_STATE + "."):
            continue
        hand, _, suffix = key.removeprefix(EE_STATE + ".").rpartition(".")
        if hand and "." not in hand and suffix in SUFFIXES:
            hands.add(hand)
    return sorted(hands)
