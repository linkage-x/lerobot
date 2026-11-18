"""May a policy drive this arm now? The maintainer's bounded, per-arm unlock (stdlib only).

Policy inference that moves an FR3 is locked unless a maintainer who is present unlocks one arm for a
bounded time by writing, on the machine that runs inference, ``<repo>/outputs/fr3_infer_real_unlock.json``:

    {"arm": "right", "by": "<maintainer>", "reason": "<why, which checkpoint>",
     "created_utc": "2026-11-10T08:00:00Z", "expires_utc": "2026-11-10T12:00:00Z"}

at most MAX_SPAN_H hours, and deletes it afterwards. It names the arm, so an unlock for one arm never
drives the other. Agents never write it (AGENTS.md section 4). Read by the old absolute-pose runtime
(fr3_act_infer_real_runtime) only: since 11-18 the relative-TCP execution layer (rollout_execute) is started
from the Rollout page by a person who confirms that the arm moves and a hand is on the e-stop.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

PATH = Path("outputs/fr3_infer_real_unlock.json")
MAX_SPAN_H = 12.0


class RealExecutionLockedError(RuntimeError):
    """Inference would drive the arm, and no maintainer has unlocked that for this arm."""


def _utc(value: Any, field: str) -> datetime.datetime:
    if not isinstance(value, str):
        raise RealExecutionLockedError(f"unlock file: {field} must be a UTC time like 2026-11-10T08:00:00Z")
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
    except ValueError as exc:
        raise RealExecutionLockedError(f"unlock file: {field}={value!r} is not like 2026-11-10T08:00:00Z") from exc


def read(
    arm: str,
    *,
    repo_root: Path | str,
    locked: str,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """The unlock record for ``arm`` now, or RealExecutionLockedError starting with ``locked``."""
    path = Path(repo_root) / PATH
    if not path.is_file():
        raise RealExecutionLockedError(f"{locked}; there is no {PATH}.")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RealExecutionLockedError(f"{locked}; {PATH} is unreadable ({exc}).") from exc
    if not isinstance(record, dict):
        raise RealExecutionLockedError(f"{locked}; {PATH} is not a JSON object.")
    if record.get("arm") != arm:
        raise RealExecutionLockedError(f"{locked}; the unlock is for arm {record.get('arm')!r}, this run drives the {arm} arm.")
    for field in ("by", "reason"):
        if not str(record.get(field) or "").strip():
            raise RealExecutionLockedError(f"{locked}; the unlock does not say {field!r}.")
    created = _utc(record.get("created_utc"), "created_utc")
    expires = _utc(record.get("expires_utc"), "expires_utc")
    now = now or datetime.datetime.now(datetime.UTC)
    span_h = (expires - created).total_seconds() / 3600.0
    if not 0.0 < span_h <= MAX_SPAN_H:
        raise RealExecutionLockedError(f"{locked}; the unlock spans {span_h:.1f} h, it may span at most {MAX_SPAN_H:g} h.")
    if not created <= now < expires:
        raise RealExecutionLockedError(
            f"{locked}; the unlock is valid {record['created_utc']} .. {record['expires_utc']}, not now."
        )
    return record
