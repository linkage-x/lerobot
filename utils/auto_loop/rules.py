from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    reason: str
    changes: Dict[str, Any]  # flat-ish mapping of config edits


def _get(d: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur) if cur is not None else default
    except Exception:
        return default


def decide(summary: Dict[str, Any]) -> List[Decision]:
    """Turn a single-run metric summary into a list of proposed config edits.

    The summary dict is expected to follow utils/auto_loop/common.summarize_series_dict output.
    Returns a list (multiple independent variants can be spawned from the same run).
    """
    out: List[Decision] = []

    # Prefer canonical keys written by WandBLogger (train/*). Fall back to legacy
    # aliases when present to stay compatible with older summaries.
    def _first_existing(*candidates: str, which: str) -> Optional[float]:
        for key in candidates:
            val = _get(summary, key, which)
            if val is not None:
                return val
        return None

    # OT mass / diagonal
    pi_sum_last = _first_existing("train/ot_pi_sum", "train/ot_ot_pi_sum", which="last")
    pi_sum_best = _first_existing("train/ot_pi_sum", "train/ot_ot_pi_sum", which="best")

    # OT per-term costs: prefer action label when available; otherwise any image term
    cost_first = _first_existing(
        "train/ot_cost/action_lbl",
        "train/ot_cost/img_third_person",
        "train/ot_cost/img_side",
        "train/ot_cost/img_ee",
        "train/ot_ot_cost/observation.state",
        which="first",
    )
    cost_last = _first_existing(
        "train/ot_cost/action_lbl",
        "train/ot_cost/img_third_person",
        "train/ot_cost/img_side",
        "train/ot_cost/img_ee",
        "train/ot_ot_cost/observation.state",
        which="last",
    )

    # Core train metrics
    l1_first = _get(summary, "train/l1_loss", "first")
    l1_last = _get(summary, "train/l1_loss", "last")
    grad_last = _get(summary, "train/grad_norm", "last")

    # Heuristics
    # 1) pi too low -> increase regularization & temporal slack
    if pi_sum_last is not None and pi_sum_last < 1e-3:
        out.append(
            Decision(
                reason="pi_sum too low; increase reg and temporal slack",
                changes={
                    "ot.loss_config.reg@mul": 1.6,
                    "ot.loss_config.tau_src@set": 0.5,
                    "ot.loss_config.tau_tgt@set": 0.5,
                    "ot.window_size@max": 20,
                },
            )
        )

    # 2) pi high but cost not reducing -> reduce label weight and lambda_ot
    if (
        pi_sum_last is not None
        and pi_sum_last > 0.3
        and cost_first is not None
        and cost_last is not None
        and (cost_last - cost_first) / max(cost_first, 1e-6) > -0.10  # less than 10% drop
    ):
        out.append(
            Decision(
                reason="pi high but cost not decreasing; reduce label weight and lambda_ot",
                changes={
                    "ot.loss_config.features.*.weight_label@mul": 0.5,
                    "ot.lambda_ot@add": -0.05,
                },
            )
        )

    # 3) L1 plateau -> small LR/batch tweaks
    if l1_first is not None and l1_last is not None:
        drop = (l1_last - l1_first) / max(l1_first, 1e-6)
        if drop > -0.02:  # less than 2% improvement
            out.append(
                Decision(
                    reason="L1 plateau; try modest LR up and slightly larger batch",
                    changes={
                        "policy.optimizer_lr@mul": 1.4,
                        "optimizer.lr@mul": 1.4,
                        "batch_size@add": 4,
                    },
                )
            )

    # 4) Stability guard
    if grad_last is not None and grad_last > 100.0:
        out.append(
            Decision(
                reason="High grad_norm; strengthen regularization or reduce LR",
                changes={
                    "ot.loss_config.reg@mul": 1.3,
                    "policy.optimizer_lr@mul": 0.7,
                    "optimizer.lr@mul": 0.7,
                },
            )
        )

    # Default gentle OT variant if nothing triggered
    if not out:
        out.append(
            Decision(
                reason="Default gentle OT variant",
                changes={
                    "ot.loss_config.reg@mul": 1.2,
                    "ot.window_size@max": 20,
                },
            )
        )

    return out


def apply_changes(cfg: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a limited set of dotted edits with simple ops.

    Supported ops via suffix:
      - @set: set value (floats/ints/str)
      - @mul: multiply numeric (if missing, ignored)
      - @add: add numeric
    Dotted key can include wildcard for features list index: features.*.weight_label
    """
    import copy

    out = copy.deepcopy(cfg)

    def _assign(obj: Any, key_path: List[str], op: str, value: Any) -> None:
        if not key_path:
            return
        k = key_path[0]
        is_last = len(key_path) == 1

        if k == "*":
            if isinstance(obj, list):
                for i in range(len(obj)):
                    _assign(obj[i], key_path[1:], op, value)
            return

        if isinstance(obj, dict):
            if is_last:
                cur = obj.get(k)
                if op == "set":
                    obj[k] = value
                elif op == "mul":
                    try:
                        obj[k] = float(cur) * float(value)
                    except Exception:
                        pass
                elif op == "add":
                    try:
                        obj[k] = float(cur) + float(value)
                    except Exception:
                        pass
                return
            nxt = obj.get(k)
            if nxt is None:
                # create nested dicts if needed
                nxt = {}
                obj[k] = nxt
            _assign(nxt, key_path[1:], op, value)
        elif isinstance(obj, list):
            try:
                idx = int(k)
            except Exception:
                return
            if 0 <= idx < len(obj):
                _assign(obj[idx], key_path[1:], op, value)

    for dotted, val in changes.items():
        if "@" in dotted:
            dotted_key, op = dotted.split("@", 1)
        else:
            dotted_key, op = dotted, "set"
        parts = dotted_key.split(".")
        _assign(out, parts, op, val)

    # Safety: keep use_policy_training_preset as-is; user configs rely on it
    return out
