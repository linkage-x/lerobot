"""Align measured FR3 telemetry to Thor's camera exposure clock, with validity."""
from bisect import bisect_left
import math

FR3_FEATURE_NAMES = {
    "observation.fr3.q": [f"joint{i}.pos" for i in range(1, 8)],
    "observation.fr3.dq": [f"joint{i}.vel" for i in range(1, 8)],
    "observation.fr3.tau_J": [f"joint{i}.torque_Nm" for i in range(1, 8)],
    "observation.fr3.tau_ext_hat_filtered": [f"joint{i}.external_torque_Nm" for i in range(1, 8)],
    "observation.fr3.O_T_EE": [f"column{c}.row{r}" for c in range(4) for r in range(4)],
    "observation.fr3.O_F_ext_hat_K": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
    "observation.fr3.tcp": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"],
    "action": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"],
    "fr3.timestamps": ["source_monotonic_s", "thor_monotonic_s", "receiver_monotonic_s", "uncertainty_s"],
    "fr3.valid": ["valid"],
    "fr3.action_valid": ["valid"],
    "fr3.control_command_success_rate": ["rate"],
}


def fr3_features() -> dict:
    return {key: {"dtype": "float64" if key == "fr3.timestamps" else "float32",
                  "shape": [len(names)], "names": names}
            for key, names in FR3_FEATURE_NAMES.items()}


def aligned_fr3_rows(samples: list[dict], frame_times: list[float | None], *,
                     t0_mono_s: float, max_skew_s: float = 0.025) -> list[dict]:
    ordered = sorted(samples, key=lambda sample: sample["thor_sample_monotonic_s"])
    times = [s["thor_sample_monotonic_s"] for s in ordered]
    rows = []
    for relative_s in frame_times:
        row = {key: [0.0] * len(names) for key, names in FR3_FEATURE_NAMES.items()}
        if relative_s is not None and math.isfinite(relative_s) and times:
            target_s = t0_mono_s + relative_s
            index = bisect_left(times, target_s)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
            sample = ordered[min(candidates, key=lambda i: abs(times[i] - target_s))]
            uncertainty = sample["clock_uncertainty_s"]
            valid = abs(sample["thor_sample_monotonic_s"] - target_s) + uncertainty <= max_skew_s
            row["fr3.timestamps"] = [sample["sample_monotonic_s"], sample["thor_sample_monotonic_s"],
                                     sample["receiver_monotonic_s"], uncertainty]
            if valid:
                for field in ("q", "dq", "tau_J", "tau_ext_hat_filtered", "O_T_EE", "O_F_ext_hat_K"):
                    row[f"observation.fr3.{field}"] = sample[field]
                row["observation.fr3.tcp"] = sample["measured_tcp"]
                row["fr3.valid"] = [1.0]
                row["fr3.control_command_success_rate"] = [sample["control_command_success_rate"]]
                if sample.get("commanded_ee") is not None:
                    row["action"] = [*sample["commanded_ee"], sample["gripper_command"]]
                    row["fr3.action_valid"] = [1.0]
        rows.append(row)
    return rows


def append_fr3_columns(pa, table, samples: list[dict], frame_times: list[float | None], *, t0_mono_s: float):
    # Missing camera timestamps are explicitly invalid, never replaced by pickup time.
    targets = [frame_times[i] if i < len(frame_times) else None for i in range(table.num_rows)]
    rows = aligned_fr3_rows(samples, targets, t0_mono_s=t0_mono_s)
    for key, names in FR3_FEATURE_NAMES.items():
        dtype = pa.float64() if key == "fr3.timestamps" else pa.float32()
        flat = [v for row in rows for v in row[key]]
        table = table.append_column(key, pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype), len(names)))
    return table
