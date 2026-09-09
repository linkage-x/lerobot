"""Unit tests run entirely offline, without importing a hardware SDK."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import replay_ik_trajectory_guarded as guarded


class GuardedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.csv"
        width = .05
        retreat = (49.699345 - np.sqrt(49.699345 ** 2 - 5.474953 * 50 - 50 ** 2 / 4)) / 1000
        q = (guarded.LOWER + guarded.UPPER) / 2
        self.rows = [dict(episode_index=0, frame_index=i, timestamp_s=i / 60,
                          gripper_width_m=width, tcp_retreat_m=retreat, ik_ok="True",
                          **dict(zip(guarded.JOINTS, q))) for i in range(3)]

    def write(self, fields=None):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(self.rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_no_change(self):
        self.write()
        digest = guarded.sha256(self.path)
        self.assertEqual(len(guarded.read_contact_ik(self.path)[0]), 3)
        self.assertEqual(guarded.sha256(self.path), digest)

    def test_invalid_samples_rejected_not_dropped(self):
        for field, value in [("fr3_joint6", "NaN"), ("ik_ok", "False"),
                             ("frame_index", 0), ("frame_index", 4),
                             ("timestamp_s", 0), ("gripper_width_m", .09),
                             ("gripper_width_m", -.01), ("tcp_retreat_m", .123),
                             ("fr3_joint6", 4.7)]:
            with self.subTest(field=field, value=value):
                original = self.rows[1][field]
                self.rows[1][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    guarded.read_contact_ik(self.path)
                self.rows[1][field] = original

    def test_schema_and_empty(self):
        self.write([k for k in self.rows[0] if k != "ik_ok"])
        with self.assertRaises(ValueError):
            guarded.read_contact_ik(self.path)
        fields = list(self.rows[0])
        self.rows = []
        self.write(fields)
        with self.assertRaises(ValueError):
            guarded.read_contact_ik(self.path)

    def test_wrong_or_missing_candidate(self):
        for core in [SimpleNamespace(), SimpleNamespace(_FR3_LIMITS_PATCH=guarded.PATCH_ID)]:
            with self.assertRaises(RuntimeError):
                guarded.require_candidate(core)

    def test_unknown_version_gate(self):
        core = SimpleNamespace(_FR3_LIMITS_PATCH=guarded.PATCH_ID, _FR3_CONFIGURED_NATIVE_API=1,
                               joint_limits_for_server_version=lambda v: dict(lower=guarded.LOWER, upper=guarded.UPPER))
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            guarded.require_candidate(core)

    def test_hardware_modes_always_block_before_sdk_or_file_access(self):
        for flag in ["--execute", "--start-only"]:
            with self.subTest(flag=flag), patch.dict("sys.modules", {"panda_py": None}), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(guarded, "read_contact_ik", side_effect=AssertionError("input must not be read")):
                with self.assertRaises(SystemExit) as caught:
                    guarded.main([flag])
                self.assertEqual(caught.exception.code, 9)


if __name__ == "__main__":
    unittest.main()
