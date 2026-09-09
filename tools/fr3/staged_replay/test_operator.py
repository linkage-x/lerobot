"""Pure tests; never instantiate a robot or SDK."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock
import numpy as np
from box_transport import BoxTransport
from hinge_contacts import inside_pin_region
from operator_replay import predecessor, validate_release
from plans import start_check, validate
from replay_timed_candidate import evaluate
from runtime import save, sha


class OperatorTests(unittest.TestCase):
    def test_pin_contacts_scoped(self):
        self.assertTrue(inside_pin_region([[.0099, .0065, 0.], [-.0100, 0., 0.]]))
        for points in ([], [[.0102, 0., 0.]], [[0., .0068, 0.]], [[0., float('nan'), 0.]], [1., 2., 3.]):
            self.assertFalse(inside_pin_region(points))
        self.assertFalse(inside_pin_region([[0., 0., 0.]], capped=True))

    def test_short_check_is_slow_small_and_constant_width(self):
        q = np.array([-.15, -.41, .067, -2.4, .045, 2.06, .44])
        ep = start_check(q, .088, q + [.1, .7, .4, .1, .6, 1.8, -.2])
        validate(ep)
        self.assertEqual(ep['duration_s'], 5.)
        np.testing.assert_allclose(evaluate(ep, 0.), np.r_[q, .088])
        self.assertLessEqual(np.max(abs(evaluate(ep, 5.)[:7] - q)), .03000001)
        for t in np.linspace(0., 5., 101):
            self.assertAlmostEqual(evaluate(ep, t)[7], .088)
        for t in (0., 5.):
            for d in (1, 2):
                np.testing.assert_allclose(evaluate(ep, t, d), np.zeros(8), atol=1e-10)

    def test_mode_gate_never_changes_mode(self):
        b = BoxTransport('/none', 123, '127.0.0.1')
        b.box = Mock(); b.mode = 0
        b.box.get_mode.return_value = (0, 0)
        with self.assertRaises(RuntimeError): b.require_control_mode()
        b.box.set_mode.assert_not_called()
        b.allow_commands = True
        with self.assertRaises(RuntimeError): b.require_control_mode()
        b.box.get_mode.return_value = (0, 1)
        b.require_control_mode()
        b.box.set_mode.assert_not_called()
        self.assertEqual(b.mode, 1)
        b.close()
        b.box = Mock(); b.mode = 0
        b.box.get_mode.return_value = (2, 1)
        with self.assertRaises(RuntimeError): b.require_control_mode()
        self.assertEqual(b.mode, 0)
        b.box.get_mode.return_value = (0, 9)
        with self.assertRaises(RuntimeError): b.require_control_mode()
        b.box.set_mode.assert_not_called()

    def test_scoped_receipt_and_stage_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'manifest.json', {})
            self.assertIsNone(predecessor(root, 0, 'start-check'))
            with self.assertRaises(RuntimeError): predecessor(root, 0, 'replay')
            d = root / 'logs/operator_test'; (d / 'execute').mkdir(parents=True)
            receipt = dict(approved=True, prepared_sha256='digest',
                bundle_manifest_sha256=sha(root / 'manifest.json'), phase='start-check', episode=0,
                observer_confirmed=True, gripper_ready_mode_required=1,
                automatic_mode_switching_allowed=False, authorization_source='explicit_cli_execute',
                created_unix_s=time.time(), predecessor=None)
            rp = d / 'receipt.json'; save(rp, receipt)
            prepared = dict(episode=0, phase='start-check')
            validate_release(root, rp, prepared, 'digest')
            with self.assertRaises(RuntimeError): validate_release(root, rp, prepared, 'wrong')
            with self.assertRaises(RuntimeError): predecessor(root, 0, 'start-only')
            save(d / 'execute/execution_result.json', dict(completed=True, fault_code=0))
            save(d / 'execute/report.json', dict(execution=dict(completed=True)))
            prev = predecessor(root, 0, 'start-only')
            with self.assertRaises(RuntimeError): predecessor(root, 0, 'replay')
            later = dict(receipt, phase='start-only', predecessor=prev)
            sp = root / 'later.json'; save(sp, later)
            validate_release(root, sp, dict(episode=0, phase='start-only'), 'digest')
            with self.assertRaises(RuntimeError): validate_release(root, sp, dict(episode=1, phase='start-only'), 'digest')
            later['created_unix_s'] -= 700
            stale = root / 'stale.json'; save(stale, later)
            with self.assertRaises(RuntimeError): validate_release(root, stale, dict(episode=0, phase='start-only'), 'digest')


if __name__ == '__main__':
    unittest.main(verbosity=2)
