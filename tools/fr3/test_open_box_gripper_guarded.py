import unittest
from unittest.mock import Mock
from open_box_gripper_guarded import open_gradually


class Fake:
    def __init__(self):
        self.mode, self.device_id = 1, 123
        self.width, self.t, self.seq = .0008, 0., 0
        self.targets = []
        self.box = Mock()
        self.box.get_mode.return_value = (0, 1)

    def clock(self): return self.t
    def sleep(self, t): self.t += t
    def wait_fresh(self, timeout):
        self.t += .01
        self.seq += 1
        return self.seq, self.width
    def send(self, target):
        self.targets.append((self.t, target))
        self.width = target
        return True


class Tests(unittest.TestCase):
    def test_monotonic_small_targets_and_verified_endpoint(self):
        b = Fake()
        r = open_gradually(b, .08806, [], clock=b.clock, sleep=b.sleep)
        self.assertTrue(r['completed'])
        self.assertAlmostEqual(r['final_measured_width_m'], .08806)
        prev = .0008
        for i, (t, target) in enumerate(b.targets):
            self.assertGreater(target, prev)
            self.assertLessEqual(target-prev, .000500001)
            if i: self.assertGreaterEqual(t-b.targets[i-1][0], .09999999)
            prev = target
        b.box.set_mode.assert_not_called()

    def test_mode_not_enabled_never_sends(self):
        b = Fake(); b.mode = 0
        with self.assertRaises(RuntimeError): open_gradually(b, .08806, [])
        self.assertEqual(b.targets, [])
        b.box.set_mode.assert_not_called()

    def test_closing_and_out_of_range_rejected(self):
        for target in (0., .10, float('nan')):
            b = Fake()
            with self.assertRaises(ValueError): open_gradually(b, target, [], clock=b.clock, sleep=b.sleep)
            self.assertEqual(b.targets, [])

    def test_rejected_command_not_retried(self):
        b = Fake(); b.send = Mock(return_value=False)
        with self.assertRaises(RuntimeError): open_gradually(b, .08806, [], clock=b.clock, sleep=b.sleep)
        self.assertEqual(b.send.call_count, 1)

    def test_lost_feedback_stops(self):
        b = Fake(); calls = 0
        original = b.wait_fresh
        def sample(timeout):
            nonlocal calls
            calls += 1
            if calls > 22: raise TimeoutError('lost')
            return original(timeout)
        b.wait_fresh = sample
        with self.assertRaises(TimeoutError): open_gradually(b, .08806, [], clock=b.clock, sleep=b.sleep)
        self.assertEqual(len(b.targets), 1)


if __name__ == '__main__': unittest.main(verbosity=2)
