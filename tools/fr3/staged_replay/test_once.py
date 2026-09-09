"""No hardware: assert unattended-by-prompt sequencing retains all stop gates."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import operator_replay
from plans import start_check
from runtime import require_stable_gripper, save, sha


class OnceTests(unittest.TestCase):
    def enter(self, stack, root, args, runner):
        save(root / 'manifest.json', {})
        stack.enter_context(patch.object(operator_replay, '__file__', str(root / 'operator_replay.py')))
        stack.enter_context(patch('runtime.verify_bundle', return_value={}))
        stack.enter_context(patch('sys.argv', ['operator_replay.py'] + args))
        stdin = stack.enter_context(patch('sys.stdin'))
        stdin.isatty.return_value = False
        user_input = stack.enter_context(patch('builtins.input', side_effect=AssertionError('Unexpected prompt')))
        stack.enter_context(patch.object(operator_replay.subprocess, 'run', runner))
        stack.enter_context(redirect_stdout(StringIO()))
        return user_input

    def test_sequence_has_no_input_or_retry(self):
        for codes, expected_calls in [([0, 0, 0], 3), ([1], 1), ([0, 1], 2)]:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                runner = Mock(side_effect=[SimpleNamespace(returncode=n) for n in codes])
                ui = self.enter(stack, Path(tmp), ['0', 'full-replay', '--execute'], runner)
                self.assertEqual(operator_replay.main(), codes[-1])
                self.assertEqual(runner.call_count, expected_calls)
                phases = ['start-check', 'start-only', 'replay'][:expected_calls]
                for call, phase in zip(runner.call_args_list, phases):
                    self.assertEqual(call.args[0][-2:], [phase, '--execute'])
                ui.assert_not_called()

    def test_explicit_flag_required_without_tty(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            runner = Mock()
            self.enter(stack, Path(tmp), ['0', 'full-replay'], runner)
            with self.assertRaises(RuntimeError): operator_replay.main()
            runner.assert_not_called()

    def test_prepare_and_collision_failure_never_execute(self):
        for prepare_rc, collision_ok in [(1, True), (0, False), (0, True)]:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                actions = []
                def runner(command, **kwargs):
                    mode = command[2]
                    actions.append(mode)
                    if mode == 'prepare' and prepare_rc == 0:
                        output = Path(command[command.index('--output') + 1])
                        output.mkdir()
                        q = [-.15, -.41, .067, -2.4, .045, 2.06, .44]
                        plan = dict(episode=0, phase='start-check',
                                    geometry=dict(sampled_geometry_pass=collision_ok),
                                    execution_plan=start_check(q, .08806, q))
                        save(output / 'prepared_plan.json', plan)
                    elif mode == 'execute':
                        rp = Path(command[command.index('--operator-release') + 1])
                        prepared = Path(command[command.index('--prepared') + 1])
                        receipt = json.loads(rp.read_text())
                        self.assertEqual(receipt['authorization_source'], 'explicit_cli_execute')
                        self.assertFalse(receipt['automatic_mode_switching_allowed'])
                        self.assertEqual(receipt['prepared_sha256'], sha(prepared))
                        operator_replay.validate_release(root, rp, json.loads(prepared.read_text()), sha(prepared))
                    return SimpleNamespace(returncode=prepare_rc if mode == 'prepare' else 0)
                ui = self.enter(stack, root, ['0', 'start-check', '--execute'], runner)
                if prepare_rc or not collision_ok:
                    with self.assertRaises(RuntimeError): operator_replay.main()
                    self.assertEqual(actions, ['prepare'])
                else:
                    self.assertEqual(operator_replay.main(), 0)
                    self.assertEqual(actions, ['prepare', 'execute'])
                ui.assert_not_called()

    def test_actual_width_must_stabilize(self):
        readings = [dict(width_m=.08806)] * 20
        require_stable_gripper(readings)
        for bad in (readings[:19], readings[:19] + [dict(width_m=.08)],
                    readings[:19] + [dict(width_m=float('nan'))]):
            with self.assertRaises(RuntimeError): require_stable_gripper(bad)


if __name__ == '__main__': unittest.main(verbosity=2)
