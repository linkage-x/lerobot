"""No hardware: input, planner, execution-failure handling contract tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import native_arm as app


class Offline(unittest.TestCase):
    def test_provenance(self):
        app.verify_inputs()

    def test_bad_shape_before_planner(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                app.prepare(None,[[0]*6]*2,.01,.088,Path(d))

    def test_duplicates_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                app.prepare(None,[[0]*7]*2,.01,.088,Path(d))

    def test_unhealthy_start_does_not_continue(self):
        class Controller:
            def get_time(self):return 0.
        class Core:
            @staticmethod
            def NativeJointTrajectoryController(*args):return Controller()
        class Panda:
            def start_controller_guarded(self,c):raise RuntimeError('Reflex / no auto recovery')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'Reflex'):
                app.execute(Panda(),Core(),None,np.zeros(7),{},Path(d))
            result=app.json.loads((Path(d)/'execution.json').read_text())
            self.assertFalse(result['completed'])
            self.assertFalse(result['motion_started'])
            self.assertEqual(result['gripper_commands_sent'],0)

    def test_nominal_frame_counts_and_identity(self):
        source=app.json.loads((app.SOURCE/'timed_plan.json').read_text())
        for ep,n in zip(source['episodes'],[571,420]):
            q=app.knots(ep)
            self.assertEqual(q.shape,(n,8))
            indexes=[]
            for k,(a,b) in enumerate(app.chunk_indices(n,200)):
                indexes.extend(range(a+(k>0),b))
            self.assertEqual(indexes,list(range(n)))


if __name__=='__main__':unittest.main()
