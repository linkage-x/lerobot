"""Synthetic-state regression tests only. Never instantiate Robot or Panda."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from panda_py import _core as core
from arm_runtime import passive_plan,make_controller,save
from plans import transit

ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'timed_plan.json').read_text())
chain=source['chain']


def small():
    start=np.asarray(source['episodes'][0]['coeff_descending_unit_interval'])[0,-1,:7]
    return passive_plan(transit(start,.088,start+.002,.088),.088)


def begin(ep=None):
    c=make_controller(core,small() if ep is None else ep,chain)
    target=c.evaluate(0.)[:7]
    c._offline_start(target,np.zeros(7),c.fk(target))
    return c


class Tests(unittest.TestCase):
    def test_no_gripper_api_or_import(self):
        c=begin()
        for name in ('feed_gripper','acknowledge_gripper_command','send','set_mode'):
            self.assertFalse(hasattr(c,name))
        self.assertNotIn('box_sdk',sys.modules)
        tree=ast.parse((ROOT/'arm_runtime.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('box_transport',imports)

    def test_changing_gripper_width_rejected(self):
        ep=small();ep['coeff_descending_unit_interval'][0][4][7]=1e-5
        with self.assertRaises(ValueError):make_controller(core,ep,chain)

    def test_arm_coefficients_preserved(self):
        for ep in source['episodes']:
            p=passive_plan(ep,.088)
            np.testing.assert_array_equal(np.asarray(ep['coeff_descending_unit_interval'])[:,:,:7],
                                          np.asarray(p['coeff_descending_unit_interval'])[:,:,:7])
            self.assertEqual(p['times_s'],ep['times_s'])

    def test_complete_without_gripper_feedback(self):
        for ep in source['episodes']:
            p=passive_plan(ep,.088);c=begin(p)
            for k in range(int(np.ceil(p['duration_s']*1000))+2):
                t=k*.001;target=c.evaluate(t)[:7];dq=c.evaluate(t,1)[:7]
                result=c._offline_step(t,target,dq,c.fk(target),1)
                if c.completed():break
            self.assertTrue(c.completed());self.assertEqual(c.fault_code(),0)
            rows=c.telemetry();self.assertEqual(rows.shape[1],48)
            self.assertTrue(np.isnan(rows[:,39]).all())
            self.assertTrue(np.isfinite(np.delete(rows,39,axis=1)).all())
            with self.assertRaises(RuntimeError):c._offline_start(target,np.zeros(7),c.fk(target))

    def test_callback_gap_stops(self):
        c=begin();q=c.evaluate(.02)[:7]
        with self.assertRaises(RuntimeError):c._offline_step(.02,q,np.zeros(7),c.fk(q),20)
        self.assertEqual(c.fault_code(),5)

    def test_external_abort_stops(self):
        c=begin();c.abort_external();q=c.evaluate(.001)[:7]
        with self.assertRaises(RuntimeError):c._offline_step(.001,q,np.zeros(7),c.fk(q),1)
        self.assertEqual(c.fault_code(),6)

    def test_joint_tracking_stops(self):
        c=begin()
        with self.assertRaises(RuntimeError):
            for i in range(1,110):
                t=i*.001;q=c.evaluate(t)[:7].copy();q[0]+=.06
                c._offline_step(t,q,np.zeros(7),c.fk(q),1)
        self.assertEqual(c.fault_code(),11)

    def test_tcp_inconsistency_stops(self):
        c=begin();q=c.evaluate(.001)[:7];pose=c.fk(q);pose[2,3]+=.003
        with self.assertRaises(RuntimeError):c._offline_step(.001,q,np.zeros(7),pose,1)
        self.assertEqual(c.fault_code(),2)

    def test_overspeed_stops(self):
        ep=small();ep['times_s'][-1]=.02;ep['duration_s']=.02
        c=begin(ep);q=c.evaluate(.001)[:7]
        with self.assertRaises(RuntimeError):c._offline_step(.001,q,np.zeros(7),c.fk(q),1)
        self.assertEqual(c.fault_code(),10)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    save(ROOT/'offline_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        physical_motion_commands_sent=0,robot_instances_created=0,gripper_sdk_loaded=False,
        limitations='Synthetic ideal tracking only; real motion has not been tested.'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
