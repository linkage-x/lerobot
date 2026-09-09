"""No devices: regression tests for the native planner adapter and source map."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from benchmark_native import knots,chunk_indices
from prepare_native import interpolation_index,VELOCITY,ACCELERATION

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'build'))
import _native_plan_audit as core


class NativeTests(unittest.TestCase):
    def test_chunks_cover_every_waypoint_without_gap(self):
        chunks=list(chunk_indices(571,200))
        self.assertEqual(chunks,[(0,200),(199,399),(398,571)])
        self.assertEqual(set(i for a,b in chunks for i in range(a,b)),set(range(571)))

    def test_no_device_api(self):
        for name in ('Robot','Panda','Box','execute','recover','start_controller'):
            self.assertFalse(hasattr(core,name))

    def test_duplicate_not_silently_drop_width_dwell(self):
        with self.assertRaises(ValueError):
            core.Path([np.zeros(7),np.zeros(7)],0.)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            core.Path([np.zeros(7),np.full(7,np.nan)],.02)

    def test_true_query_derivatives(self):
        q=[np.zeros(7),np.array([.05,.01,0,0,0,0,0])]
        plan=core.Trajectory(core.Path(q,0.),VELOCITY,ACCELERATION,.001)
        for t in [.0003,.0043,.0133]:
            state=plan.state(t)
            h=1e-7
            qv=(plan.state(t+h)[:7]-plan.state(t-h)[:7])/(2*h)
            va=(plan.state(t+h)[7:14]-plan.state(t-h)[7:14])/(2*h)
            np.testing.assert_allclose(qv,state[7:14],atol=1e-8)
            np.testing.assert_allclose(va,state[14:21],atol=1e-7)

    def test_arc_anchors_and_last_frame(self):
        q=np.array([[0,0,0,0,0,0,0],[.1,0,0,0,0,0,0],
                    [.1,.1,0,0,0,0,0],[.2,.1,0,0,0,0,0]])
        plan=core.Trajectory(core.Path(q,.02),VELOCITY,ACCELERATION,.001)
        anchors=np.asarray(plan.waypoint_path_positions())
        self.assertTrue(np.all(np.diff(anchors)>0))
        end=plan.state(plan.duration())
        self.assertAlmostEqual(anchors[-1],end[21],places=9)
        np.testing.assert_allclose(interpolation_index(anchors,anchors),np.arange(4))
        np.testing.assert_allclose(end[:7],q[-1],atol=1e-9)

    def test_unchanged_position_plan(self):
        # Compare the compiled adapter to the original extension's saved
        # query results, including rounded paths. Timing and position remain
        # unchanged; only derivative/arc metadata is corrected by this adapter.
        benchmark=ROOT/'benchmark_v1'
        for ep in (0,1):
            source=np.load(benchmark/f'source_{ep}.npz')['q']
            for dev in (0.,.0001,.02):
                path=benchmark/f'ep{ep}_dev{dev:g}_chunk0.npz'
                original=np.load(path)
                plan=core.Trajectory(core.Path(source[:200],dev),VELOCITY,ACCELERATION,.001)
                self.assertAlmostEqual(plan.duration(),original['t'][-1],places=9)
                idx=np.linspace(0,len(original['t'])-1,73).astype(int)
                current=np.array([plan.state(float(original['t'][i]))[:7] for i in idx])
                np.testing.assert_allclose(current,original['q'][idx],atol=1e-10)


if __name__=='__main__':
    unittest.main()
